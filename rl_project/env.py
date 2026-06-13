# rl_project/env.py — 3-link planar arm + Gymnasium RL environment
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import CANVAS_BOUNDS, TARGET_REACH_THRESHOLD, MAX_STEPS, LINK_LENGTHS, DT, INIT_ANGLES, PATTERN_CONFIGS
from patterns import get_pattern


class PlanarArm3Link:
    """3-link planar robotic arm with simplified dynamics."""
    
    def __init__(self, links=LINK_LENGTHS, dt=DT):
        self.links = np.array(links, dtype=np.float32)
        self.dt = dt
        self.q = np.zeros(3, dtype=np.float32)
        self.dq = np.zeros(3, dtype=np.float32)
        self._prev_a = np.zeros(3, dtype=np.float32)
    
    def fk(self, q=None):
        """Forward kinematics: returns (joint_positions [4,2], end_effector [2])."""
        if q is None:
            q = self.q
        l1, l2, l3 = self.links
        a1, a2, a3 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
        pts = np.zeros((4, 2), dtype=np.float32)
        pts[0] = [0, 0]
        pts[1] = pts[0] + [l1*np.cos(a1), l1*np.sin(a1)]
        pts[2] = pts[1] + [l2*np.cos(a2), l2*np.sin(a2)]
        pts[3] = pts[2] + [l3*np.cos(a3), l3*np.sin(a3)]
        return pts, pts[3]
    
    def step(self, tau):
        """Apply torques and simulate one dt."""
        tau = np.clip(tau, -1, 1)
        I = np.array([0.045, 0.03, 0.012], dtype=np.float32)
        d = np.array([0.3, 0.3, 0.3], dtype=np.float32)  # stable damping for smooth trajectories
        a = tau/I - d*self.dq
        self._prev_a = a.copy()
        self.dq += a*self.dt
        self.q += self.dq*self.dt
        self.q = np.clip(self.q, -np.pi, np.pi)
        self.dq = np.clip(self.dq, -8, 8)
    
    def jerk(self):
        return float(np.linalg.norm(self._prev_a))
    
    def reset(self, q=None):
        if q is None:
            q = np.array(INIT_ANGLES, dtype=np.float32)
        self.q = q.copy()
        self.dq = np.zeros(3, dtype=np.float32)
        self._prev_a = np.zeros(3, dtype=np.float32)


class RoboticArmDrawingEnv(gym.Env):
    """RL environment: 3-link arm draws target patterns on 2D canvas."""
    
    metadata = {"render_modes": ["human", "rgb_array", "save"], "render_fps": 30}
    
    def __init__(self, target_pattern='circle', render_mode=None, max_steps=MAX_STEPS):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.bounds = CANVAS_BOUNDS
        self.thresh = TARGET_REACH_THRESHOLD
        
        self.target = get_pattern(target_pattern)
        self.pname = target_pattern
        self.arm = PlanarArm3Link()
        # Vertex indices: only for triangle pattern (sharp corners)
        if target_pattern == 'triangle':
            n_side = PATTERN_CONFIGS.get(target_pattern, {}).get('n', 10)
            self._vertex_indices = {0, n_side, 2*n_side, len(self.target)-1}
        else:
            self._vertex_indices = set()  # no vertices for smooth curves
        
        # Observation: 15 dims
        ol = np.array([-1]*6 + [-10]*3 + [-0.8]*4 + [0, 0], dtype=np.float32)
        oh = np.array([1]*6 + [10]*3 + [0.8]*4 + [2, 1], dtype=np.float32)
        self.observation_space = spaces.Box(ol, oh, dtype=np.float32)
        # Action: 3 dims [τ1, τ2, τ3] — no pen, always drawing
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        
        self._reset_state()
    
    def _reset_state(self):
        self.step_cnt = 0
        self.visited = set()
        self.path = []
        self.rewards = []
        self._prev_ee = None
        self._next_target = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        q = np.array(INIT_ANGLES, dtype=np.float32)
        q += self.np_random.uniform(-0.02, 0.02, 3)
        self.arm.reset(q)
        _, ee = self.arm.fk()
        self._prev_ee = ee.copy()
        return self._obs(), {}
    
    def _closest_edge_dir(self, ee):
        """Direction vector of the closest target path segment to EE."""
        min_d = float('inf')
        best_dir = np.array([1.0, 0.0], dtype=np.float32)
        for i in range(len(self.target) - 1):
            a, b = self.target[i], self.target[i+1]
            ab = b - a
            ap = ee - a
            t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8)
            t = np.clip(t, 0.0, 1.0)
            closest = a + t * ab
            d = float(np.linalg.norm(ee - closest))
            if d < min_d:
                min_d = d
                best_dir = ab / max(np.linalg.norm(ab), 1e-8)
        return best_dir

    def _path_progress(self, ee):
        """Return (fraction 0-1) of how far along target path the EE is."""
        best_d = float('inf')
        best_dist = 0.0
        for i in range(len(self.target) - 1):
            a, b = self.target[i], self.target[i+1]
            ab = b - a
            ap = ee - a
            t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8)
            t = np.clip(t, 0.0, 1.0)
            closest = a + t * ab
            d = float(np.linalg.norm(ee - closest))
            if d < best_d:
                best_d = d
                best_dist = self._cumlen[i] + t * (self._cumlen[i+1] - self._cumlen[i])
        return best_dist / self._total_path_len

    def _obs(self):
        q, dq = self.arm.q, self.arm.dq
        _, ee = self.arm.fk()
        if self._next_target < len(self.target):
            t = self.target[self._next_target]; d = np.linalg.norm(ee - t)
        else:
            t = self.target[-1]; d = 0.0
        p = self._next_target / max(len(self.target), 1)
        return np.array([np.cos(q[0]), np.sin(q[0]), np.cos(q[1]), np.sin(q[1]),
                         np.cos(q[2]), np.sin(q[2]), dq[0], dq[1], dq[2],
                         ee[0], ee[1], t[0], t[1], d, p], dtype=np.float32)
    
    def _segment_deviation(self, ee):
        """Distance to the current target segment only (not the whole path)."""
        if self._next_target >= len(self.target) or self._next_target == 0:
            return 0.0
        a = self.target[self._next_target - 1]  # previous (visited)
        b = self.target[self._next_target]       # current target
        return self._point_to_seg_dist(ee, a, b)
    
    def _point_to_seg_dist(self, p, a, b):
        ab = b - a; ap = p - a
        t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8)
        t = np.clip(t, 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))
    
    def step(self, action):
        self.step_cnt += 1
        self.arm.step(action)
        _, ee = self.arm.fk()
        if self._on_canvas(ee):
            self.path.append(ee.copy())
        hit = False
        if self._next_target < len(self.target):
            if np.linalg.norm(ee - self.target[self._next_target]) < self.thresh:
                self.visited.add(self._next_target)
                self._next_target += 1
                hit = True
        r = 0.0
        # Segment-guidance: reward proximity to current edge segment (stronger)
        if self._next_target > 0 and self._next_target < len(self.target):
            seg_d = self._segment_deviation(ee)
            r += np.exp(-seg_d * 12.0) * 1.0
        if hit:
            r += 3.0
            # Vertex precision bonus: distance-DEPENDENT (not binary!)
            # 20*exp(-d*30): 0.01m→14.8, 0.05m→4.5, 0.10m→1.0
            if self._next_target in self._vertex_indices:
                vtx_d = float(np.linalg.norm(ee - self.target[self._next_target - 1]))
                r += 30.0 * np.exp(-vtx_d * 50.0)
        # Continuous vertex proximity: per-step gradient pulling EE toward vertex
        if self._next_target < len(self.target) and self._next_target in self._vertex_indices:
            vtx_d = float(np.linalg.norm(ee - self.target[self._next_target]))
            r += np.exp(-vtx_d * 15.0) * 1.0
        if self._next_target >= len(self.target): r += 50.0
        if self._collision(): r -= 10.0
        if not self._on_canvas(ee): r -= 1.0
        r -= 0.002
        self.rewards.append(r)
        done = self._next_target >= len(self.target) or self._collision()
        trunc = self.step_cnt >= self.max_steps
        self._prev_ee = ee.copy()
        return self._obs(), r, done, trunc, {'visited': len(self.visited), 'total': len(self.target)}
    
    def _on_canvas(self, p):
        return self.bounds[0] <= p[0] <= self.bounds[1] and self.bounds[0] <= p[1] <= self.bounds[1]
    
    def _collision(self):
        return bool(np.any(np.abs(self.arm.q) > np.pi * 0.95))
    
    def close(self):
        pass
