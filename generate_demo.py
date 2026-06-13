# generate_demo.py — Generate 1-minute demo video of trained robotic arm drawing triangle
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from stable_baselines3 import PPO

# Add rl_project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl_project'))
from env import RoboticArmDrawingEnv, PlanarArm3Link
from config import RESULT_DIR, LINK_LENGTHS, INIT_ANGLES
from patterns import get_pattern

MODEL_PATH = os.path.join(RESULT_DIR, 'models', 'ppo_final.zip')
VECNORM_PATH = os.path.join(RESULT_DIR, 'models', 'vecnorm_final.pkl')
OUTPUT_VIDEO = os.path.join(RESULT_DIR, 'demo_triangle.mp4')
OUTPUT_GIF = os.path.join(RESULT_DIR, 'demo_triangle.gif')

# Load model and normalization
print("Loading model...")
model = PPO.load(MODEL_PATH)
with open(VECNORM_PATH, 'rb') as f:
    norm_data = pickle.load(f)
obs_rms = norm_data.obs_rms

# Run one episode and record everything
print("Running evaluation episode...")
env = RoboticArmDrawingEnv(target_pattern='triangle', render_mode=None, max_steps=600)
target = get_pattern('triangle')

obs, _ = env.reset()
done = False

# Record: step, joint angles, EE positions, target_idx
records = []
while not done:
    obs_norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
    a, _ = model.predict(obs_norm, deterministic=True)
    q = env.arm.q.copy()
    dq = env.arm.dq.copy()
    _, ee = env.arm.fk()
    records.append({
        'q': q, 'dq': dq, 'ee': ee.copy(),
        'next_target': env._next_target,
        'step': env.step_cnt,
    })
    obs, r, terminated, truncated, _ = env.step(a)
    done = terminated or truncated

# Add final state
q = env.arm.q.copy()
_, ee = env.arm.fk()
records.append({
    'q': q, 'dq': env.arm.dq.copy(), 'ee': ee.copy(),
    'next_target': env._next_target,
    'step': env.step_cnt,
})

# Draw the full drawn path
drawn_path = np.array(env.path) if env.path else np.zeros((0, 2))
env.close()

print(f"Episode: {len(records)} steps, {len(env.visited)}/{len(target)} points visited")
print(f"Trajectory points: {len(drawn_path)}")

# ---- Animation ----
# Layout: left = arm + canvas, right = metrics panel
fig = plt.figure(figsize=(16, 8), facecolor='#0d1117')
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.05)

ax_main = fig.add_subplot(gs[0, 0])
ax_main.set_facecolor('#0d1117')
ax_main.set_xlim(-0.65, 0.65)
ax_main.set_ylim(-0.65, 0.65)
ax_main.set_aspect('equal')
ax_main.set_xticks([])
ax_main.set_yticks([])
for spine in ax_main.spines.values():
    spine.set_visible(False)

# Right panel: mini subplots for info
ax_info = fig.add_subplot(gs[0, 1])
ax_info.set_facecolor('#0d1117')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
ax_info.set_xticks([])
ax_info.set_yticks([])
for spine in ax_info.spines.values():
    spine.set_visible(False)

fig.suptitle('PPO-Trained 3-Link Arm: Triangle Drawing', 
             fontsize=16, fontweight='bold', color='#e6edf3', y=0.98)

# Pre-compute arm geometry for all frames
arm_links = np.array(LINK_LENGTHS)

def get_arm_lines(q):
    """Return (joint_positions, line_segments) for given joint angles."""
    l1, l2, l3 = arm_links
    a1, a2, a3 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
    pts = np.array([
        [0, 0],
        [l1*np.cos(a1), l1*np.sin(a1)],
        [l1*np.cos(a1)+l2*np.cos(a2), l1*np.sin(a1)+l2*np.sin(a2)],
        [l1*np.cos(a1)+l2*np.cos(a2)+l3*np.cos(a3), 
         l1*np.sin(a1)+l2*np.sin(a2)+l3*np.sin(a3)],
    ])
    return pts

# Initialize plot elements
arm_line, = ax_main.plot([], [], 'o-', color='#58a6ff', lw=3, ms=8, 
                          markerfacecolor='#79c0ff', markeredgecolor='#1f6feb', zorder=5)
ee_dot, = ax_main.plot([], [], 'o', color='#ff7b72', ms=12, 
                        markerfacecolor='#ff7b72', markeredgecolor='#da3633', zorder=6)
ee_dot.set_markeredgewidth(2)

# Target triangle (ghost)
target_plot, = ax_main.plot(target[:, 0], target[:, 1], '--', color='#f0f6fc', 
                             lw=1.5, alpha=0.4, zorder=2)
# Target points
target_pts = ax_main.scatter(target[:, 0], target[:, 1], s=8, color='#f0f6fc', 
                              alpha=0.3, zorder=2)

# Current target highlight
cur_target_dot, = ax_main.plot([], [], 'o', color='#d2a8ff', ms=14, 
                                markerfacecolor='none', markeredgewidth=2, zorder=4)

# Drawn trajectory (progressive)
traj_line, = ax_main.plot([], [], '-', color='#7ee787', lw=2, alpha=0.8, zorder=3)

# Trajectory scatter (visited target markers)
visited_scatter = ax_main.scatter([], [], s=20, color='#7ee787', alpha=0.9, zorder=3)

# Vertex highlight rings
vertex_ring1, = ax_main.plot([], [], 'o', color='#ffa657', ms=18, 
                              markerfacecolor='none', markeredgewidth=2.5, zorder=4)
vertex_ring2, = ax_main.plot([], [], 'o', color='#ffa657', ms=18, 
                              markerfacecolor='none', markeredgewidth=2.5, zorder=4)
vertex_ring3, = ax_main.plot([], [], 'o', color='#ffa657', ms=18, 
                              markerfacecolor='none', markeredgewidth=2.5, zorder=4)

# Canvas border
border = plt.Rectangle((-0.6, -0.6), 1.2, 1.2, fill=False, color='#30363d', lw=1.5, ls='-')
ax_main.add_patch(border)

# Joint dots
joint_dots = [ax_main.plot([], [], 'o', color='#79c0ff', ms=8, zorder=5)[0] for _ in range(4)]

# Info text elements
title_text = ax_info.text(0.05, 0.92, 'CONTROL PANEL', fontsize=13, fontweight='bold', 
                           color='#e6edf3', family='monospace')
step_text = ax_info.text(0.05, 0.83, '', fontsize=11, color='#8b949e', family='monospace')
target_text = ax_info.text(0.05, 0.76, '', fontsize=11, color='#8b949e', family='monospace')
visited_text = ax_info.text(0.05, 0.69, '', fontsize=11, color='#8b949e', family='monospace')
ee_text = ax_info.text(0.05, 0.60, '', fontsize=11, color='#8b949e', family='monospace')
dtw_text = ax_info.text(0.05, 0.51, '', fontsize=11, color='#8b949e', family='monospace')
reward_text = ax_info.text(0.05, 0.42, '', fontsize=11, color='#8b949e', family='monospace')

# Progress bar background
progress_bg = plt.Rectangle((0.05, 0.30), 0.9, 0.04, fill=True, color='#21262d', 
                             transform=ax_info.transAxes, zorder=1)
ax_info.add_patch(progress_bg)
progress_bar = plt.Rectangle((0.05, 0.30), 0.0, 0.04, fill=True, color='#7ee787', 
                              transform=ax_info.transAxes, zorder=2)
ax_info.add_patch(progress_bar)
progress_label = ax_info.text(0.5, 0.275, '', fontsize=9, color='#e6edf3', 
                               ha='center', family='monospace')

# Legend
legend_items = [
    (ax_info.text(0.05, 0.20, '●', fontsize=14, color='#58a6ff', family='monospace'),
     ax_info.text(0.09, 0.20, 'Robot Arm', fontsize=10, color='#8b949e', family='monospace')),
    (ax_info.text(0.05, 0.15, '●', fontsize=14, color='#7ee787', family='monospace'),
     ax_info.text(0.09, 0.15, 'Drawn Path', fontsize=10, color='#8b949e', family='monospace')),
    (ax_info.text(0.05, 0.10, '--', fontsize=14, color='#f0f6fc', family='monospace'),
     ax_info.text(0.09, 0.10, 'Target', fontsize=10, color='#8b949e', family='monospace')),
    (ax_info.text(0.05, 0.05, '○', fontsize=16, color='#ffa657', family='monospace'),
     ax_info.text(0.09, 0.05, 'Vertex (bonus)', fontsize=10, color='#8b949e', family='monospace')),
]

# Pre-compute DTW continuously
def dtw_error(partial_path, target):
    """Compute DTW between partial path and closest target segment."""
    if len(partial_path) < 2:
        return 0.0
    n, m = len(partial_path), len(target)
    d = np.full((n+1, m+1), np.inf)
    d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = np.linalg.norm(partial_path[i-1]-target[j-1]) + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    return float(d[n, m] / max(n, m))

# Vertex indices
n_side = 15
vertex_idx = {0, n_side, 2*n_side, len(target)-1}
vertex_positions = [target[i] for i in sorted(vertex_idx)]

# Animation update function
total_frames = len(records)
# Target ~60 seconds at 30fps = 1800 frames. We have ~60 steps normally.
# Interpolate: each record step = multiple animation frames
frame_multiplier = max(1, 1800 // total_frames)
print(f"Total records: {total_frames}, frames: {total_frames * frame_multiplier}")

def init():
    arm_line.set_data([], [])
    ee_dot.set_data([], [])
    cur_target_dot.set_data([], [])
    traj_line.set_data([], [])
    for d in joint_dots:
        d.set_data([], [])
    # Set vertex rings
    for i, ring in enumerate([vertex_ring1, vertex_ring2, vertex_ring3]):
        vtx = target[sorted(vertex_idx)[i]]
        ring.set_data([vtx[0]], [vtx[1]])
    return [arm_line, ee_dot, cur_target_dot, traj_line] + joint_dots + [vertex_ring1, vertex_ring2, vertex_ring3]

def update(frame_idx):
    rec_idx = min(frame_idx // frame_multiplier, total_frames - 1)
    rec = records[rec_idx]
    q = rec['q']
    ee = rec['ee']
    nt = rec['next_target']
    step = rec['step']
    
    # Arm geometry
    pts = get_arm_lines(q)
    arm_line.set_data(pts[:, 0], pts[:, 1])
    ee_dot.set_data([ee[0]], [ee[1]])
    
    # Joint dots
    for i, d in enumerate(joint_dots):
        d.set_data([pts[i, 0]], [pts[i, 1]])
    
    # Current target
    if nt < len(target):
        cur_target_dot.set_data([target[nt, 0]], [target[nt, 1]])
        cur_target_dot.set_visible(True)
    else:
        cur_target_dot.set_visible(False)
    
    # Drawn trajectory (progressively)
    path_up_to = min(rec_idx + 1, len(drawn_path))
    if path_up_to > 1:
        traj_line.set_data(drawn_path[:path_up_to, 0], drawn_path[:path_up_to, 1])
    
    # Visited markers
    visited_up_to = rec['next_target']
    if visited_up_to > 0:
        visited_arr = target[:visited_up_to]
        visited_scatter.set_offsets(visited_arr)
    else:
        visited_scatter.set_offsets(np.empty((0, 2)))
    
    # Vertex highlight (flash when near)
    for i, ring in enumerate([vertex_ring1, vertex_ring2, vertex_ring3]):
        vtx_idx = sorted(vertex_idx)[i]
        if nt == vtx_idx + 1:  # just passed this vertex
            ring.set_markersize(24)
            ring.set_alpha(1.0)
        elif abs(nt - vtx_idx) <= 2:
            ring.set_markersize(20)
            ring.set_alpha(0.6)
        else:
            ring.set_markersize(18)
            ring.set_alpha(0.3)

    # Info panel
    step_text.set_text(f'Step: {step:4d} / {env.max_steps}')
    target_text.set_text(f'Target #: {min(nt, len(target)):3d} / {len(target)}')
    visited_text.set_text(f'Visited: {rec["next_target"]:3d} / {len(target)}')
    ee_text.set_text(f'EE pos: ({ee[0]:+.3f}, {ee[1]:+.3f})')
    
    # DTW (compute progressively)
    partial = drawn_path[:min(rec_idx+1, len(drawn_path))]
    dtw_val = dtw_error(partial, target)
    dtw_text.set_text(f'DTW: {dtw_val:.4f}')
    
    # Reward if available
    if rec_idx < len(env.rewards):
        rew = sum(env.rewards[:rec_idx+1]) if rec_idx < len(env.rewards) else sum(env.rewards)
        reward_text.set_text(f'Reward: {rew:+.1f}')
    
    # Progress bar
    progress = min(nt, len(target)) / len(target)
    progress_bar.set_width(0.9 * progress)
    progress_label.set_text(f'{progress*100:.0f}%')
    
    # Check for completion
    if nt >= len(target):
        progress_label.set_text('COMPLETE!')
        progress_label.set_color('#7ee787')
        progress_label.set_fontweight('bold')
    
    return [arm_line, ee_dot, cur_target_dot, traj_line, visited_scatter,
            vertex_ring1, vertex_ring2, vertex_ring3,
            step_text, target_text, visited_text, ee_text, dtw_text, reward_text,
            progress_bar, progress_label] + joint_dots

# Create animation
print(f"Creating animation ({total_frames * frame_multiplier} frames)...")
ani = animation.FuncAnimation(
    fig, update, frames=total_frames * frame_multiplier,
    init_func=init, blit=False, interval=1000/30  # 30 fps
)

# Save as MP4 using imageio
print(f"Saving MP4 to {OUTPUT_VIDEO}...")
try:
    import imageio_ffmpeg
    writer = animation.FFMpegWriter(fps=30, bitrate=4000, codec='libx264')
    ani.save(OUTPUT_VIDEO, writer=writer, dpi=120)
    print(f"MP4 saved: {OUTPUT_VIDEO}")
except Exception as e:
    print(f"FFmpeg failed: {e}")
    print(f"Trying alternative method...")
    try:
        # Try imageio-based export
        import imageio
        # Render frames manually
        writer = imageio.get_writer(OUTPUT_VIDEO, fps=30, codec='libx264', bitrate='4000k')
        for i in range(total_frames * frame_multiplier):
            update(i)
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            writer.append_data(img[..., :3])  # drop alpha
            if i % 300 == 0:
                print(f"  Frame {i}/{total_frames * frame_multiplier}")
        writer.close()
        print(f"MP4 saved: {OUTPUT_VIDEO}")
    except Exception as e2:
        print(f"Imageio also failed: {e2}")
        print(f"Saving GIF instead to {OUTPUT_GIF}...")
        ani.save(OUTPUT_GIF, writer='pillow', fps=15, dpi=100)
        print(f"GIF saved: {OUTPUT_GIF}")

plt.close()

# Print final stats
from scipy.spatial.distance import directed_hausdorff
if len(drawn_path) > 1:
    n, m = len(drawn_path), len(target)
    d = np.full((n+1, m+1), np.inf); d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = np.linalg.norm(drawn_path[i-1]-target[j-1]) + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    final_dtw = float(d[n, m] / max(n, m))
    final_haus = float(max(directed_hausdorff(drawn_path, target)[0], directed_hausdorff(target, drawn_path)[0]))
    print(f"\nFinal Metrics:")
    print(f"  DTW: {final_dtw:.4f}")
    print(f"  Hausdorff: {final_haus:.4f}")
    print(f"  Points visited: {len(env.visited)}/{len(target)}")
    print(f"  Steps: {env.step_cnt}")
    
    # Vertex distances
    for i in sorted(vertex_idx):
        if i < len(target):
            d_v = float(np.linalg.norm(drawn_path[min(i, len(drawn_path)-1)] - target[i]))
            print(f"  Vertex {i}: {d_v*1000:.1f}mm")
