"""Generate supplementary analysis figures from trained model."""
import os, sys, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'font.size': 10, 'font.family': 'serif', 'axes.titlesize': 12})

sys.path.insert(0, 'rl_project')
from env import RoboticArmDrawingEnv
from stable_baselines3 import PPO

RESULT_DIR = 'results'
MODEL_DIR = os.path.join(RESULT_DIR, 'models')

model = PPO.load(os.path.join(MODEL_DIR, 'ppo_final.zip'))
with open(os.path.join(MODEL_DIR, 'vecnorm_final.pkl'), 'rb') as f:
    nd = pickle.load(f)
orm = nd.obs_rms

# Run 5 episodes, recording full state
all_episodes = []
for ep in range(5):
    env = RoboticArmDrawingEnv(target_pattern='triangle', max_steps=600)
    obs, _ = env.reset()
    states = []  # (q, dq, ee, action) per step
    done, trunc = False, False
    while not (done or trunc):
        obs_n = np.clip((obs - orm.mean) / np.sqrt(orm.var + 1e-8), -10, 10)
        act, _ = model.predict(obs_n, deterministic=True)
        q = env.arm.q.copy()
        dq = env.arm.dq.copy()
        _, ee = env.arm.fk()
        states.append(dict(q=q, dq=dq, ee=ee.copy(), action=act.copy(),
                           next_target=env._next_target, reward=0.0))
        obs, r, done, trunc, info = env.step(act)
        states[-1]['reward'] = float(r)
    path = np.array(env.path)
    target = env.target
    # DTW
    n, m = len(path), len(target)
    d = np.full((n+1, m+1), np.inf); d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = np.linalg.norm(path[i-1]-target[j-1]) + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    dtw_val = float(d[n, m] / max(n, m))
    all_episodes.append(dict(states=states, path=path, target=target, dtw=dtw_val,
                             visited=info['visited'], total=info['total']))
    env.close()

best = all_episodes[np.argmin([ep['dtw'] for ep in all_episodes])]
print(f'Best DTW: {best["dtw"]:.4f}, {best["visited"]}/{best["total"]} pts')

# === FIGURE 3: Multi-episode trajectory overlay ===
fig, ax = plt.subplots(figsize=(10, 9))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
target = best['target']
ax.plot(target[:, 0], target[:, 1], 'k--', lw=3, label='Target', zorder=2, alpha=0.7)
for i, ep in enumerate(all_episodes):
    p = ep['path']
    ax.plot(p[:, 0], p[:, 1], '-', color=colors[i], lw=1.2, alpha=0.8,
            label=f'Ep {i+1} (DTW={ep["dtw"]:.4f})')
ax.scatter(target[[0, 15, 30], 0], target[[0, 15, 30], 1],
           c=['#e67e22', '#9b59b6', '#3498db'], s=150, marker='s', zorder=10,
           edgecolors='black', linewidth=1.5)
ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.25, 0.45)
ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
ax.set_title('Figure 3: 5-Episode Trajectory Overlay — Strategy Stability', fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig3_multiep_overlay.png'), dpi=300)
plt.close()
print('Figure 3 saved')

# === FIGURE 4: Joint angle and velocity trajectories ===
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
states = best['states']
t = np.arange(len(states)) * 0.02  # dt=0.02

joint_names = ['Joint 1', 'Joint 2', 'Joint 3']
colors_j = ['#e74c3c', '#2ecc71', '#3498db']

for ji in range(3):
    q = [s['q'][ji] for s in states]
    dq = [s['dq'][ji] for s in states]
    tau = [s['action'][ji] for s in states]

    ax = axes[0, ji]
    ax.plot(t, q, color=colors_j[ji], lw=1.8)
    ax.set_ylabel(f'$q_{ji+1}$ (rad)')
    ax.set_title(f'{joint_names[ji]} Angle', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, ji]
    ax2 = ax.twinx()
    ax.plot(t, dq, color=colors_j[ji], lw=1.5, alpha=0.8, label='$\dot{q}$')
    ax2.plot(t, tau, '--', color='gray', lw=1.2, alpha=0.7, label=r'$\tau$')
    ax.set_ylabel(r'$\dot{q}$ (rad/s)')
    ax2.set_ylabel(r'$\tau$ (Nm)')
    ax.set_title(f'{joint_names[ji]} Velocity & Torque', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

fig.suptitle('Figure 4: Joint Trajectories During Triangle Drawing', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig4_joint_trajectories.png'), dpi=300)
plt.close()
print('Figure 4 saved')

# === FIGURE 5: Per-segment DTW and distance analysis ===
segments = [(0, 15, 'Edge 1\n(Top→Bottom-R)'), (15, 30, 'Edge 2\n(Bottom-R→Bottom-L)'),
            (30, 45, 'Edge 3\n(Bottom-L→Top)')]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for idx, (start, end, label) in enumerate(segments):
    ax = axes[idx]
    seg_target = target[start:end+1]
    # Find closest path points to this segment
    path = best['path']
    # Compute per-point distance to segment
    dists = []
    for pt in path:
        min_d = float('inf')
        for i_seg in range(len(seg_target) - 1):
            a, b = seg_target[i_seg], seg_target[i_seg+1]
            ab = b - a; ap = pt - a
            t_val = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8)
            t_val = np.clip(t_val, 0.0, 1.0)
            closest = a + t_val * ab
            d_val = np.linalg.norm(pt - closest)
            if d_val < min_d:
                min_d = d_val
        dists.append(min_d)

    dists = np.array(dists)
    ax.fill_between(np.arange(len(dists)) * 0.02, dists * 1000, alpha=0.3, color=colors_j[idx])
    ax.plot(np.arange(len(dists)) * 0.02, dists * 1000, color=colors_j[idx], lw=1.5)
    ax.axhline(y=np.mean(dists) * 1000, color='red', ls='--', lw=1, label=f'Mean={np.mean(dists)*1000:.1f}mm')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to segment (mm)')
    ax.set_title(label, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle('Figure 5: Per-Edge Drawing Accuracy Over Time', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig5_per_edge_accuracy.png'), dpi=300)
plt.close()
print('Figure 5 saved')

# === FIGURE 6: EE speed profile and target progress ===
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
path = best['path']
speeds = np.linalg.norm(np.diff(path, axis=0), axis=1) / 0.02
t_speed = np.arange(len(speeds)) * 0.02

axes[0].plot(t_speed, speeds, color='#3498db', lw=1.8)
axes[0].set_ylabel('EE Speed (m/s)')
axes[0].set_title('End-Effector Speed Profile', fontweight='bold')
axes[0].grid(True, alpha=0.3)
# Mark vertex regions
for vi, col, lbl in zip([0, 15, 30], ['#e67e22', '#9b59b6', '#3498db'],
                         ['Top', 'Bottom-R', 'Bottom-L']):
    # Find when target near this vertex
    nt_arr = np.array([s['next_target'] for s in best['states'][1:]])
    region = np.where((nt_arr >= vi-2) & (nt_arr <= vi+2))[0]
    if len(region) > 0:
        for r_idx in region:
            if r_idx < len(speeds):
                axes[0].axvspan(r_idx * 0.02, (r_idx + 1) * 0.02, alpha=0.15, color=col)
        axes[0].text(region[len(region)//2] * 0.02, max(speeds) * 0.95, lbl,
                     color=col, fontsize=8, ha='center', fontweight='bold')

# Progress
nt_arr = np.array([s['next_target'] for s in best['states'][1:]])
progress = nt_arr / max(nt_arr.max(), 1)
axes[1].fill_between(t_speed, progress * 100, alpha=0.4, color='#2ecc71')
axes[1].plot(t_speed, progress * 100, color='#27ae60', lw=2)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Target Progress (%)')
axes[1].set_title('Target Completion Progress', fontweight='bold')
axes[1].set_ylim(0, 105)
axes[1].grid(True, alpha=0.3)
# Mark vertex hits
for vi in [0, 15, 30, 45]:
    hits = np.where(nt_arr == vi + 1)[0]
    if len(hits) > 0:
        axes[1].axvline(x=hits[0] * 0.02, color='#e74c3c', ls=':', lw=1.2, alpha=0.8)
        axes[1].text(hits[0] * 0.02, 95, f'Vtx{vi}', color='#e74c3c', fontsize=7, rotation=90)

fig.suptitle('Figure 6: Drawing Dynamics — Speed and Progress', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig6_speed_progress.png'), dpi=300)
plt.close()
print('Figure 6 saved')

print('\nAll supplementary figures generated.')
