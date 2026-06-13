"""Generate comprehensive report figures from trained model."""
import os, sys, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'font.size': 10, 'font.family': 'serif',
                      'axes.titlesize': 12, 'axes.labelsize': 11})

sys.path.insert(0, 'rl_project')
from env import RoboticArmDrawingEnv
from stable_baselines3 import PPO

RESULT_DIR = 'results'
MODEL_DIR = os.path.join(RESULT_DIR, 'models')

# Load model and normalization
print("Loading model...")
model = PPO.load(os.path.join(MODEL_DIR, 'ppo_final.zip'))
with open(os.path.join(MODEL_DIR, 'vecnorm_final.pkl'), 'rb') as f:
    nd = pickle.load(f)
orm = nd.obs_rms

# Run multiple episodes
env = RoboticArmDrawingEnv(target_pattern='triangle', max_steps=600)
results = []
for ep in range(10):
    obs, _ = env.reset()
    done, trunc = False, False
    while not (done or trunc):
        obs_n = np.clip((obs - orm.mean) / np.sqrt(orm.var + 1e-8), -10, 10)
        act, _ = model.predict(obs_n, deterministic=True)
        obs, _, done, trunc, info = env.step(act)
    path = np.array(env.path)
    target = env.target
    # DTW
    n, m = len(path), len(target)
    d = np.full((n+1, m+1), np.inf)
    d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = np.linalg.norm(path[i-1]-target[j-1]) + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    dtw_val = float(d[n, m] / max(n, m))
    results.append(dict(path=path, target=target, dtw=dtw_val,
                        v=info['visited'], t=info['total']))
env.close()

best = results[np.argmin([r['dtw'] for r in results])]

# === FIGURE 1: Full trajectory comparison with vertex callouts ===
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, height_ratios=[2.2, 1])

ax_main = fig.add_subplot(gs[0, :])
target = best['target']
path = best['path']
ax_main.plot(target[:, 0], target[:, 1], 'k--', lw=2.5, label='Target triangle (46 pts)')
ax_main.plot(path[:, 0], path[:, 1], 'b-', lw=2.2, alpha=0.85, label='Arm trajectory')
ax_main.scatter(path[0, 0], path[0, 1], c='#2ecc71', s=120, marker='o', zorder=10,
                edgecolors='black', linewidth=1.2, label='Start')
ax_main.scatter(path[-1, 0], path[-1, 1], c='#e74c3c', s=120, marker='X', zorder=10,
                edgecolors='black', linewidth=1.2, label='End')

vertex_colors = ['#e67e22', '#9b59b6', '#3498db']
vertex_labels = ['Top (Vtx 0)', 'Bottom-Right (Vtx 15)', 'Bottom-Left (Vtx 30)']
for vi, col, lbl in zip([0, 15, 30], vertex_colors, vertex_labels):
    ax_main.scatter(target[vi, 0], target[vi, 1], c=col, s=180, marker='s', zorder=12,
                    edgecolors='black', linewidth=1.5, label=lbl)

ax_main.set_xlim(-0.4, 0.4)
ax_main.set_ylim(-0.25, 0.45)
ax_main.set_aspect('equal')
ax_main.grid(True, alpha=0.25)
title_main = f'PPO Trained 3-Link Arm: Triangle Drawing | DTW={best["dtw"]:.4f} | {best["v"]}/{best["t"]} pts | {len(path)} steps'
ax_main.set_title(title_main, fontweight='bold')
ax_main.legend(loc='lower right', fontsize=8, ncol=2)

zoom_titles = ['Top Vertex', 'Bottom-Right Vertex', 'Bottom-Left Vertex']
for idx, (vi, col, ttl) in enumerate(zip([0, 15, 30], vertex_colors, zoom_titles)):
    ax = fig.add_subplot(gs[1, idx])
    z = 0.08
    ax.plot(target[:, 0], target[:, 1], 'gray', lw=1.5, alpha=0.4)
    ax.plot(path[:, 0], path[:, 1], 'b-', lw=2, alpha=0.9)
    ax.scatter(target[vi, 0], target[vi, 1], c=col, s=200, marker='s', zorder=12,
               edgecolors='black', linewidth=2)
    dv = np.linalg.norm(path - target[vi], axis=1)
    ci = np.argmin(dv)
    ax.scatter(path[ci, 0], path[ci, 1], c='red', s=100, marker='*', zorder=15, edgecolors='black')
    ax.set_xlim(target[vi, 0] - z, target[vi, 0] + z)
    ax.set_ylim(target[vi, 1] - z, target[vi, 1] + z)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{ttl}: {dv[ci]*1000:.1f}mm', fontweight='bold')

fig.suptitle('Figure 1: Trajectory Evaluation with Vertex Precision Analysis', fontweight='bold', y=1.01)
plt.savefig(os.path.join(RESULT_DIR, 'fig1_trajectory.png'), dpi=300)
plt.close()
print('Figure 1 saved')

# === FIGURE 2: Metrics summary ===
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
episodes = list(range(1, 11))

dtws = [r['dtw'] for r in results]
colors_dtw = ['#2ecc71' if d == min(dtws) else '#3498db' for d in dtws]
axes[0].bar(episodes, [d * 1000 for d in dtws], color=colors_dtw, edgecolor='black', linewidth=0.5)
axes[0].axhline(y=np.mean(dtws) * 1000, color='red', ls='--', lw=1.5, label=f'Mean={np.mean(dtws)*1000:.1f}mm')
axes[0].set_xlabel('Evaluation Episode')
axes[0].set_ylabel('DTW Distance (mm)')
axes[0].set_title('DTW Distance (10 Evaluations)', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')

vtx_data = {vi: [] for vi in [0, 15, 30, 45]}
for r in results:
    for vi in vtx_data:
        dv = np.min(np.linalg.norm(r['path'] - r['target'][vi], axis=1))
        vtx_data[vi].append(dv * 1000)

vtx_means = [np.mean(vtx_data[vi]) for vi in [0, 15, 30, 45]]
vtx_stds = [np.std(vtx_data[vi]) for vi in [0, 15, 30, 45]]
vtx_colors = ['#e67e22', '#9b59b6', '#3498db', '#2ecc71']
vtx_names = ['Top (Vtx 0)', 'Bottom-R (Vtx 15)', 'Bottom-L (Vtx 30)', 'Close (Vtx 45)']
bars = axes[1].bar(vtx_names, vtx_means, yerr=vtx_stds, capsize=8, color=vtx_colors,
                    edgecolor='black', linewidth=0.8)
axes[1].set_ylabel('Distance (mm)')
axes[1].set_title('Vertex Precision (mean +/- std)', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for b, v in zip(bars, vtx_means):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f'{v:.1f}', ha='center', fontsize=9)

steps = [r['path'].shape[0] for r in results]
axes[2].bar(episodes, steps, color='#9b59b6', edgecolor='black', linewidth=0.5)
axes[2].axhline(y=np.mean(steps), color='red', ls='--', lw=1.5, label=f'Mean={np.mean(steps):.0f}')
axes[2].set_xlabel('Evaluation Episode')
axes[2].set_ylabel('Steps')
axes[2].set_title('Episode Length (46 targets)', fontweight='bold')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 2: Quantitative Evaluation Metrics', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig2_metrics.png'), dpi=300)
plt.close()
print('Figure 2 saved')

# Summary
print(f'\n=== Final Metrics ===')
print(f'DTW: mean={np.mean(dtws):.4f}, min={np.min(dtws):.4f}, std={np.std(dtws):.4f}')
print(f'Success: {sum(1 for r in results if r["v"]==r["t"])}/10 (100%)')
print(f'Steps: mean={np.mean(steps):.1f}, std={np.std(steps):.1f}')
for vi in [0, 15, 30, 45]:
    m = np.mean(vtx_data[vi]); s = np.std(vtx_data[vi])
    print(f'Vertex {vi}: {m:.1f}mm +/- {s:.1f}mm')
