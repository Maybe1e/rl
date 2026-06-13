"""Generate training curve figures from training_log.json."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'font.size': 10, 'font.family': 'serif', 'axes.titlesize': 12})

RESULT_DIR = 'results'
LOG_PATH = os.path.join(RESULT_DIR, 'training_log.json')

with open(LOG_PATH) as f:
    log = json.load(f)

ep_r = np.array(log['ep_rewards'])
ep_l = np.array(log['ep_lengths'])
ep_ts = np.array(log['ep_timesteps'])
eval_r = np.array(log['eval_rewards'])
eval_ts = np.array(log['eval_timesteps'])
loss_data = log['losses']

print(f'Episodes: {len(ep_r)}, Eval points: {len(eval_r)}, Loss points: {len(loss_data)}')
print(f'Reward range: [{ep_r.min():.1f}, {ep_r.max():.1f}]')
print(f'Eval reward range: [{eval_r.min():.1f}, {eval_r.max():.1f}]')

# Smoothing helper
def smooth(y, window=20):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

# === FIGURE 7: Training Curves (Reward + Episode Length) ===
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Episode reward
ax = axes[0]
ax.plot(ep_ts, ep_r, color='#3498db', lw=0.3, alpha=0.5, label='Episode reward')
if len(ep_r) >= 100:
    w = max(50, len(ep_r) // 200)
    roll = smooth(ep_r, w)
    ax.plot(ep_ts[w-1:], roll, color='#e74c3c', lw=2, label=f'Rolling mean (w={w})')
ax.set_ylabel('Episode Reward')
ax.set_title('Figure 7: PPO Training — Reward Progression', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Episode length
ax = axes[1]
ax.plot(ep_ts, ep_l, color='#2ecc71', lw=0.3, alpha=0.5, label='Episode length')
if len(ep_l) >= 100:
    w = max(50, len(ep_l) // 200)
    roll = smooth(ep_l, w)
    ax.plot(ep_ts[w-1:], roll, color='#e74c3c', lw=2, label=f'Rolling mean (w={w})')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Episode Length (steps)')
ax.set_title('Episode Length Progression', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig7_training_curves.png'), dpi=300)
plt.close()
print('Figure 7 saved')

# === FIGURE 8: Eval Reward + Loss Components ===
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Eval reward
ax = axes[0, 0]
ax.plot(eval_ts // 1000, eval_r, 'o-', color='#2196F3', lw=2, ms=4)
ax.set_xlabel('Training Steps (k)')
ax.set_ylabel('Mean Eval Reward')
ax.set_title('Evaluation Reward (every 10k steps)', fontweight='bold')
ax.grid(True, alpha=0.3)

# Policy loss
ax = axes[0, 1]
loss_steps = np.array([d['step'] for d in loss_data]) / 1000.0
policy_loss = np.array([d['policy_loss'] for d in loss_data])
ax.plot(loss_steps, policy_loss, color='#e74c3c', lw=0.8, alpha=0.8)
# Rolling
if len(policy_loss) >= 50:
    roll = smooth(policy_loss, 50)
    ax.plot(loss_steps[49:], roll, color='#c0392b', lw=2, label='Rolling (w=50)')
ax.set_xlabel('Training Steps (k)')
ax.set_ylabel('Policy Gradient Loss')
ax.set_title('Policy Loss', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# Value loss
ax = axes[1, 0]
value_loss = np.array([d['value_loss'] for d in loss_data])
ax.plot(loss_steps, value_loss, color='#2ecc71', lw=0.8, alpha=0.8)
if len(value_loss) >= 50:
    roll = smooth(value_loss, 50)
    ax.plot(loss_steps[49:], roll, color='#27ae60', lw=2, label='Rolling (w=50)')
ax.set_xlabel('Training Steps (k)')
ax.set_ylabel('Value Loss')
ax.set_title('Value Function Loss', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# Entropy + Clip fraction
ax = axes[1, 1]
entropy = np.array([d['entropy'] for d in loss_data])
clip = np.array([d['clip_fraction'] for d in loss_data])
ax2 = ax.twinx()
ax.plot(loss_steps, entropy, color='#9b59b6', lw=0.6, alpha=0.6, label='Entropy')
if len(entropy) >= 50:
    roll = smooth(entropy, 50)
    ax.plot(loss_steps[49:], roll, color='#8e44ad', lw=2, label='Entropy (roll)')
ax2.plot(loss_steps, clip, color='#e67e22', lw=0.6, alpha=0.4, label='Clip fraction')
if len(clip) >= 50:
    roll_c = smooth(clip, 50)
    ax2.plot(loss_steps[49:], roll_c, color='#d35400', lw=2, label='Clip (roll)')
ax.set_xlabel('Training Steps (k)')
ax.set_ylabel('Entropy Loss')
ax2.set_ylabel('Clip Fraction')
ax.set_title('Entropy & Clip Fraction', fontweight='bold')
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

fig.suptitle('Figure 8: PPO Training — Evaluation & Loss Diagnostics', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'fig8_loss_curves.png'), dpi=300)
plt.close()
print('Figure 8 saved')

# Summary stats
print(f'\n=== Training Summary ===')
print(f'Episodes: {len(ep_r)}')
print(f'Final ep_len_mean: {ep_l[-100:].mean():.1f}')
print(f'Final ep_rew_mean: {ep_r[-100:].mean():.1f}')
print(f'Best eval reward: {eval_r.max():.1f} at step {eval_ts[eval_r.argmax()]:,}')
print(f'Policy loss converged: {policy_loss[-50:].mean():.4f}')
print(f'Value loss converged: {value_loss[-50:].mean():.6f}')
print(f'Final entropy: {entropy[-50:].mean():.2f}')
print(f'Final clip fraction: {clip[-50:].mean():.4f}')
