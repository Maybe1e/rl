# rl_project/eval.py — Evaluation, metrics, and visualization
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config import RESULT_DIR, DEVICE, SUCCESS_RATIO
from env import RoboticArmDrawingEnv
from patterns import get_pattern

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.size': 11})


# ---- Metrics ----
def dtw_dist(a, b):
    n, m = len(a), len(b)
    if n < 2 or m < 2: return float('inf')
    d = np.full((n+1, m+1), np.inf); d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = np.linalg.norm(a[i-1]-b[j-1]) + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    return float(d[n, m] / max(n, m))

def hausdorff(a, b):
    if len(a) < 2 or len(b) < 2: return float('inf')
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))

def dimless_jerk(traj):
    if len(traj) < 4: return 0.0
    v = np.diff(traj, axis=0); L = np.sum(np.linalg.norm(v, axis=1))
    if L < 1e-6: return 0.0
    j = np.diff(traj, n=3, axis=0)
    return float((len(traj)**5 / L**2) * np.mean(np.sum(j**2, axis=1)))


# ---- Evaluation ----
def evaluate(model_path, vec_norm_path, patterns=('circle','square','spiral','heart','triangle','star'), n_ep=5):
    """Run evaluation — raw env, no VecNormalize wrapper to avoid loading bugs."""
    print(f'\n{"="*60}')
    print(f'Evaluating on {len(patterns)} patterns ({n_ep} episodes each)...')
    print(f'{"="*60}')
    
    loaded_model = PPO.load(model_path)
    # Load normalization stats manually (bypass VecNormalize wrapper bugs)
    import pickle
    with open(vec_norm_path, 'rb') as f:
        norm_data = pickle.load(f)
    obs_rms = norm_data.obs_rms  # RunningMeanStd
    
    all_metrics = {}
    for pat in patterns:
        env = RoboticArmDrawingEnv(target_pattern=pat, render_mode=None, max_steps=400)
        dtw_l, haus_l, sm_l, succ_l, len_l, rew_l, trajs = [], [], [], [], [], [], []
        for ep_idx in range(n_ep):
            obs, _ = env.reset(); done = False; er = 0.0
            while not done:
                # Manual normalization using training stats
                obs_norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                a, _ = loaded_model.predict(obs_norm, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(a)
                er += float(r); done = terminated or truncated
            drawn = np.array(env.path) if env.path else np.zeros((0, 2))
            tgt = env.target
            dtw_l.append(dtw_dist(drawn, tgt) if len(drawn) > 1 else float('inf'))
            haus_l.append(hausdorff(drawn, tgt) if len(drawn) > 1 else float('inf'))
            sm_l.append(dimless_jerk(drawn) if len(drawn) > 3 else float('inf'))
            succ_l.append(len(env.visited) >= int(len(tgt) * SUCCESS_RATIO))
            len_l.append(env.step_cnt); rew_l.append(er); trajs.append(drawn)
        env.close()
        vd = [x for x in dtw_l if x != float('inf')]
        vh = [x for x in haus_l if x != float('inf')]
        vs = [x for x in sm_l if x != float('inf')]
        all_metrics[pat] = {
            'dtw_mean': float(np.mean(vd)) if vd else float('inf'),
            'dtw_std': float(np.std(vd)) if vd else 0,
            'hausdorff_mean': float(np.mean(vh)) if vh else float('inf'),
            'hausdorff_std': float(np.std(vh)) if vh else 0,
            'smoothness_mean': float(np.mean(vs)) if vs else float('inf'),
            'smoothness_std': float(np.std(vs)) if vs else 0,
            'success_rate': float(np.mean(succ_l)),
            'ep_len_mean': float(np.mean(len_l)), 'ep_len_std': float(np.std(len_l)),
            'rew_mean': float(np.mean(rew_l)), 'rew_std': float(np.std(rew_l)),
            'trajs': trajs, 'target': tgt,
        }
        print(f'  {pat:8s} | succ={all_metrics[pat]["success_rate"]:.0%} | '
              f'dtw={all_metrics[pat]["dtw_mean"]:.4f} | '
              f'rew={all_metrics[pat]["rew_mean"]:.1f} | '
              f'len={all_metrics[pat]["ep_len_mean"]:.0f}')
    
    # Save metrics
    m_ser = {k: {mk: mv for mk, mv in v.items() if mk not in ('trajs', 'target')}
             for k, v in all_metrics.items()}
    with open(os.path.join(RESULT_DIR, 'eval_metrics.json'), 'w') as f:
        json.dump(m_ser, f, indent=2)
    
    # CSV
    with open(os.path.join(RESULT_DIR, 'eval_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Pattern', 'Success', 'DTW', 'DTW_Std', 'Hausdorff', 'Smoothness',
                     'EpLen', 'EpLen_Std', 'Reward', 'Reward_Std'])
        for p in patterns:
            m = all_metrics[p]
            w.writerow([p, m['success_rate'], m['dtw_mean'], m['dtw_std'],
                        m['hausdorff_mean'], m['smoothness_mean'],
                        m['ep_len_mean'], m['ep_len_std'], m['rew_mean'], m['rew_std']])
    
    for k, v in all_metrics.items():
        sd = {f'traj_{i}': v['trajs'][i] for i in range(len(v['trajs']))}
        sd['target'] = v['target']
        np.savez(os.path.join(RESULT_DIR, f'eval_traj_{k}.npz'), **sd)
    
    return all_metrics


# ---- Visualization ----
def plot_training_curves(callback, save_path=None):
    """Plot training reward curves."""
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, 'training_curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    if callback.eval_ts:
        ax.plot(callback.eval_ts, callback.eval_r, 'o-', color='#2196F3', lw=1.5, ms=4, label='Eval Reward')
        w = max(1, len(callback.eval_r)//5)
        if len(callback.eval_r) >= w:
            roll = np.convolve(callback.eval_r, np.ones(w)/w, mode='valid')
            ax.plot(callback.eval_ts[w-1:], roll, '-', color='#F44336', lw=2, label=f'Rolling(w={w})')
    ax.set(xlabel='Timesteps', ylabel='Mean Reward', title='Evaluation Reward')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    if callback.ep_r:
        ax.plot(callback.ep_r, color='#4CAF50', lw=0.5, alpha=0.6)
        if len(callback.ep_r) >= 20:
            roll = np.convolve(callback.ep_r, np.ones(20)/20, mode='valid')
            ax.plot(range(19, len(callback.ep_r)), roll, '-', color='#E91E63', lw=2, label='Rolling(w=20)')
    ax.set(xlabel='Episode', ylabel='Reward', title='Training Episodes')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f'  Training curves saved: {save_path}')


def plot_trajectories(metrics, patterns, save_path=None):
    """Plot target vs drawn trajectories."""
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, 'trajectory_comparison.png')
    n = len(patterns)
    cols = max(2, min(3, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.atleast_1d(axes).flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    
    for i, (pat, col) in enumerate(zip(patterns, colors)):
        ax = axes[i]; m = metrics[pat]
        ax.plot(m['target'][:, 0], m['target'][:, 1], 'gray', lw=1, ls='--', alpha=0.5, label='Target')
        for j, traj in enumerate(m['trajs']):
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.5,
                        alpha=0.8 if j == 0 else 0.3, label='Drawn' if j == 0 else None)
        ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), aspect='equal',
               title=f'{pat} (S:{m["success_rate"]:.0%})')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7) if i == 0 else None
    for j in range(len(patterns), len(axes)):
        axes[j].axis('off')
    fig.suptitle('Trajectory Comparison', fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f'  Trajectory plot saved: {save_path}')


def plot_metrics_bars(metrics, patterns, save_path=None):
    """Plot Success/DTW/Smoothness/Length bar charts."""
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, 'metrics_bar_charts.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, len(patterns)))
    
    for ax, (title, ylabel, key) in zip(
        axes.flatten(),
        [('Success Rate', 'Success Rate (%)', 'success_rate'),
         ('Drawing Similarity (DTW)', 'DTW Distance', 'dtw_mean'),
         ('Trajectory Smoothness', 'Jerk (lower=smoother)', 'smoothness_mean'),
         ('Episode Length', 'Steps', 'ep_len_mean')]
    ):
        vals = [metrics[p][key] for p in patterns]
        if key == 'success_rate':
            vals = [v * 100 for v in vals]
        bars = ax.bar(patterns, vals, color=colors, ec='black', lw=0.5)
        ax.set(ylabel=ylabel, title=title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.02,
                    f'{v:.1f}' if key != 'success_rate' else f'{v:.0f}%',
                    ha='center', fontsize=8)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f'  Metrics bar chart saved: {save_path}')
