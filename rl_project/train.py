# rl_project/train.py — PPO training module
import os, time, json
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from config import SEED, DEVICE, RESULT_DIR, TRAIN_PATTERN, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES, PPO_CONFIG
from env import RoboticArmDrawingEnv


class BestModelCallback(BaseCallback):
    """Callback that saves best model based on eval reward."""
    
    def __init__(self, eval_env, eval_freq, n_eval=N_EVAL_EPISODES):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.ep_r, self.ep_l = [], []
        self.eval_r, self.eval_ts = [], []
        self.best_eval = -float('inf')
        self.best_step = 0
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            if hasattr(self.eval_env, 'obs_rms'):
                self.eval_env.obs_rms = self.training_env.obs_rms
            mr, _ = evaluate_policy(self.model, self.eval_env,
                                     n_eval_episodes=self.n_eval, deterministic=True)
            self.eval_r.append(mr)
            self.eval_ts.append(self.n_calls)
            if mr > self.best_eval:
                self.best_eval = mr
                self.best_step = self.n_calls
                self.model.save(os.path.join(RESULT_DIR, 'models', 'ppo_best'))
        if 'episode' in self.locals.get('infos', [{}])[0]:
            self.ep_r.append(self.locals['infos'][0]['episode']['r'])
            self.ep_l.append(self.locals['infos'][0]['episode']['l'])
        return True


def train(target_pattern=TRAIN_PATTERN, total_timesteps=TOTAL_TIMESTEPS):
    """Train PPO policy and return (model, callback, train_time, model_path)."""
    
    print(f'\n{"="*60}')
    print(f'Training PPO: pattern={target_pattern}, steps={total_timesteps:,}, device={DEVICE}')
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'{"="*60}\n')
    
    def make_env():
        return Monitor(RoboticArmDrawingEnv(
            target_pattern=target_pattern, render_mode=None, max_steps=400))
    
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.99, training=False)
    
    callback = BestModelCallback(eval_env, EVAL_FREQ)
    
    model = PPO(env=train_env, tensorboard_log=os.path.join(RESULT_DIR, 'tensorboard'), **PPO_CONFIG)
    
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    train_time = time.time() - t0
    
    model_dir = os.path.join(RESULT_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'ppo_arm_{target_pattern}')
    model.save(model_path)
    train_env.save(os.path.join(model_dir, f'vec_norm_{target_pattern}.pkl'))
    
    print(f'\nTraining done: {train_time:.1f}s ({train_time/60:.1f} min)')
    print(f'Best eval reward: {callback.best_eval:.2f} at step {callback.best_step:,}')
    print(f'Model saved: {model_path}.zip\n')
    
    # Save training log
    log = {
        'train_time': train_time, 'total_timesteps': total_timesteps,
        'eval_rewards': callback.eval_r, 'eval_timestamps': callback.eval_ts,
        'best_eval': callback.best_eval, 'best_step': callback.best_step,
    }
    with open(os.path.join(RESULT_DIR, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    
    return model, callback, train_time, model_path
