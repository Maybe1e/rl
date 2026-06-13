"""Train with callback data logging for training curves."""
import os, sys, time, pickle, json, warnings
warnings.filterwarnings('ignore')
os.environ['RICH_DISABLE'] = '1'

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl_project'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from config import SEED, DEVICE, RESULT_DIR, PPO_CONFIG
from env import RoboticArmDrawingEnv

TOTAL_STEPS = 600_000
EVAL_FREQ = 10_000


class LoggingCallback(BaseCallback):
    """Records per-episode and per-eval metrics to disk."""

    def __init__(self, eval_env, eval_freq):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_timesteps = []
        self.eval_rewards = []
        self.eval_timesteps = []
        self.losses = []  # approximate loss from model.logger
        self.entropy = []
        self.clip_fractions = []
        self.save_path = os.path.join(RESULT_DIR, 'training_log.json')

    def _on_step(self):
        # Per-step: log approximate loss if available
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            d = self.model.logger.name_to_value
            if len(self.losses) == 0 or self.n_calls - self._last_loss_step > 500:
                self.losses.append(dict(
                    step=int(self.n_calls),
                    policy_loss=float(d.get('train/policy_gradient_loss', 0)),
                    value_loss=float(d.get('train/value_loss', 0)),
                    entropy=float(d.get('train/entropy_loss', 0)),
                    clip_fraction=float(d.get('train/clip_fraction', 0)),
                    approx_kl=float(d.get('train/approx_kl', 0)),
                ))
                self._last_loss_step = self.n_calls

        # Per eval_freq: evaluation
        if self.n_calls % self.eval_freq == 0:
            if hasattr(self.eval_env, 'obs_rms'):
                self.eval_env.obs_rms = self.training_env.obs_rms
            mr, _ = evaluate_policy(self.model, self.eval_env,
                                     n_eval_episodes=5, deterministic=True)
            self.eval_rewards.append(float(mr))
            self.eval_timesteps.append(int(self.n_calls))

        # Per episode
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]['episode']
            self.ep_rewards.append(float(info['r']))
            self.ep_lengths.append(int(info['l']))
            self.ep_timesteps.append(int(self.n_calls))

        # Save periodically
        if self.n_calls % 50000 == 0 and self.n_calls > 0:
            self._save()
        return True

    def _save(self):
        data = dict(
            ep_rewards=self.ep_rewards, ep_lengths=self.ep_lengths,
            ep_timesteps=self.ep_timesteps, eval_rewards=self.eval_rewards,
            eval_timesteps=self.eval_timesteps, losses=self.losses,
        )
        with open(self.save_path, 'w') as f:
            json.dump(data, f)

    def on_training_end(self):
        self._save()
        print(f'Training log saved: {self.save_path}')
        print(f'  Episodes: {len(self.ep_rewards)}, Eval points: {len(self.eval_rewards)}, Loss points: {len(self.losses)}')


def main():
    print(f'Training PPO: triangle, {TOTAL_STEPS:,} steps, device={DEVICE}')
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

    def make_env():
        return Monitor(RoboticArmDrawingEnv(target_pattern='triangle', max_steps=600))

    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, training=False)

    callback = LoggingCallback(eval_env, EVAL_FREQ)
    callback._last_loss_step = 0

    model = PPO(env=train_env, tensorboard_log=os.path.join(RESULT_DIR, 'tensorboard'), **PPO_CONFIG)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback, progress_bar=True)
    elapsed = time.time() - t0
    print(f'\nTraining done: {elapsed:.0f}s ({elapsed/60:.1f} min)')

    # Save model
    model_path = os.path.join(RESULT_DIR, 'models', 'ppo_train_curves')
    model.save(model_path)
    train_env.save(os.path.join(RESULT_DIR, 'models', 'vecnorm_train_curves.pkl'))
    print(f'Model saved: {model_path}.zip')


if __name__ == '__main__':
    main()
