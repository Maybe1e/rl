# rl_project/config.py — Shared configuration
import os, torch
import numpy as np

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'results')
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---- Reproducibility ----
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- Device ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Environment ----
CANVAS_BOUNDS = (-0.6, 0.6)
TARGET_REACH_THRESHOLD = 0.10  # balanced: precision vs success rate
MAX_STEPS = 600  # more steps for 46-pt dense triangle
LINK_LENGTHS = (0.3, 0.25, 0.15)  # shorter links = EE stays in canvas
DT = 0.02
INIT_ANGLES = (2.699, -1.637, -1.106)  # EE at (0.001,0.340) — triangle top vertex
SUCCESS_RATIO = 0.6  # 19/31 points for closed triangle (31 pts)

# ---- Training ----
TRAIN_PATTERN = 'triangle'
TOTAL_TIMESTEPS = 600_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5

PPO_CONFIG = dict(
    policy='MlpPolicy',
    learning_rate=lambda f: 3e-4 * f,
    n_steps=2048, batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    verbose=1, device='auto', seed=SEED,
)

# ---- Pattern config (circle + spiral for generalization test) ----
PATTERN_CONFIGS = {
    'circle': dict(type='circle', n=20, r=0.3, c=(0,0)),         # 20 pts, spacing 0.094 ~ threshold 0.10
    'spiral': dict(type='spiral_out_in', n=80, loops=2, r_max=0.3),  # outside→inside spiral
    'triangle': dict(type='triangle', s=0.5, n=15),  # 46 pts, 0.033m spacing
}
