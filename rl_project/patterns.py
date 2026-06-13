# rl_project/patterns.py — Target drawing pattern generation
import numpy as np
from config import PATTERN_CONFIGS

def gen_circle(r=0.3, c=(0,0), n=30):
    # Start at top (π/2) so first point aligns with arm at (0, ~0.34)
    t = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n)
    return np.column_stack([c[0]+r*np.cos(t), c[1]+r*np.sin(t)])

def gen_square(s=0.5, c=(0,0), n=12):
    h = s/2; cx, cy = c
    top = np.column_stack([np.linspace(cx-h, cx+h, n), np.full(n, cy+h)])
    right = np.column_stack([np.full(n, cx+h), np.linspace(cy+h, cy-h, n)])
    bottom = np.column_stack([np.linspace(cx+h, cx-h, n), np.full(n, cy-h)])
    left = np.column_stack([np.full(n, cx-h), np.linspace(cy-h, cy+h, n)])
    return np.vstack([top, right, bottom, left])

def gen_spiral_out_in(c=(0,0), loops=2, n=80, r_max=0.3):
    """Spiral from outside to inside, starts at top for arm alignment."""
    t = np.linspace(np.pi/2, np.pi/2 + loops*2*np.pi, n)
    r = np.linspace(r_max, 0.03, n)
    return np.column_stack([c[0]+r*np.cos(t), c[1]+r*np.sin(t)])

def gen_heart(c=(0,0.1), size=0.3, n=50):
    t = np.linspace(0, 2*np.pi, n)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    sc = size/max(np.max(np.abs(x)), np.max(np.abs(y)))
    return np.column_stack([c[0]+x*sc, c[1]+y*sc*0.8])

def gen_triangle(s=0.5, c=(0,0.05), n=10):
    cx, cy = c; h = s*np.sqrt(3)/2
    top = np.array([cx, cy+h*2/3])
    bl = np.array([cx-s/2, cy-h*1/3])
    br = np.array([cx+s/2, cy-h*1/3])
    s1 = np.column_stack([np.linspace(top[0], br[0], n), np.linspace(top[1], br[1], n)])
    s2 = np.column_stack([np.linspace(br[0], bl[0], n), np.linspace(br[1], bl[1], n)])
    s3 = np.column_stack([np.linspace(bl[0], top[0], n), np.linspace(bl[1], top[1], n)])
    path = np.vstack([s1, s2, s3])
    # Close the loop: explicitly return to exact starting vertex
    return np.vstack([path, top.reshape(1, 2)])

def gen_star(size=0.35, c=(0,0), n_pts=5, n_per=10):
    angles = np.linspace(np.pi/2, np.pi/2+2*np.pi, 2*n_pts+1)[:-1]
    radii = np.tile([size, size*0.4], n_pts)
    pts = np.column_stack([c[0]+radii*np.cos(angles), c[1]+radii*np.sin(angles)])
    t = np.linspace(0, 1, len(pts)*n_per)
    x = np.interp(t, np.linspace(0,1,len(pts)), pts[:,0])
    y = np.interp(t, np.linspace(0,1,len(pts)), pts[:,1])
    return np.column_stack([x, y])

# Generator registry
_GENERATORS = {
    'circle': gen_circle, 'square': gen_square, 'spiral_out_in': gen_spiral_out_in,
    'heart': gen_heart, 'triangle': gen_triangle, 'star': gen_star,
}

def generate_all_patterns():
    """Generate all patterns and return as dict."""
    patterns = {}
    for name, cfg in PATTERN_CONFIGS.items():
        gen = _GENERATORS[cfg['type']]
        params = {k: v for k, v in cfg.items() if k != 'type'}
        p = gen(**params)
        patterns[name] = np.clip(p, -0.5, 0.5)
    return patterns

def get_pattern(name):
    """Get a single pattern by name."""
    if name not in PATTERN_CONFIGS:
        raise ValueError(f'Unknown pattern: {name}. Available: {list(PATTERN_CONFIGS.keys())}')
    cfg = PATTERN_CONFIGS[name]
    gen = _GENERATORS[cfg['type']]
    params = {k: v for k, v in cfg.items() if k != 'type'}
    return np.clip(gen(**params), -0.5, 0.5)
