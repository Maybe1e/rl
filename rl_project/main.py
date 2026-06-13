# rl_project/main.py — Main pipeline: patterns → train → eval → viz → ONNX
"""Run the complete RL robotic arm drawing pipeline."""
import os, sys, time, warnings, csv
warnings.filterwarnings('ignore')
os.environ['RICH_DISABLE'] = '1'

import torch
import torch.nn as nn

from config import RESULT_DIR, DEVICE, TRAIN_PATTERN, TOTAL_TIMESTEPS
from patterns import generate_all_patterns
from train import train
from eval import evaluate, plot_training_curves, plot_trajectories, plot_metrics_bars


def main():
    t0 = time.time()
    print(f'{"="*60}')
    print(f'RL Robotic Arm Drawing Pipeline')
    print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')
    print(f'{"="*60}')
    
    # 1. Generate patterns
    print('\n[1/6] Generating patterns...')
    patterns = generate_all_patterns()
    for k, v in patterns.items():
        print(f'  {k}: {v.shape[0]} pts')
    
    # 2. Train
    print('\n[2/6] Training...')
    model, callback, train_time, model_path = train(TRAIN_PATTERN, TOTAL_TIMESTEPS)
    
    # 3. Evaluate
    print('\n[3/6] Evaluating...')
    model_dir = os.path.join(RESULT_DIR, 'models')
    vec_norm_path = os.path.join(model_dir, f'vec_norm_{TRAIN_PATTERN}.pkl')
    eval_patterns = list(patterns.keys())
    metrics = evaluate(model_path, vec_norm_path, eval_patterns)
    
    # 4. Visualize
    print('\n[4/6] Generating visualizations...')
    plot_training_curves(callback)
    plot_trajectories(metrics, eval_patterns)
    plot_metrics_bars(metrics, eval_patterns)
    
    # 5. ONNX export
    print('\n[5/6] Exporting ONNX...')
    import onnx
    policy = model.policy
    dummy = torch.randn(1, 15, device=policy.device)  # 17-dim observation
    
    class PolicyWrap(nn.Module):
        def __init__(self, p): super().__init__(); self.p = p
        def forward(self, obs):
            f = self.p.mlp_extractor.forward_actor(self.p.features_extractor(obs))
            return self.p.action_net(f)
    
    onnx_path = os.path.join(RESULT_DIR, 'ppo_arm_drawing.onnx')
    try:
        wrapped = PolicyWrap(policy); wrapped.eval()
        torch.onnx.export(wrapped, dummy, onnx_path, export_params=True, opset_version=14,
                          do_constant_folding=True, input_names=['obs'], output_names=['actions'],
                          dynamic_axes={'obs': {0: 'batch'}, 'actions': {0: 'batch'}})
        onnx.checker.check_model(onnx.load(onnx_path))
        print(f'  ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)')
    except Exception as e:
        print(f'  ONNX note: {e}')
        torch.save(policy.state_dict(), os.path.join(RESULT_DIR, 'policy.pt'))
    
    # 6. Summary
    total_time = time.time() - t0
    print(f'\n[6/6] Pipeline complete in {total_time:.1f}s ({total_time/60:.1f} min)')
    print(f'{"="*60}')
    print('Results:')
    for fname in sorted(os.listdir(RESULT_DIR)):
        fp = os.path.join(RESULT_DIR, fname)
        if os.path.isfile(fp):
            print(f'  {fname} ({os.path.getsize(fp)/1024:.1f} KB)')
    print('Done! 🎨🤖')


if __name__ == '__main__':
    main()
