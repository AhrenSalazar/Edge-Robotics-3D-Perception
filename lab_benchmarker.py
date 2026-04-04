"""
SCOPE v2: Latency Benchmark
Compares the CPU inference speed of Occupancy-SCOPE vs. PointPillars.
"""

import argparse
import glob
import os
import time
import csv
import torch
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)

import sys
sys.path.append(os.getcwd())
from scope_kitti_full_bbox import point_cloud_to_occupancy, train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--ckpt', type=str, default='OpenPCDet/weights/pointpillar_7728.pth')
    parser.add_argument('--data_path', type=str, default='dataset/training/velodyne')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_csv', type=str, default='benchmark_results.csv')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SCOPE V2: LATENCY BENCHMARK (CPU ONLY)")
    print("="*60)

    # --- 1. Setup PointPillars ---
    print("\n[Loading PointPillars...]")
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.datasets import DatasetTemplate

    cfg_from_yaml_file(args.cfg_file, cfg)

    class DemoDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, root_path):
            super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=Path(root_path))
            self.sample_file_list = sorted(glob.glob(os.path.join(root_path, '*.bin')))
        def __len__(self): return len(self.sample_file_list)
        def __getitem__(self, index):
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            return self.prepare_data({'points': points})

    dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.data_path)
    pp_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)

    # Try to load checkpoint if it exists, otherwise use random weights for the speed test
    try:
        pp_model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model_state_dict'], strict=False)
    except Exception:
        print("  Note: Checkpoint not found. Using uninitialized weights (fine for latency testing).")

    pp_model = pp_model.cpu().eval()

    # --- 2. Setup SCOPE ---
    print("[Loading Occupancy-SCOPE...]")
    # Train a quick dummy model just to get the weight matrix shape right for inference
    X_dummy = np.random.randint(0, 2, (100, 256))
    Y_dummy = np.random.rand(100, 8)
    scope_theta = train_model(X_dummy, Y_dummy)

    # --- 3. Warmup ---
    # PyTorch needs a few passes to initialize compute graphs and reach normal speeds
    print("[Warming up models...]")
    with torch.no_grad():
        dummy_points = np.random.rand(10000, 4).astype(np.float32)
        dummy_batch = dataset.collate_batch([dataset.prepare_data({'points': dummy_points})])
        for _ in range(3):
            try: pp_model.forward(dummy_batch)
            except: pass
    
    # --- 4. Benchmark Loop ---
    print("\n[Running Benchmark]")
    data_files = sorted(glob.glob(os.path.join(args.data_path, '*.bin')))[:args.num_samples]
    
    results = []
    pp_times, scope_times = [], []

    for i, file_path in enumerate(data_files):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        sample_id = os.path.basename(file_path)[:-4]
        
        # Test PointPillars
        pp_ms = 0
        try:
            with torch.no_grad():
                batch = dataset.collate_batch([dataset.prepare_data({'points': points})])
                t0 = time.time()
                pp_model.forward(batch)
                pp_ms = (time.time() - t0) * 1000
                pp_times.append(pp_ms)
        except Exception: pass

        # Test SCOPE
        scope_ms = 0
        try:
            t0 = time.time()
            feats = point_cloud_to_occupancy(points)
            pred = np.concatenate(([1], feats)).dot(scope_theta)
            scope_ms = (time.time() - t0) * 1000
            scope_times.append(scope_ms)
        except Exception: pass

        print(f"Sample {sample_id} | Points: {len(points)} | PointPillars: {pp_ms:.2f} ms | SCOPE: {scope_ms:.4f} ms")
        results.append({'sample_id': sample_id, 'num_points': len(points), 'pointpillars_ms': pp_ms, 'scope_ms': scope_ms})

    # --- 5. Results ---
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    if pp_times and scope_times:
        pp_avg = np.mean(pp_times)
        scope_avg = np.mean(scope_times)
        print(f"PointPillars Average:  {pp_avg:.2f} ms")
        print(f"Occupancy-SCOPE Avg:   {scope_avg:.4f} ms")
        print(f"\nSpeedup: SCOPE is {pp_avg / scope_avg:.2f}x faster on CPU.")
    else:
        print("Benchmark failed to record times.")

    # Save CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'num_points', 'pointpillars_ms', 'scope_ms'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved detailed log to {args.output_csv}")

if __name__ == '__main__':
    main()