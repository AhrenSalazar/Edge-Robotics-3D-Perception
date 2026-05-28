"""
Instead of relying on a single train/test split, it runs the evaluation 5 times 
using different random seeds. For a statistically significant average and 
standard deviation, checking the model's accuracy is consistent across all data.
"""

import argparse
import glob
import os
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from scope_kitti_full_bbox import compute_box_3d

def extract_occupancy_grid(points):
    mask = ((points[:, 0] > 0) & (points[:, 0] < 10) &
            (points[:, 1] > -5) & (points[:, 1] < 5) &
            (points[:, 2] > -2) & (points[:, 2] < 2))
    pts = points[mask]
    
    grid = np.zeros((32, 32, 32), dtype=np.uint8)
    if len(pts) > 0:
        x_idx = (np.clip(pts[:, 0] / 10.0, 0, 0.99) * 31).astype(int)
        y_idx = (np.clip((pts[:, 1] + 5) / 10.0, 0, 0.99) * 31).astype(int)
        z_idx = (np.clip((pts[:, 2] + 2) / 4.0, 0, 0.99) * 31).astype(int)
        grid[x_idx, y_idx, z_idx] = 1
    
    features = []
    for x in range(0, 32, 4):
        for y in range(0, 32, 4):
            for z in range(0, 32, 8):
                features.append(float(np.mean(grid[x:x+4, y:y+4, z:z+8])))
    return np.array(features)

def train_ridge(X, Y, alpha=0.5):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    I = np.eye(X_aug.shape[1])
    return np.linalg.inv(X_aug.T @ X_aug + alpha * I) @ X_aug.T @ Y

def predict_ridge(theta, features):
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features_aug = np.hstack([np.ones((features.shape[0], 1)), features])
    return features_aug @ theta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/training')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("EDGE-3D: VALIDATION (5-RUN AVERAGE)")
    print("="*60)
    
    bin_files = sorted(glob.glob(os.path.join(args.data_path, 'velodyne', '*.bin')))
    print("Getting features")
    
    X_all, Y_all = [], []
    for bin_file in bin_files:
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        mask = ((points[:, 0] > 0) & (points[:, 0] < 10) &
                (points[:, 1] > -5) & (points[:, 1] < 5) &
                (points[:, 2] > -2) & (points[:, 2] < 2))
        pts = points[mask]
        
        if len(pts) > 20: 
            geom_label = np.array([
                np.mean(pts[:, 0]), np.mean(pts[:, 1]), np.mean(pts[:, 2]),
                np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2]),
                0.0, 1.0
            ])
            X_all.append(extract_occupancy_grid(points))
            Y_all.append(geom_label)
            
    X_all, Y_all = np.array(X_all), np.array(Y_all)
    indices = np.arange(len(X_all))
    
    seeds = [42, 123, 456, 789, 1024]
    all_depth_errors = []
    all_latencies = []
    
    print(f"\nRunning tests across {len(seeds)} different data splits...\n")
    
    for run, seed in enumerate(seeds):
        train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=seed)
        
        theta = train_ridge(X_all[train_idx], Y_all[train_idx])
        
        run_depth_errors = []
        run_times = []
        
        for i in range(len(test_idx)):
            feat = X_all[test_idx][i]
            
            start_time = time.perf_counter()
            pred = predict_ridge(theta, feat)[0]
            end_time = time.perf_counter()
            
            run_times.append((end_time - start_time) * 1000)
            run_depth_errors.append(abs(pred[0] - Y_all[test_idx][i][0]))
        
        fold_mean_error = np.mean(run_depth_errors)
        fold_mean_time = np.mean(run_times)
        
        all_depth_errors.append(fold_mean_error)
        all_latencies.append(fold_mean_time)
        
        print(f"Run {run+1}/5 (Seed {seed}): Depth Error = {fold_mean_error:.4f}m, Latency = {fold_mean_time:.4f}ms")
    
    print("\n" + "="*60)
    print("FINAL EVAL SUMMARY")
    print("="*60)
    print(f"Mean Depth Error: {np.mean(all_depth_errors):.4f}m ± {np.std(all_depth_errors):.4f}m")
    print(f"Mean Latency:     {np.mean(all_latencies):.4f}ms ± {np.std(all_latencies):.4f}ms")

if __name__ == '__main__':
    main()