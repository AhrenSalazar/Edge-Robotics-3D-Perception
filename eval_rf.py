"""
Ridge Regression 10m Evaluation

Runs a Random Forest Regressor for 3D bounding box detection. 
It acts as a non-linear baseline to compare against the Ridge Regression model. 
It converts LiDAR points into a 32x32x32 spatial grid, compresses it into 
256 features, and predicts the object's geometry using decision trees.

"""

import argparse
import glob
import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/training')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RANDOM FOREST EVALUATION (10m RANGE)")
    print("="*60)
    
    bin_files = sorted(glob.glob(os.path.join(args.data_path, 'velodyne', '*.bin')))
    
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
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)
    
    print(f"Training on {len(train_idx)} samples, testing on {len(test_idx)} samples...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf_model.fit(X_all[train_idx], Y_all[train_idx])
    
    depth_errors, times = [], []
    for i in range(len(test_idx)):
        feat = X_all[test_idx][i].reshape(1, -1)
        
        start_time = time.perf_counter()
        pred = rf_model.predict(feat)[0]
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)
        depth_errors.append(abs(pred[0] - Y_all[test_idx][i][0]))

    np.save('rf_real_times.npy', np.array(times)[:20])
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Evaluated Samples: {len(test_idx)}")
    print(f"Depth Error:       {np.mean(depth_errors):.2f}m +/- {np.std(depth_errors):.2f}m")
    print(f"Average Latency:   {np.mean(times):.4f} ms")

if __name__ == '__main__':
    main()