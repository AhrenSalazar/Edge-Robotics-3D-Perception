"""
SCOPE v2: Random Forest Evaluation (10m Near-Field)
Evaluates a Non-Linear Random Forest model on a 256-feature occupancy grid.
Optimized for edge robotics by constraining the detection range to <10 meters.
"""

import argparse
import glob
import os
import sys
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.getcwd())
from scope_kitti_full_bbox import compute_box_3d

class KITTICalibration:
    def __init__(self, calib_path):
        self.R0_rect = np.eye(3)
        self.Tr_velo_to_cam = np.eye(4)
        if os.path.exists(calib_path):
            self._load_from_file(calib_path)
        else:
            self._load_default()
            
    def _load_from_file(self, calib_path):
        try:
            with open(calib_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('R0_rect:'):
                        self.R0_rect = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
                    elif line.startswith('Tr_velo_to_cam:'):
                        tr_matrix = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                        self.Tr_velo_to_cam = np.vstack((tr_matrix, [0, 0, 0, 1]))
        except Exception as e:
            print(f"Warning: Could not parse calib file: {e}")
            self._load_default()
            
    def _load_default(self):
        self.R0_rect = np.eye(3)
        self.Tr_velo_to_cam = np.array([
            [7.49916597e-03, -9.99971648e-01, -8.65110622e-04, -1.27654328e-02],
            [1.18652889e-02,  9.54520517e-04, -9.99910318e-01, -5.40398304e-02],
            [9.99882144e-01,  7.60571522e-03,  1.18119621e-02, -2.79683117e-01],
            [0, 0, 0, 1]
        ])
        
    def cam_to_velo(self, points_cam):
        pts_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        pts_velo = pts_h @ np.linalg.inv(self.Tr_velo_to_cam).T
        return pts_velo[:, :3]
        
    def box_cam_to_velo(self, center_cam, size, rotation_cam):
        center_velo = self.cam_to_velo(center_cam.reshape(1, 3))[0]
        return center_velo, size, rotation_cam

def load_kitti_label(label_path, calib):
    objects = []
    if not os.path.exists(label_path): return objects
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] not in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']:
                continue
            
            h_cam, w_cam, l_cam = float(parts[8]), float(parts[9]), float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            ry_cam = float(parts[14])
            
            center_cam = np.array([x_cam, y_cam, z_cam])
            center_velo, size, ry_velo = calib.box_cam_to_velo(center_cam, np.array([h_cam, w_cam, l_cam]), ry_cam)
            
            # CONSTRAINED DOMAIN: Only keep objects within 10m forward and 5m left/right
            if 0 < center_velo[0] < 10 and -5 < center_velo[1] < 5:
                objects.append({
                    'type': parts[0], 'center': center_velo, 'size': size, 'rotation': ry_velo
                })
    return objects

def extract_occupancy_grid(points):
    # CONSTRAINED DOMAIN: 10m range
    X_RANGE, Y_RANGE, Z_RANGE, GRID_SIZE = (0, 10), (-5, 5), (-2, 2), 32
    
    mask = ((points[:, 0] > X_RANGE[0]) & (points[:, 0] < X_RANGE[1]) &
            (points[:, 1] > Y_RANGE[0]) & (points[:, 1] < Y_RANGE[1]) &
            (points[:, 2] > Z_RANGE[0]) & (points[:, 2] < Z_RANGE[1]))
    pts_valid = points[mask]
    
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if len(pts_valid) > 0:
        # Normalization updated for the 10m range
        x_idx = (np.clip((pts_valid[:, 0] - X_RANGE[0]) / 10.0, 0, 0.99) * 31).astype(np.int32)
        y_idx = (np.clip((pts_valid[:, 1] - Y_RANGE[0]) / 10.0, 0, 0.99) * 31).astype(np.int32)
        z_idx = (np.clip((pts_valid[:, 2] - Z_RANGE[0]) / 4.0, 0, 0.99) * 31).astype(np.int32)
        grid[x_idx, y_idx, z_idx] = 1
    
    features = []
    for x in range(0, 32, 4):
        for y in range(0, 32, 4):
            for z in range(0, 32, 8):
                features.append(float(np.mean(grid[x:x+4, y:y+4, z:z+8])))
    return np.array(features)

def train_rf(X, Y):
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf_model.fit(X, Y)
    return rf_model

def predict_rf(model, features):
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return model.predict(features)[0]

def compute_iou_bev(pred_box, gt_box):
    px, py, pw, pl = pred_box[0], pred_box[1], pred_box[4], pred_box[5]
    gx, gy, gw, gl = gt_box[0], gt_box[1], gt_box[4], gt_box[5]
    
    xi1, xi2 = max(px - pl/2, gx - gl/2), min(px + pl/2, gx + gl/2)
    yi1, yi2 = max(py - pw/2, gy - gw/2), min(py + pw/2, gy + gw/2)
    
    if xi2 < xi1 or yi2 < yi1: return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    union = pl * pw + gl * gw - inter
    return inter / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/training')
    parser.add_argument('--num_train', type=int, default=2000)
    parser.add_argument('--num_viz', type=int, default=200)
    args = parser.parse_args()
    
    calib = KITTICalibration(os.path.join(args.data_path, 'calib', '000000.txt'))
    bin_files = sorted(glob.glob(os.path.join(args.data_path, 'velodyne', '*.bin')))
    label_dir = os.path.join(args.data_path, 'label_2')
    
    print("\n" + "="*60)
    print("SCOPE V2: RANDOM FOREST EVALUATION (10m RANGE)")
    print("="*60)
    
    print("\n[Training Phase]")
    X_train, Y_train = [], []
    for i, bin_file in enumerate(bin_files[:args.num_train]):
        if i % 400 == 0: print(f"Processing training sample {i}/{args.num_train}...")
        
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        sample_id = os.path.basename(bin_file)[:-4]
        objects = load_kitti_label(os.path.join(label_dir, f"{sample_id}.txt"), calib)
        
        if objects:
            feat = extract_occupancy_grid(points)
            obj = objects[0]
            label = np.array([obj['center'][0], obj['center'][1], obj['center'][2],
                              obj['size'][0], obj['size'][1], obj['size'][2],
                              np.sin(obj['rotation']), np.cos(obj['rotation'])])
            X_train.append(feat)
            Y_train.append(label)
            
    rf_model = train_rf(np.array(X_train), np.array(Y_train))
    print(f"Model trained successfully on {len(X_train)} near-field samples.")
    
    print("\n[Evaluation Phase]")
    ious, depth_errors = [], []
    eval_start, eval_end = args.num_train, args.num_train + args.num_viz
    
    for i, bin_file in enumerate(bin_files[eval_start:eval_end]):
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        sample_id = os.path.basename(bin_file)[:-4]
        objects = load_kitti_label(os.path.join(label_dir, f"{sample_id}.txt"), calib)
        
        if objects:
            feat = extract_occupancy_grid(points)
            pred = predict_rf(rf_model, feat)
            
            pred_box = pred[:6]
            gt_box = np.array([objects[0]['center'][0], objects[0]['center'][1], objects[0]['center'][2],
                               objects[0]['size'][0], objects[0]['size'][1], objects[0]['size'][2]])
            
            iou = compute_iou_bev(pred_box, gt_box)
            depth_err = abs(pred_box[0] - gt_box[0])
            
            ious.append(iou)
            depth_errors.append(depth_err)
            
            if i % 20 == 0:
                print(f"Sample {sample_id} | GT depth: {gt_box[0]:.2f}m | Pred depth: {pred_box[0]:.2f}m | Depth error: {depth_err:.2f}m | IoU: {iou:.4f}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY (10m Unseen Data)")
    print("="*60)
    if ious:
        print(f"Evaluated Samples: {len(ious)}")
        print(f"BEV IoU:           {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
        print(f"Depth Error:       {np.mean(depth_errors):.2f}m +/- {np.std(depth_errors):.2f}m")
        print(f"Max IoU:           {np.max(ious):.4f}")
        print(f"Min IoU:           {np.min(ious):.4f}")
    else:
        print("No validation objects found in the 10m range.")

if __name__ == '__main__':
    main()