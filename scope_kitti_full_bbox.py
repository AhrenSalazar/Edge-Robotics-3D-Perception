import numpy as np

X_RANGE = (0, 40)   
Y_RANGE = (-20, 20) 
Z_RANGE = (-2, 2)   
GRID_SIZE = 32

def compute_box_3d(center, size, heading):
    h, w, l = size[0], size[1], size[2]
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return corners_3d

def point_cloud_to_occupancy(points):
    """
    Converts raw points into a 256-feature vector.
    Resolution: 8x8 (horizontal) x 4 (vertical)
    """
    mask = (points[:, 0] > X_RANGE[0]) & (points[:, 0] < X_RANGE[1]) & \
           (points[:, 1] > Y_RANGE[0]) & (points[:, 1] < Y_RANGE[1]) & \
           (points[:, 2] > Z_RANGE[0]) & (points[:, 2] < Z_RANGE[1])
    pts = points[mask]
    
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    if len(pts) > 0:
        x_n = np.clip((pts[:, 0] - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]), 0, 0.99)
        y_n = np.clip((pts[:, 1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]), 0, 0.99)
        z_n = np.clip((pts[:, 2] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]), 0, 0.99)
        grid[(x_n * (GRID_SIZE - 1)).astype(int), 
             (y_n * (GRID_SIZE - 1)).astype(int), 
             (z_n * (GRID_SIZE - 1)).astype(int)] = 1.0
             
    features = []
    # 8 blocks along X (step 4), 8 blocks along Y (step 4), 4 blocks along Z (step 8)
    for x in range(0, GRID_SIZE, 4):
        for y in range(0, GRID_SIZE, 4):
            for z in range(0, GRID_SIZE, 8):
                block = grid[x:x+4, y:y+4, z:z+8]
                features.append(1.0 if np.max(block) > 0 else 0.0)
    return np.array(features)

def train_model(X, Y):
    X_b = np.concatenate((np.ones((len(X), 1)), X), axis=1)
    I = np.eye(X_b.shape[1])
    alpha = 0.5 
    return np.linalg.inv(X_b.T.dot(X_b) + alpha * I).dot(X_b.T).dot(Y)