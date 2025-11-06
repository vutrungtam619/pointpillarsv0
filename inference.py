import argparse
import numpy as np
import torch
import open3d as o3d
import time
from openni import openni2
from models.pointpillars import PointPillars
from configs import config

# ==========================
# 1. Tham số
# ==========================
lidar_height = 0.85  # m
fx, fy = 525.0, 525.0
cx, cy = 160, 120
depth_scale = 1000.0  # mm -> m

# ==========================
# 2. Initialize OpenNI
# ==========================
openni2.initialize("C:/Program Files/OpenNI2/Redist")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
color_stream = dev.create_color_stream()
color_stream.start()

# ==========================
# 3. Load PointPillars model
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointPillars(nclasses=config['num_classes']).to(device)
checkpoint = torch.load("checkpoints/epoch_128.pth")
model.load_state_dict(checkpoint["checkpoint"])
model.eval()

# Warm-up
with torch.no_grad():
    dummy_pts = [torch.zeros((1,3), device=device)]
    for _ in range(5):
        _ = model(batched_pts=dummy_pts, mode='test',
                  batched_gt_bboxes=[None], batched_gt_labels=[None])

# ==========================
# 4. Visualization helper
# ==========================
def visualize(points, boxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries = [pcd]

    for box in boxes:
        center = box[:3].copy()
        l, w, h = box[3:6]
        yaw = box[6]
        center[2] += h/2  # nâng bbox
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,yaw])
        obb = o3d.geometry.OrientedBoundingBox(center, R, [l,w,h])
        obb.color = (1,0,0)
        geometries.append(obb)

    o3d.visualization.draw_geometries(geometries)

# ==========================
# 5. Realtime loop
# ==========================
vis = o3d.visualization.Visualizer()
vis.create_window("Realtime PointPillars KITTI")
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

try:
    while True:
        # --- Depth ---
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)

        # --- Color ---
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color = np.frombuffer(color_data, dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)

        if depth.max() == 0:
            continue

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth.astype(np.float32)/depth_scale
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        mask = Z > 0
        points_kitti = np.stack([Z[mask], X[mask], -Y[mask] + lidar_height], axis=-1)
        colors_points = color[mask]/255.0

        # Update point cloud visualization
        pcd.points = o3d.utility.Vector3dVector(points_kitti)
        pcd.colors = o3d.utility.Vector3dVector(colors_points)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # --- PointPillars inference ---
        pts_tensor = torch.from_numpy(points_kitti).float().cuda()
        batched_pts = [pts_tensor]

        with torch.no_grad():
            results = model(batched_pts=batched_pts, mode='test',
                            batched_gt_bboxes=[None], batched_gt_labels=[None])

        bboxes = results[0]['lidar_bboxes']
        if len(bboxes) > 0:
            visualize(points_kitti, bboxes)

        time.sleep(0.03)

except KeyboardInterrupt:
    print("\nStopped realtime inference.")

finally:
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    vis.destroy_window()
