import argparse
import numpy as np
import torch
import open3d as o3d
import time
import os
from models.pointpillars import PointPillars
from configs import config


class LidarPlayer:
    def __init__(self, bin_dir, model_path):
        self.bin_dir = bin_dir
        self.files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
        if not self.files:
            raise FileNotFoundError(f"No .bin files found in {bin_dir}")

        # Model
        self.model = PointPillars(nclasses=config["num_classes"]).cuda()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt["checkpoint"])
        self.model.eval()

        # Visualization
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="KITTI LiDAR Player", width=1280, height=720)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.frame_idx = 0

        # Box list
        self.boxes = []

        # Key bindings
        self.vis.register_key_callback(ord(" "), self.next_frame)
        self.vis.register_key_callback(256, self.exit)

        # Load first frame
        self.load_frame(0, reset_camera=True)

    def clear_boxes(self):
        for box in self.boxes:
            self.vis.remove_geometry(box, reset_bounding_box=False)
        self.boxes = []

    def load_frame(self, idx, reset_camera=False):
        if idx >= len(self.files):
            print("✅ Finished sequence.")
            return False

        bin_path = os.path.join(self.bin_dir, self.files[idx])
        print(f"\n➡️ Frame {idx:06d}: {bin_path}")
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        # Inference
        batched_pts = [torch.from_numpy(points).cuda()]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            results = self.model(
                batched_pts=batched_pts, mode="test",
                batched_gt_bboxes=[None], batched_gt_labels=[None]
            )
        torch.cuda.synchronize()
        t1 = time.time()

        infer_time = t1 - t0
        fps = 1.0 / infer_time if infer_time > 0 else float("inf")
        print(f"   Inference time: {infer_time*1000:.2f} ms ({fps:.2f} FPS)")

        bboxes = results[0]["lidar_bboxes"]

        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        self.vis.update_geometry(self.pcd)

        # Remove old boxes
        self.clear_boxes()

        # Add new boxes
        for box in bboxes:
            center = box[:3].copy()
            l, w, h = box[3:6]
            yaw = box[6]
            center[2] += h / 2  # lift to ground
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
            obb = o3d.geometry.OrientedBoundingBox(center, R, [l, w, h])
            obb.color = (1, 0, 0)
            self.vis.add_geometry(obb, reset_bounding_box=False)
            self.boxes.append(obb)

        if reset_camera:
            self.vis.reset_view_point(True)

        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def next_frame(self, vis):
        self.frame_idx += 1
        ok = self.load_frame(self.frame_idx, reset_camera=False)
        if not ok:
            print("End of sequence.")
        return True

    def exit(self, vis):
        print("👋 Exiting.")
        self.vis.destroy_window()
        exit(0)

    def run(self):
        self.vis.run()


def main(args):
    player = LidarPlayer(args.bin_dir, args.model_path)
    player.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI LiDAR Player")
    parser.add_argument(
        "--bin_dir",
        type=str,
        default=r"C:\SOURCE CODE\pointpillars\datasets\point_cloud_reduced",
        help="Folder containing .bin files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/epoch_100.pth",
        help="Path to .pth checkpoint"
    )
    args = parser.parse_args()
    main(args)
