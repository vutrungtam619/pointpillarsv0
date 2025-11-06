from openni import openni2
import numpy as np
import open3d as o3d
import time

# === 1️⃣ Khởi tạo OpenNI2 ===
openni2.initialize("C:/Program Files/OpenNI2/Redist")
dev = openni2.Device.open_any()

# === 2️⃣ Mở luồng Depth và Color ===
depth_stream = dev.create_depth_stream()
depth_stream.start()
color_stream = dev.create_color_stream()
color_stream.start()

# === 3️⃣ Thông số nội tại camera ===
fx, fy = 525.0, 525.0
cx, cy = 160, 120
depth_scale = 1000.0  # mm -> m

# === 4️⃣ Khởi tạo cửa sổ Open3D ===
vis = o3d.visualization.Visualizer()
vis.create_window("ASUS Xtion → Real-time Point Cloud (KITTI)")
pcd = o3d.geometry.PointCloud()
first_frame = True  # dùng để add geometry lần đầu

# === 5️⃣ Kiểm tra kích thước ảnh RGB và Depth map ===
depth_frame = depth_stream.read_frame()
depth_data = depth_frame.get_buffer_as_uint16()
depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)

color_frame = color_stream.read_frame()
color_data = color_frame.get_buffer_as_uint8()
color = np.frombuffer(color_data, dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)

h, w = depth.shape
print(f"Depth map size: {w}x{h}")
print(f"RGB image size: {color.shape[1]}x{color.shape[0]}")
print("Bắt đầu stream point cloud với màu RGB (Ctrl+C để dừng)")

# === 6️⃣ Vòng lặp realtime với FPS ===
frame_count = 0
fps_timer = time.time()

try:
    while True:
        # --- Depth frame ---
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)

        # --- Color frame ---
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color = np.frombuffer(color_data, dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)

        if depth.max() == 0:
            continue  # bỏ frame rỗng

        # --- Chuyển sang hệ trục KITTI ---
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth.astype(np.float32) / depth_scale
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        mask = Z > 0
        points_kitti = np.stack([Z[mask], X[mask], -Y[mask]], axis=-1)
        colors_points = color.reshape(-1, 3)[mask.flatten()] / 255.0

        pcd.points = o3d.utility.Vector3dVector(points_kitti)
        pcd.colors = o3d.utility.Vector3dVector(colors_points)

        # --- Add geometry lần đầu ---
        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False

        # --- Cập nhật renderer ---
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # --- Tính FPS trung bình mỗi 5s ---
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 5.0:
            fps = frame_count / elapsed
            print(f"FPS trung bình trong 5s: {fps:.2f}")
            frame_count = 0
            fps_timer = time.time()

except KeyboardInterrupt:
    print("\nDừng hiển thị realtime.")

finally:
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    vis.destroy_window()
