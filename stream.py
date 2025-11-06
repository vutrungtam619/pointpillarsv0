from openni import openni2
import numpy as np
import open3d as o3d
import time

# === 1️⃣ Khởi tạo OpenNI2 ===
openni2.initialize("C:/Program Files/OpenNI2/Redist")
dev = openni2.Device.open_any()

# === 2️⃣ Mở luồng Depth và Color ===
depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

# 🔹 Tắt chế độ mirror (trái ↔ phải)
depth_stream.set_mirroring_enabled(False)
color_stream.set_mirroring_enabled(False)

# 🔹 Bắt đầu stream
depth_stream.start()
color_stream.start()

# === 3️⃣ Thông số nội tại camera (cần chỉnh theo camera thực nếu có file calibration) ===
fx, fy = 262, 262
cx, cy = 160, 120
depth_scale = 1000.0  # mm → m

# === 4️⃣ Khởi tạo cửa sổ Open3D ===
vis = o3d.visualization.Visualizer()
vis.create_window("ASUS Xtion → Real-time Point Cloud (Camera View)")
pcd = o3d.geometry.PointCloud()
first_frame = True

# === 5️⃣ Đọc thử frame đầu ===
depth_frame = depth_stream.read_frame()
depth_data = depth_frame.get_buffer_as_uint16()
depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)

color_frame = color_stream.read_frame()
color_data = color_frame.get_buffer_as_uint8()
color = np.frombuffer(color_data, dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)

h, w = depth.shape
print(f"Depth map size: {w}x{h}")
print(f"RGB image size: {color.shape[1]}x{color.shape[0]}")
print("Bắt đầu stream point cloud (Ctrl+C để dừng)")

# Chuẩn bị sẵn u,v để tiết kiệm CPU
u, v = np.meshgrid(np.arange(w), np.arange(h))
frame_count = 0
fps_timer = time.time()

try:
    while True:
        # --- Đọc depth ---
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)

        # --- Đọc color ---
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color = np.frombuffer(color_data, dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)

        if depth.max() == 0:
            continue  # bỏ frame rỗng

        # --- Tính toạ độ 3D trong hệ CAMERA ---
        Z = depth.astype(np.float32) / depth_scale
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        mask = Z > 0
        points = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
        colors_points = color.reshape(-1, 3)[mask.flatten()] / 255.0

        # --- Cập nhật point cloud ---
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_points)

        # --- Add geometry lần đầu ---
        if first_frame:
            vis.add_geometry(pcd)
            
            # === Thêm khung tọa độ tại gốc (0,0,0) ===
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            vis.add_geometry(axis)

            # 🔹 Đặt góc nhìn giống góc nhìn camera
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])     # nhìn dọc theo trục Z (ra phía trước)
            ctr.set_up([0, -1, 0])        # trục Y hướng xuống
            ctr.set_lookat([0, 0, 1])     # nhìn vào vùng trước mặt camera
            ctr.set_zoom(0.8)             # zoom vừa phải

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
