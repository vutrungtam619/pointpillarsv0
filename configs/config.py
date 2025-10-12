from pathlib import Path
root = root = Path(__file__).parent.parent

config = {
    'data_root': 'Kitti',
    'checkpoint_dir': Path(root) / 'checkpoints',
    'log_dir': Path(root) / 'logs',
    'saved_path': Path(root) / 'results',
    'classes': {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2},
    'num_classes': 3,
    'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1],
    'voxel_size': [0.16, 0.16, 4],
    'max_voxels': (16000, 16000),
    'max_points': 32,
    'batch_size_train': 4,
    'batch_size_val': 4,
    'num_workers': 4,
    'init_lr': 0.00025,
    'epoch': 100,
    'ckpt_freq': 2,
}