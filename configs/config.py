from pathlib import Path
root = root = Path(__file__).parent.parent

config = {
    'data_root': 'KITTICrowd',
    'checkpoint_dir': Path(root) / 'checkpoints',
    'log_dir': Path(root) / 'logs',
    'saved_path': Path(root) / 'results',
    'classes': {'Pedestrian': 0},
    'num_classes': 1,
    'point_cloud_range': [0, -20.48, -3, 30.72, 20.48, 1],
    'voxel_size': [0.12, 0.16, 4],
    'max_voxels': (20000, 20000),
    'max_points': 64,
    'batch_size_train': 4,
    'batch_size_val': 4,
    'num_workers': 4,
    'init_lr': 0.00025,
    'epoch': 100,
    'ckpt_freq': 2,
}