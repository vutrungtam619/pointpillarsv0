import numpy as np
from pathlib import Path
from configs import config
from torch.utils.data import Dataset
from utils import read_pickle, read_points, bbox_camera2lidar
from datasets import train_data_aug, val_data_aug

root = Path(__file__).parent.parent
    
class Kitti(Dataset): 
    def __init__(self, data_root, split):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.data_infos = read_pickle(Path(root) / 'datasets' / f'infos_{split}.pkl')
        self.sorted_ids = list(self.data_infos.keys())        
        self.classes = config['classes']
        if split == 'train':          
            self.data_aug_config = {
                'global_rot_scale_trans': {'rot_range': [-0.78539816, 0.78539816], 'scale_ratio_range': [0.95, 1.05], 'translation_std': [0, 0, 0]},
                'point_range_filter': config['point_cloud_range'],
                'object_range_filter': config['point_cloud_range'],
            }
        else:
            self.data_aug_config = {
                'point_range_filter': config['point_cloud_range'],
                'object_range_filter': config['point_cloud_range'],
            }           

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        """ Get the information of one item
        Args:
            index [int]: index of item

        Returns: dict with the following, m is the number of objects in item
            pts [np.ndarray float32, (n, 4)]: LiDAR points in this item
            gt_bboxes_3d [np.ndarray float32, (m, 7)]: bounding box in LiDAR coordinate
            gt_labels [np.ndarray int32, (m, )]: numerical labels for each object
            gt_names [np.ndarray string, (m, )]: object class name
            image_info [dict]: image shape in (height, width)
            calib_info [dict]: calib information
            index [int]: index sample
        """
        
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info, lidar_info = data_info['image'], data_info['calib'], data_info['annos'], data_info['lidar']
        idx = data_info['index']
        pts = read_points(lidar_info['lidar_path']).astype(np.float32)    
        
        names = annos_info['names']
        locations = annos_info['locations']
        dimensions = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        
        gt_bboxes = np.concatenate([locations, dimensions, rotation_y[:, None]], axis=1) # (m, 7) inlcude cx, cy, cz, L, H, W, rotation_y
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, calib_info['tr_velo_to_cam'], calib_info['r0_rect'])
        gt_labels = np.array([self.classes.get(name, -1) for name in names])
        
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels,
            'gt_names': names,
            'image_info': image_info,
            'calib_info': calib_info,
            'index': idx
        }
        
        if self.split in ['train', 'trainval']:
            data_dict = train_data_aug(
                data_dict=data_dict, 
                data_aug_config=self.data_aug_config,
            )
        else:
            data_dict = val_data_aug(
                data_dict=data_dict,
                data_aug_config=self.data_aug_config,
            )        
            
        return data_dict