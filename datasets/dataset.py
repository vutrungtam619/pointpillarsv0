import numpy as np
from pathlib import Path
from configs import config
from torch.utils.data import Dataset
from utils import read_pickle, read_points, bbox_camera2lidar
from datasets import train_data_aug, val_data_aug

root = Path(__file__).parent.parent

class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret
    
class Kitti(Dataset): 
    def __init__(self, data_root, split):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.data_infos = read_pickle(Path(root) / 'datasets' / f'infos_{split}.pkl')
        self.sorted_ids = list(self.data_infos.keys())        
        self.classes = config['classes']
        if split == 'train':
            db_infos = read_pickle(Path(root) / 'datasets' / f'infos_{split}_database.pkl')
            db_infos = self.filter_db(db_infos)
            
            db_sample = {}
            for cat_name in self.classes:
                db_sample[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
            
            self.data_aug_config = {
                'database_sampled': {'db_sample': db_sample, 'sample_groups': {'Car': 10, 'Pedestrian': 15, 'Cyclist': 10},},
                'object_noise': {'num_try': 100, 'translation_std': [0.25, 0.25, 0.25], 'rot_range': [-0.15707963267, 0.15707963267]},
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
    
    def remove_label(self, annos_info):
        # Remove label that not use in training
        keep_ids = [i for i, name in enumerate(annos_info['names']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info
    
    def filter_db(self, db_infos): # Filter the database that too hard or not enough point to training
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]
        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.classes:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos
    
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
        
        annos_info = self.remove_label(annos_info)
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
            'difficulty': annos_info['difficulty'],
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