import numpy as np
from configs import config
from utils import (
    limit_period,
)
     
def global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std, ratio):
    if np.random.rand() < ratio:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        
        # rotation
        rot_angle = np.random.uniform(rot_range[0], rot_range[1])
        rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
        
        gt_bboxes_3d[:, :2] = gt_bboxes_3d[:, :2] @ rot_mat.T
        gt_bboxes_3d[:, 6] += rot_angle
        pts[:, :2] = pts[:, :2] @ rot_mat.T
        
        # scale
        scale_factor = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])
        gt_bboxes_3d[:, :6] *= scale_factor
        pts[:, :3] *= scale_factor
        
        # translate
        trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
        gt_bboxes_3d[:, :3] += trans_factor
        pts[:, :3] += trans_factor
        
        data_dict.update({
            'pts': pts, 
            'gt_bboxes_3d': gt_bboxes_3d
        })
    
    return data_dict

def random_flip(data_dict, ratio):
    if np.random.rand() < ratio:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        # Flip along Y axis
        pts[:, 1] = -pts[:, 1]
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi

        data_dict.update({
            'pts': pts, 
            'gt_bboxes_3d': gt_bboxes_3d
        })
    
    return data_dict

def point_range_filter(data_dict, point_range):
    pts = data_dict['pts']
    keep_mask = np.all([
        pts[:, 0] > point_range[0], pts[:, 0] < point_range[3],
        pts[:, 1] > point_range[1], pts[:, 1] < point_range[4],
        pts[:, 2] > point_range[2], pts[:, 2] < point_range[5]
    ], axis=0)
    
    data_dict.update({
        'pts': pts[keep_mask]
    })
    
    return data_dict

def object_range_filter(data_dict, object_range):
    gt_bboxes_3d = data_dict['gt_bboxes_3d']
    gt_labels = data_dict['gt_labels']
    gt_names = data_dict['gt_names']

    keep_mask = np.all([
        gt_bboxes_3d[:,0] >= object_range[0], gt_bboxes_3d[:,0] <= object_range[3],
        gt_bboxes_3d[:,1] >= object_range[1], gt_bboxes_3d[:,1] <= object_range[4]
    ], axis=0)

    gt_bboxes_3d = gt_bboxes_3d[keep_mask]
    gt_labels = gt_labels[keep_mask]
    gt_names = gt_names[keep_mask]

    gt_bboxes_3d[:,6] = limit_period(gt_bboxes_3d[:,6], 0.5, 2*np.pi)

    data_dict.update({
        'gt_bboxes_3d': gt_bboxes_3d,
        'gt_labels': gt_labels,
        'gt_names': gt_names,
    })
    
    return data_dict  

def points_shuffle(data_dict):
    pts = data_dict['pts']
    indices = np.arange(0, len(pts))
    np.random.shuffle(indices)
    pts = pts[indices]
    
    data_dict.update({
        'pts': pts
    })
    
    return data_dict
          
def train_data_aug(data_dict, data_aug_config):
    database_sampled = data_aug_config['database_sampled']
    
    global_rot_scale_trans_cfg = data_aug_config['global_rot_scale_trans']
    data_dict = global_rot_scale_trans(
        data_dict=data_dict,
        rot_range=global_rot_scale_trans_cfg['rot_range'],
        scale_ratio_range=global_rot_scale_trans_cfg['scale_ratio_range'],
        translation_std=global_rot_scale_trans_cfg['translation_std'],
        ratio=0.5,
    )

    data_dict = random_flip(
        data_dict=data_dict,
        ratio=0.5,
    )
       
    data_dict = point_range_filter(
        data_dict=data_dict,
        point_range=data_aug_config['point_range_filter'],
    )
    
    data_dict = object_range_filter(
        data_dict=data_dict,
        object_range=data_aug_config['object_range_filter'],
    )
    
    data_dict = points_shuffle(
        data_dict=data_dict,
    )
    
    return data_dict
    
def val_data_aug(data_dict, data_aug_config):
    data_dict = point_range_filter(
        data_dict=data_dict,
        point_range=data_aug_config['point_range_filter'],
    )
    
    data_dict = object_range_filter(
        data_dict=data_dict,
        object_range=data_aug_config['object_range_filter'],
    )    
    
    return data_dict