import copy
import numba
import numpy as np
from configs import config
from utils import (
    read_points,
    bbox3d2bevcorners, 
    box_collision_test, 
    remove_pts_in_bboxes, 
    limit_period
)

def db_sample_aug(data_dict, db_sample, sample_groups):
    pts = data_dict['pts']
    gt_bboxes_3d = data_dict['gt_bboxes_3d']
    gt_labels = data_dict['gt_labels']
    gt_names = data_dict['gt_names']
    gt_difficulty = data_dict['difficulty']
    image_info = data_dict['image_info']
    calib_info = data_dict['calib_info']
    index = data_dict['index']
    
    classes = config['classes']
    
    sampled_pts, sampled_names, sampled_labels, sampled_bboxes, sampled_difficulty = [], [], [], [], []
    
    avoid_coll_boxes = copy.deepcopy(gt_bboxes_3d) # copy of all gt box in cur frame
    
    for name, value in sample_groups.items():
        sampled_num = value - np.sum(gt_names == name) # calculate how many object need to sample of each class
        if sampled_num <= 0:
            continue
        
        sampled_cls_list = db_sample[name].sample(sampled_num) # List of db object to sampled, each object is a dict (db_info in preprocess.py)
        sampled_cls_bboxes = np.array([item['box3d_lidar'] for item in sampled_cls_list], dtype=np.float32) # (number_of_bboxes, 7)
        
        # Calculate bev corners of gt bboxes and sample bboxes
        avoid_coll_boxes_bev_corners = bbox3d2bevcorners(avoid_coll_boxes) 
        sampled_cls_bboxes_bev_corners = bbox3d2bevcorners(sampled_cls_bboxes)
        
        coll_query_matrix = np.concatenate([avoid_coll_boxes_bev_corners, sampled_cls_bboxes_bev_corners], axis=0)
        coll_mat = box_collision_test(coll_query_matrix, coll_query_matrix)
        
        n_gt = len(avoid_coll_boxes_bev_corners)
        tmp_bboxes = []
        
        for i in range(n_gt, len(coll_mat)):
            if any(coll_mat[i]):
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                cur_sample = sampled_cls_list[i - n_gt]
                
                pt_path = cur_sample['path'] # path to sample db object, example root/datasets/train_gt_database/000001_Car_0.bin               
                sampled_pts_cur = read_points(pt_path) 
                
                sampled_pts_cur[:, :3] += cur_sample['box3d_lidar'][:3] # db points is in bboxes coordinate, so we have to change back to lidar coordinate
                
                sampled_pts.append(sampled_pts_cur) # add db points to gt points
                
                sampled_names.append(cur_sample['name']) # add db name
                
                sampled_labels.append(classes[cur_sample['name']]) # add db name label
                
                sampled_bboxes.append(cur_sample['box3d_lidar']) # add db box
                
                tmp_bboxes.append(cur_sample['box3d_lidar']) # update the valid box to make sure that next box will not collide 
                
                sampled_difficulty.append(cur_sample['difficulty']) # add difficulty
                
        if len(tmp_bboxes) == 0:
            tmp_bboxes = np.array(tmp_bboxes).reshape(-1, 7)
        else:
            tmp_bboxes = np.array(tmp_bboxes)
            
        avoid_coll_boxes = np.concatenate([avoid_coll_boxes, tmp_bboxes], axis=0)
        
    pts = remove_pts_in_bboxes(pts, np.stack(sampled_bboxes, axis=0))
    pts = np.concatenate([np.concatenate(sampled_pts, axis=0), pts], axis=0)
    gt_bboxes_3d = avoid_coll_boxes.astype(np.float32)
    gt_labels = np.concatenate([gt_labels, np.array(sampled_labels)], axis=0)
    gt_names = np.concatenate([gt_names, np.array(sampled_names)], axis=0)
    difficulty = np.concatenate([gt_difficulty, np.array(sampled_difficulty)], axis=0)
    
    data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels, 
            'gt_names': gt_names,
            'difficulty': difficulty,
            'image_info': image_info,
            'calib_info': calib_info,
            'index': index,
        }
    return data_dict

@numba.jit(nopython=True)
def object_noise_core(pts, gt_bboxes_3d, bev_corners, trans_vec, rot_angle, rot_mat, masks):
    '''
    pts: (N, 4)
    gt_bboxes_3d: (n_bbox, 7)
    bev_corners: ((n_bbox, 4, 2))
    trans_vec: (n_bbox, num_try, 3)
    rot_mat: (n_bbox, num_try, 2, 2)
    masks: (N, n_bbox), bool
    return: gt_bboxes_3d, pts
    '''
    # 1. select the noise of num_try for each bbox under the collision test
    n_bbox, num_try = trans_vec.shape[:2]
    
    # succ_mask: (n_bbox, ), whether each bbox can be added noise successfully. -1 denotes failure.
    succ_mask = -np.ones((n_bbox, ), dtype=np.int_)
    for i in range(n_bbox):
        for j in range(num_try):
            cur_bbox = bev_corners[i] - np.expand_dims(gt_bboxes_3d[i, :2], 0) # (4, 2) - (1, 2) -> (4, 2)
            rot = np.zeros((2, 2), dtype=np.float32)
            rot[:] = rot_mat[i, j] # (2, 2)
            trans = trans_vec[i, j] # (3, )
            cur_bbox = cur_bbox @ rot
            cur_bbox += gt_bboxes_3d[i, :2]
            cur_bbox += np.expand_dims(trans[:2], 0) # (4, 2)
            coll_mat = box_collision_test(np.expand_dims(cur_bbox, 0), bev_corners)
            coll_mat[0, i] = False
            if coll_mat.any():
                continue
            else:
                bev_corners[i] = cur_bbox # update the bev_corners when adding noise succseefully.
                succ_mask[i] = j
                break
    # 2. points and bboxes noise
    visit = {}
    for i in range(n_bbox):
        jj = succ_mask[i] 
        if jj == -1:
            continue
        cur_trans, cur_angle = trans_vec[i, jj], rot_angle[i, jj]
        cur_rot_mat = np.zeros((2, 2), dtype=np.float32)
        cur_rot_mat[:] = rot_mat[i, jj]
        for k in range(len(pts)):
            if masks[k][i] and k not in visit:
                cur_pt = pts[k] # (4, )
                cur_pt_xyz = np.zeros((1, 3), dtype=np.float32)
                cur_pt_xyz[0] = cur_pt[:3] - gt_bboxes_3d[i][:3]
                tmp_cur_pt_xy = np.zeros((1, 2), dtype=np.float32)
                tmp_cur_pt_xy[:] = cur_pt_xyz[:, :2]
                cur_pt_xyz[:, :2] = tmp_cur_pt_xy @ cur_rot_mat # (1, 2)
                cur_pt_xyz[0] = cur_pt_xyz[0] + gt_bboxes_3d[i][:3]
                cur_pt_xyz[0] = cur_pt_xyz[0] + cur_trans[:3]
                cur_pt[:3] = cur_pt_xyz[0]
                visit[k] = 1

        gt_bboxes_3d[i, :3] += cur_trans[:3]
        gt_bboxes_3d[i, 6] += cur_angle

    return gt_bboxes_3d, pts

def object_noise(data_dict, num_try, translation_std, rot_range, ratio):
    if np.random.rand() < ratio:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        n_bboxes = len(gt_bboxes_3d)
        
        trans_vec = np.random.normal(scale=translation_std, size=(n_bboxes, num_try, 3)).astype(np.float32)
        rot_angle = np.random.uniform(rot_range[0], rot_range[1], size=(n_bboxes, num_try)).astype(np.float32)
        rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)    
            
        rot_mat = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
        rot_mat = np.transpose(rot_mat, (2, 3, 1, 0))
        
        bev_corners = bbox3d2bevcorners(gt_bboxes_3d)
        masks = remove_pts_in_bboxes(pts, gt_bboxes_3d, rm=False)
        
        gt_bboxes_3d, pts = object_noise_core(
            pts=pts,
            gt_bboxes_3d=gt_bboxes_3d,
            bev_corners=bev_corners,
            trans_vec=trans_vec,
            rot_angle=rot_angle,
            rot_mat=rot_mat,
            masks=masks,
        )   
        data_dict.update({
            'gt_bboxes_3d': gt_bboxes_3d,
            'pts': pts,
        })
        
    return data_dict
        
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
    difficulty = data_dict['difficulty']

    keep_mask = np.all([
        gt_bboxes_3d[:,0] >= object_range[0], gt_bboxes_3d[:,0] <= object_range[3],
        gt_bboxes_3d[:,1] >= object_range[1], gt_bboxes_3d[:,1] <= object_range[4]
    ], axis=0)

    gt_bboxes_3d = gt_bboxes_3d[keep_mask]
    gt_labels = gt_labels[keep_mask]
    gt_names = gt_names[keep_mask]
    difficulty = difficulty[keep_mask]

    gt_bboxes_3d[:,6] = limit_period(gt_bboxes_3d[:,6], 0.5, 2*np.pi)

    data_dict.update({
        'gt_bboxes_3d': gt_bboxes_3d,
        'gt_labels': gt_labels,
        'gt_names': gt_names,
        'difficulty': difficulty
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
    data_dict = db_sample_aug(
        data_dict=data_dict,
        db_sample=database_sampled['db_sample'],
        sample_groups=database_sampled['sample_groups']
    )
    
    object_noise_cfg = data_aug_config['object_noise']
    data_dict = object_noise(
        data_dict=data_dict,
        num_try=object_noise_cfg['num_try'],
        translation_std=object_noise_cfg['translation_std'],
        rot_range=object_noise_cfg['rot_range'],
        ratio=0.5,
    )
    
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