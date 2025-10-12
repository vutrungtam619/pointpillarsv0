import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from configs import config
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import (
    read_calib,
    read_points,
    read_label,
    write_pickle,
    write_points,
    get_points_num_in_bbox,
    points_in_bboxes_v2,
    bbox_camera2lidar,
)

root = Path(__file__).parent

def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int32)

def filter_points_by_range(pts, point_range):
    keep_mask = np.all([
        pts[:, 0] > point_range[0], pts[:, 0] < point_range[3],
        pts[:, 1] > point_range[1], pts[:, 1] < point_range[4],
        pts[:, 2] > point_range[2], pts[:, 2] < point_range[5]
    ], axis=0)
    
    return pts[keep_mask]

def filter_objects_by_range(annotation_dict, object_range, tr_velo_to_cam, r0_rect):
    locs = annotation_dict['locations']
    dims = annotation_dict['dimensions']
    rots = annotation_dict['rotation_y']

    bboxes_cam = np.concatenate([locs, dims, rots[:, None]], axis=1)
    bboxes_lidar = bbox_camera2lidar(bboxes_cam, tr_velo_to_cam, r0_rect)

    keep_mask = np.all([
        bboxes_lidar[:, 0] >= object_range[0], bboxes_lidar[:, 0] <= object_range[3],
        bboxes_lidar[:, 1] >= object_range[1], bboxes_lidar[:, 1] <= object_range[4],
        bboxes_lidar[:, 2] >= object_range[2], bboxes_lidar[:, 2] <= object_range[5]
    ], axis=0)

    filtered_dict = {}
    for key, val in annotation_dict.items():
        if isinstance(val, np.ndarray) and val.shape[0] == locs.shape[0]:
            filtered_dict[key] = val[keep_mask]
        else:
            filtered_dict[key] = val

    return filtered_dict  

def process_one_idx(idx, data_root, split, label, lidar_reduced_folder, db=False, db_points_folder=None):
    cur_info_dict = {}
    dbinfos_part = {}

    image_path = Path(data_root) / split / 'image_2' / f'{idx}.png'
    lidar_path = Path(data_root) / split / 'velodyne' / f'{idx}.bin'
    calib_path = Path(data_root) / split / 'calib' / f'{idx}.txt'

    cur_info_dict['index'] = idx

    image = cv2.imread(str(image_path))
    image_shape = image.shape[:2]
    cur_info_dict['image'] = {
        'image_shape': image_shape,
        'image_path': Path(*image_path.parts[-3:]) # training/image/000000.jpg
    }
    
    calib_dict = read_calib(calib_path)
    cur_info_dict['calib'] = calib_dict
    
    lidar_reduced_path = Path(lidar_reduced_folder) / f'{idx}.bin'
    lidar_points = read_points(lidar_path)
    reduced_points = filter_points_by_range(pts=lidar_points, point_range=config['point_cloud_range'])
    write_points(lidar_reduced_path, reduced_points)
    
    cur_info_dict['lidar'] = {
        'lidar_total': reduced_points.shape[0],
        'lidar_path': lidar_reduced_path,
    }
    
    if label:
        label_path = Path(data_root) / split / 'label_2' / f'{idx}.txt'
        annotation_dict = read_label(label_path)
        annotation_dict = filter_objects_by_range(annotation_dict, config['point_cloud_range'], calib_dict['tr_velo_to_cam'], calib_dict['r0_rect'])
        annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
        annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
            points=lidar_points,
            r0_rect=calib_dict['r0_rect'],
            tr_velo_to_cam=calib_dict['tr_velo_to_cam'],
            dimensions=annotation_dict['dimensions'],
            locations=annotation_dict['locations'],
            rotation_y=annotation_dict['rotation_y'],
            names=annotation_dict['names']  
        ) 
        cur_info_dict['annos'] = annotation_dict
             
        if db and db_points_folder is not None:
            indices, n_total_bbox, n_valid_bbox, bboxes_lidar, names = points_in_bboxes_v2(
                points=lidar_points,
                r0_rect=calib_dict['r0_rect'].astype(np.float32), 
                tr_velo_to_cam=calib_dict['tr_velo_to_cam'].astype(np.float32),
                dimensions=annotation_dict['dimensions'].astype(np.float32),
                locations=annotation_dict['locations'].astype(np.float32),
                rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                names=annotation_dict['names']    
            )
            for j in range(n_valid_bbox):
                db_points = lidar_points[indices[:, j]]
                db_points[:, :3] -= bboxes_lidar[j, :3]
                db_points_saved_name = db_points_folder / f'{int(idx)}_{names[j]}_{j}.bin'
                write_points(db_points_saved_name, db_points)
                db_info = {
                    'name': names[j],
                    'path': str(db_points_saved_name), # root/datasets/train_gt_database/000001_Car_0.bin
                    'box3d_lidar': bboxes_lidar[j],
                    'difficulty': annotation_dict['difficulty'][j], 
                    'num_points_in_gt': len(db_points), 
                }
                if names[j] not in dbinfos_part:
                    dbinfos_part[names[j]] = [db_info]
                else:
                    dbinfos_part[names[j]].append(db_info)
    
    return int(idx), cur_info_dict, dbinfos_part


def create_data_info_pkl(data_root, data_type, label):
    print(f"Processing {data_type} data into pkl file....")
    
    index_files = Path(root) / 'index' / f'{data_type}.txt'
    ids = index_files.read_text(encoding="utf-8").splitlines()
    
    split = 'training' if label else 'testing'
    
    lidar_reduced_folder = Path(root) / 'datasets' / 'point_cloud_reduced'
    lidar_reduced_folder.mkdir(exist_ok=True)
    
    db_flag = True if data_type == 'train' else False
    db_points_folder = Path(root) / 'datasets' / 'train_gt_database'
    if db_flag:
        db_points_folder.mkdir(exist_ok=True)
        
    infos_dict = {}
    dbinfos_train = {}
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_one_idx, idx, data_root, split, label, lidar_reduced_folder, db=db_flag, db_points_folder=db_points_folder if db_flag else None): idx
            for idx in ids
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, cur_info_dict, dbinfos_part = future.result()
            infos_dict[idx] = cur_info_dict

            for k, v in dbinfos_part.items():
                if k not in dbinfos_train:
                    dbinfos_train[k] = v
                else:
                    dbinfos_train[k].extend(v)

    save_pkl_path = Path(root) / 'datasets' / f'infos_{data_type}.pkl'
    write_pickle(save_pkl_path, infos_dict)  
    
    if db_flag:
        save_db_path = Path(root) / 'datasets' / f'infos_{data_type}_database.pkl'
        write_pickle(save_db_path, dbinfos_train)
    
    return infos_dict
  
    
def main(args):
    data_root = args.data_root
    
    create_data_info_pkl(data_root, data_type='train', label=True)
    create_data_info_pkl(data_root, data_type='val', label=True)

    print("......Processing finished!!!")  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default=config['data_root'])
    args = parser.parse_args()

    main(args)