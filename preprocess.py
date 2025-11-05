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
    bbox_camera2lidar,
    get_points_num_in_bbox,
)

root = Path(__file__).parent

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

def filter_bboxes_by_points(points, annotation_dict, calib_dict, min_points=10):
    """
    Lọc bỏ các bbox có ít hơn min_points điểm point cloud.
    
    Args:
        points [np.ndarray float32, (N, 4)]: Điểm LiDAR.
        annotation_dict [dict]: Bao gồm keys 'dimensions', 'locations', 'rotation_y', 'names', ...
        calib_dict [dict]: Gồm 'r0_rect' và 'tr_velo_to_cam' để chuyển đổi hệ tọa độ.
        min_points [int]: Ngưỡng tối thiểu số điểm cần có trong bbox.
    
    Returns:
        filtered_dict [dict]: annotation_dict sau khi lọc.
    """
    if len(annotation_dict['dimensions']) == 0:
        return annotation_dict

    # Đếm số điểm trong mỗi bbox
    points_num = get_points_num_in_bbox(
        points=points,
        r0_rect=calib_dict['r0_rect'],
        tr_velo_to_cam=calib_dict['tr_velo_to_cam'],
        dimensions=annotation_dict['dimensions'],
        locations=annotation_dict['locations'],
        rotation_y=annotation_dict['rotation_y'],
        names=annotation_dict['names']
    )

    # Giữ lại bbox có đủ điểm
    keep_mask = points_num >= min_points

    filtered_dict = {}
    for key, val in annotation_dict.items():
        if isinstance(val, np.ndarray) and val.shape[0] == keep_mask.shape[0]:
            filtered_dict[key] = val[keep_mask]
        else:
            filtered_dict[key] = val

    return filtered_dict

def process_one_idx(idx, data_root, split, label, lidar_reduced_folder):
    cur_info_dict = {}

    image_path = Path(data_root) / split / 'image_2' / f'{idx}.jpg'
    lidar_path = Path(data_root) / split / 'velodyne' / f'{idx}.bin'
    calib_path = Path(data_root) / split / 'camera.json'

    cur_info_dict['index'] = idx

    image = cv2.imread(str(image_path))
    image_shape = image.shape[:2]
    cur_info_dict['image'] = {
        'image_shape': image_shape,
        'image_path': Path(*image_path.parts[-3:])
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
        annotation_dict = filter_objects_by_range(
            annotation_dict, 
            config['point_cloud_range'],
            calib_dict['tr_velo_to_cam'], calib_dict['r0_rect']
        )
        annotation_dict = filter_bboxes_by_points(
            points=reduced_points,
            annotation_dict=annotation_dict,
            calib_dict=calib_dict,
            min_points=10
        )
        cur_info_dict['annos'] = annotation_dict

    return int(idx), cur_info_dict


def create_data_info_pkl(data_root, data_type, label):
    print(f"Processing {data_type} data into pkl file....")

    index_files = Path(root) / 'index' / f'{data_type}.txt'
    ids = index_files.read_text(encoding="utf-8").splitlines()
    split = 'training' if label else 'testing'

    lidar_reduced_folder = Path(root) / 'datasets' / 'point_cloud_reduced'
    lidar_reduced_folder.mkdir(exist_ok=True)

    infos_dict = {}

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_one_idx, idx, data_root, split, label, lidar_reduced_folder): idx
            for idx in ids
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, cur_info_dict = future.result()
            infos_dict[idx] = cur_info_dict

    save_pkl_path = Path(root) / 'datasets' / f'infos_{data_type}.pkl'
    write_pickle(save_pkl_path, infos_dict)

    return infos_dict


def main(args):
    data_root = args.data_root
    create_data_info_pkl(data_root, data_type='train', label=True)
    create_data_info_pkl(data_root, data_type='val', label=True)
    print("......Processing finished!!!")  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset information')
    parser.add_argument('--data_root', default=config['data_root'])
    args = parser.parse_args()
    main(args)
