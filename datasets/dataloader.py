import torch
from torch.utils.data import DataLoader

def collate_fn(list_data):
    batched_pts_list = []
    batched_gt_bboxes_list = []
    batched_labels_list = []
    batched_names_list = []
    batched_image_info_list = []
    batched_calib_list = []
    batched_idx_list = []
    
    for data_dict in list_data:
        pts = data_dict['pts']
        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        gt_labels = data_dict['gt_labels']
        gt_names = data_dict['gt_names']
        image_info = data_dict['image_info'] 
        calib_info = data_dict['calib_info']
        idx = data_dict['index']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names) 
        batched_image_info_list.append(image_info)
        batched_calib_list.append(calib_info)
        batched_idx_list.append(idx)
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list, 
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_image_info=batched_image_info_list,
        batched_calibs=batched_calib_list,
        batched_idx=batched_idx_list
    )

    return rt_data_dict

def get_train_dataloader(dataset, batch_size, num_workers, drop_last=False, shuffle=True):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader


def get_val_dataloader(dataset, batch_size, num_workers, drop_last=False, shuffle=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader