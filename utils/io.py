import pickle
import numpy as np
from pathlib import Path

def read_points(file_path, dim=4, suffix='.bin'):
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        raise ValueError(f"File must be {suffix}, got {file_path.suffix}")
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)

def write_points(file_path, data: np.ndarray):
    file_path = Path(file_path)
    data.tofile(file_path)
        
def read_pickle(file_path, suffix='.pkl'):
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        raise ValueError(f"File must be {suffix}, got {file_path.suffix}")
    with file_path.open('rb') as f:
        return pickle.load(f)

def write_pickle(file_path, results):
    file_path = Path(file_path)
    with file_path.open('wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def read_calib(file_path, extend_matrix=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        r0_rect=R0_rect,
        tr_velo_to_cam=Tr_velo_to_cam,
        tr_imu_to_velo=Tr_imu_to_velo
    )
    return calib_dict
    
def read_label(file_path, suffix=".txt"):
    """ Read label file, each line is one object in image """
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        raise ValueError(f"File must be {suffix}, got {file_path.suffix}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:  
        return {
            "names": np.array([], dtype=str),
            "truncated": np.array([], dtype=np.float32),
            "occluded": np.array([], dtype=np.int32),
            "alpha": np.array([], dtype=np.float32),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "locations": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.array([], dtype=np.float32),
        }

    lines = [line.split() for line in text.splitlines()]
    return {
        "names": np.array([l[0] for l in lines]),
        "truncated": np.array([l[1] for l in lines], dtype=np.float32),
        "occluded": np.array([l[2] for l in lines], dtype=np.int32),
        "alpha": np.array([l[3] for l in lines], dtype=np.float32),
        "bbox": np.array([l[4:8] for l in lines], dtype=np.float32),
        "dimensions": np.array([l[8:11] for l in lines], dtype=np.float32)[:, [2, 0, 1]], # HWL -> LHW
        "locations": np.array([l[11:14] for l in lines], dtype=np.float32),
        "rotation_y": np.array([l[14] for l in lines], dtype=np.float32),
    }

def write_label(file_path, result):
    file_path = Path(file_path)
    with file_path.open("w", encoding="utf-8") as f:
        num_objects = len(result["names"])
        if num_objects == 0:
            return  
        
        for i in range(num_objects):
            line = (
                f'{result["names"][i]} '
                f'{result["truncated"][i]} '
                f'{result["occluded"][i]} '
                f'{result["alpha"][i]} '
                f'{" ".join(map(str, result["bbox"][i]))} '
                f'{" ".join(map(str, result["dimensions"][i]))} '
                f'{" ".join(map(str, result["locations"][i]))} '
                f'{result["rotation_y"][i]} '
                f'{result.get("scores", ["-1"] * num_objects)[i]}\n'
            )
            f.write(line)