# common/ho3dutils.py

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import torch
import trimesh
from pytorch3d.io import load_obj

# 환경변수로 캐시 경로 지정 가능
CACHE_MESH_DIR = os.environ.get('HO3D_LOAD_CACHE',
    '/data/kyne0127/proj/PhysicsNDF/data/mesh_cache_resampled')

# 전처리된 SDF root 디렉터리
PROCESSED_CROSS_SDF_ROOT = '/data/kyne0127/proj/PhysicsNDF/data/processed_cross_sdf_resampled'

def list_scenes(root_dir: str) -> List[str]:
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

def list_trials(root_dir: str, scene: str) -> List[str]:
    scene_path = os.path.join(root_dir, scene)
    return sorted(
        t for t in os.listdir(scene_path)
        if os.path.isdir(os.path.join(scene_path, t))
    )

def _load_or_cache_mesh(path: str, cache_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    .obj 파일(path)을 파싱해서 (verts, faces) 반환.
    cache_path에 토치 파일이 없으면 한 번만 파싱 후 저장, 있으면 저장된 파일 로드.
    """
    if os.path.exists(cache_path):
        loaded = torch.load(cache_path)
        return loaded['verts'], loaded['faces']
    # cache가 없으면 파싱
    verts, faces, _ = load_obj(path)  # verts: FloatTensor, faces: struct
    v_np = verts.numpy().astype(np.float32)
    f_np = faces.verts_idx.numpy().astype(np.int64)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({'verts': v_np, 'faces': f_np}, cache_path)
    return v_np, f_np

def load_trial_data(root_dir: str, scene: str, trial: str) -> Dict[str, Any]:
    trial_dir = os.path.join(root_dir, scene, trial)
    data: Dict[str, Any] = {}

    # 1) sim_data.json
    sim_path = os.path.join(trial_dir, 'sim_data.json')
    with open(sim_path, 'r') as f:
        sim = json.load(f)
    data['trans'] = np.array(sim.get('trans', []), dtype=np.float32)
    data['rot']   = np.array(sim.get('rot', []), dtype=np.float32)

    # 2) lhand.json, rhand.json → MANO 파라미터
    for part in ('lhand', 'rhand'):
        mano_path = os.path.join(trial_dir, f'{part}.json')
        with open(mano_path, 'r') as f:
            mano = json.load(f)
        # 배열화
        data[f'{part}_global_orient'] = np.array(mano['global_orient'], dtype=np.float32)
        data[f'{part}_hand_pose']     = np.array(mano['hand_pose'],     dtype=np.float32)
        data[f'{part}_betas']         = np.array(mano['betas'],         dtype=np.float32)
        data[f'{part}_transl']        = np.array(mano['transl'],        dtype=np.float32)

    # 3) 기본 메쉬 (lhand, rhand, obj) 캐시 로드
    for part in ('lhand', 'rhand'):
        mesh_file  = os.path.join(trial_dir, f'{part}.obj')
        cache_file = os.path.join(CACHE_MESH_DIR, scene, trial, f'{part}.pt')
        verts_np, faces_np = _load_or_cache_mesh(mesh_file, cache_file)
        data[f'{part}_verts'] = verts_np
        data[f'{part}_faces'] = faces_np

    # 5) 교차 SDF
    cross_path = os.path.join(PROCESSED_CROSS_SDF_ROOT, scene, trial, 'cross_sdf.npz')
    if not os.path.exists(cross_path):
        raise FileNotFoundError(f"Cross-mesh SDF not found: {cross_path}")
    with np.load(cross_path) as npz:
        data['distance']          = float(npz['distance'])
        data['lhand_to_obj_sdf']  = npz['lhand_to_obj_sdf'].astype(np.float32)
        data['rhand_to_obj_sdf']  = npz['rhand_to_obj_sdf'].astype(np.float32)
        data['obj_to_lhand_sdf']  = npz['obj_to_lhand_sdf'].astype(np.float32)
        data['obj_to_rhand_sdf']  = npz['obj_to_rhand_sdf'].astype(np.float32)
        data['obj_verts']         = npz['obj_verts'].astype(np.float32)  # (1000,3)

    return data



def load_all_trials(root_dir: str) -> Dict[str, Dict[str, Any]]:
    all_data: Dict[str, Dict[str, Any]] = {}
    for scene in list_scenes(root_dir):
        for trial in list_trials(root_dir, scene):
            key = f"{scene}/{trial}"
            all_data[key] = load_trial_data(root_dir, scene, trial)
    return all_data


def load_cross_sdf(root_dir: str, scene: str, trial: str) -> Dict[str, np.ndarray]:
    """
    Load precomputed cross-mesh SDFs and drop distance from:
      {root_dir}/{scene}/{trial}/cross_sdf.npz

    Returns dict with:
      - 'distance':         float
      - 'lhand_to_obj_sdf': np.ndarray (Vh,)
      - 'rhand_to_obj_sdf': np.ndarray (Vh,)
      - 'obj_to_lhand_sdf': np.ndarray (Vo,)
      - 'obj_to_rhand_sdf': np.ndarray (Vo,)
    """
    path = os.path.join(root_dir, scene, trial, 'cross_sdf.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cross-mesh SDF not found: {path}")
    with np.load(path) as npz:
        return {
            'distance':         float(npz['distance']),
            'lhand_to_obj_sdf': npz['lhand_to_obj_sdf'],
            'rhand_to_obj_sdf': npz['rhand_to_obj_sdf'],
            'obj_to_lhand_sdf': npz['obj_to_lhand_sdf'],
            'obj_to_rhand_sdf': npz['obj_to_rhand_sdf'],
        }
