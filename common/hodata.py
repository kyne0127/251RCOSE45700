# File: common/hodata.py

import os
import json
import time
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset

from .ho3dutils import list_scenes, list_trials, load_trial_data

# 메쉬 버텍스 캐시 디렉터리 (인덱스 저장용)
CACHE_ROOT = os.environ.get(
    'HO3D_MESH_CACHE',
    '/data/kyne0127/proj/PhysicsNDF/data/mesh_cache_resampled'
)
INDEX_FILE = os.path.join(CACHE_ROOT, 'scene_trial_index.json')


class HO3DTrialDataset(Dataset):
    """
    Genesis HO3D Trial-level Dataset.
    모든 전처리·캐싱 로직은 ho3dutils.load_trial_data 내부에서 처리됩니다.
    """

    def __init__(
        self,
        root_dir: str,
        scenes: Optional[List[str]] = None,
        trials: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.root_dir = root_dir

        # 1) scene/trial 인덱스 로드 또는 생성
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'r') as f:
                all_pairs = json.load(f)
            print(f"[DEBUG] Loaded index from {INDEX_FILE} ({len(all_pairs)} entries)")
        else:
            all_pairs = []
            for scene in list_scenes(root_dir):
                for trial in list_trials(root_dir, scene):
                    all_pairs.append([scene, trial])
            os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
            with open(INDEX_FILE, 'w') as f:
                json.dump(all_pairs, f)
            print(f"[DEBUG] Scanned filesystem and wrote index to {INDEX_FILE} ({len(all_pairs)} entries)")

        # 2) 필터링
        self.scenes = scenes or sorted({s for s, _ in all_pairs})
        self.pairs: List[Tuple[str, str]] = []
        for scene, trial in all_pairs:
            if scene not in self.scenes:
                continue
            if trials and scene in trials and trial not in trials[scene]:
                continue
            self.pairs.append((scene, trial))
        print(f"[DEBUG] Using {len(self.pairs)} (scene,trial) pairs after filter")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        scene, trial = self.pairs[idx]
        try:
            t0_total = time.perf_counter()

            # ho3dutils.load_trial_data 안에서
            # • sim_data.json
            # • lhand.json, rhand.json (MANO 파라미터)
            # • mesh(.obj) → 캐시된 .pt 로드
            # • cross_sdf.npz
            # 전부 처리됨
            t0 = time.perf_counter()
            data = load_trial_data(self.root_dir, scene, trial)
            t1 = time.perf_counter()
            # print(f"[DEBUG] [{scene}/{trial}] load_trial_data: {t1-t0:.4f}s (source: {__file__})")

            t_total = time.perf_counter() - t0_total
            # print(f"[DEBUG] [{scene}/{trial}] total __getitem__: {t_total:.4f}s (source: {__file__})")

        except FileNotFoundError:
            return None

        data['scene'] = scene
        data['trial'] = trial
        return data

    def get_trial_index(self, scene: str, trial: str) -> int:
        try:
            return self.pairs.index((scene, trial))
        except ValueError:
            raise KeyError(f"(scene,trial)=({scene},{trial}) not in dataset")
