# common/preprocess_cache.py

import os
import argparse
import torch
import trimesh

def main(root_dir: str, cache_dir: str):
    for scene in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path): continue
        for trial in sorted(os.listdir(scene_path)):
            trial_path = os.path.join(scene_path, trial)
            if not os.path.isdir(trial_path): continue

            # parts 순회: lhand, rhand, obj
            for part in ('lhand', 'rhand', 'obj'):
                mesh_path = os.path.join(trial_path, f'{part}.obj')
                if not os.path.exists(mesh_path):
                    print(f"[WARN] Missing {mesh_path}, skipping")
                    continue

                mesh = trimesh.load(mesh_path, process=False)
                verts = torch.from_numpy(mesh.vertices.astype('float32'))

                cache_p = os.path.join(cache_dir, scene, trial, f'{part}_verts.pt')
                os.makedirs(os.path.dirname(cache_p), exist_ok=True)
                torch.save(verts, cache_p)
                print(f"[INFO] Cached {cache_p}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',  required=True,
                   help='HO3D root with scene/trial folders')
    p.add_argument('--cache_dir', required=True,
                   help='Where to save cached tensors')
    args = p.parse_args()
    main(args.root_dir, args.cache_dir)
