#!/usr/bin/env python3
# common/processing_pipeline.py

"""
Processing pipeline for cross-mesh SDF and drop-distance preprocessing.
  - drop distance (first vs last frame, with rotation & translation applied)
  - cross-mesh SDF arrays (lhand->obj, rhand->obj, obj->lhand, obj->rhand)
  - vertex counts

오브젝트에 대해서는 **항상 정확히 1000개의** Poisson‐disk 샘플을 뽑아서 SDF를 계산하도록 보장한다.
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
import trimesh
import point_cloud_utils as pcu   # 포아송 디스크 샘플링
from tqdm import tqdm
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance


def sample_exact_poisson_disk(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    target_num: int = 1000,
    max_attempts: int = 5
) -> np.ndarray:
    """
    Poisson disk sampling을 반복 호출하여 **최종적으로 정확히 target_num개**의 포인트를 리턴하려 시도한다.
    1) mesh_faces, mesh_vertices를 이용해 Poisson 샘플을 뽑는다.
    2) 만약 샘플 개수가 target_num보다 크면, 무작위로 섞은 뒤 앞에서부터 target_num개만 리턴.
    3) 만약 샘플 개수가 target_num보다 작으면, 반복해서 더 많은 샘플을 시도하여 합친 뒤 필요하면 uniform random으로 부족분을 채움.
    """
    # 1) 첫 시도: target_num개를 뽑으려 시도
    face_idx, bary = pcu.sample_mesh_poisson_disk(mesh_vertices, mesh_faces, target_num)
    sampled = pcu.interpolate_barycentric_coords(mesh_faces, face_idx, bary, mesh_vertices)

    if sampled.shape[0] >= target_num:
        perm = np.random.permutation(sampled.shape[0])
        return sampled[perm[:target_num]]

    # 2) 샘플이 부족할 때: 반복해서 더 많이 시도
    accum = sampled.copy()
    for attempt in range(1, max_attempts):
        # new_target을 조금씩 늘려가며 재시도
        new_target = target_num * (attempt + 1)
        face_idx2, bary2 = pcu.sample_mesh_poisson_disk(mesh_vertices, mesh_faces, new_target)
        sampled2 = pcu.interpolate_barycentric_coords(mesh_faces, face_idx2, bary2, mesh_vertices)

        if sampled2.shape[0] >= target_num:
            perm = np.random.permutation(sampled2.shape[0])
            return sampled2[perm[:target_num]]
        else:
            # 부족하더라도 기존 accum에 합쳐 두고 다음 시도
            accum = np.concatenate([accum, sampled2], axis=0)

    # 3) max_attempts 동안에도 부족하면 uniform random으로 부족분 채움
    if accum.shape[0] < target_num:
        deficit = target_num - accum.shape[0]
        vidx = np.random.choice(mesh_vertices.shape[0], size=deficit, replace=True)
        extra = mesh_vertices[vidx]
        accum = np.concatenate([accum, extra], axis=0)

    # 4) 최종적으로 모아둔 것 중에서 랜덤하게 target_num개 리턴
    perm = np.random.permutation(accum.shape[0])
    return accum[perm[:target_num]]


def build_faces_from_sampled_points(points: torch.Tensor):
    """
    샘플링된 점(point cloud)으로부터 메쉬 face를 생성.
    Convex Hull을 사용하여 표면을 복원.
    returns: (verts, faces) as torch.Tensor
    """
    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), process=False)
    hull = mesh.convex_hull
    verts = torch.from_numpy(hull.vertices.astype(np.float32)).to(points.device)
    faces = torch.from_numpy(hull.faces.astype(np.int64)).to(points.device)
    return verts, faces


def compute_drop_distance(sim: dict, verts_o: torch.Tensor, device: torch.device) -> float:
    """
    Compute drop distance from sim data using first and last frame.
    sim: dict with 'rot' and 'trans'
    verts_o: torch.Tensor (Vo,3) on device
    returns: drop distance (float)
    """
    rot0      = torch.tensor(sim['rot'][0],   dtype=torch.float32, device=device)
    rot_end   = torch.tensor(sim['rot'][-1],  dtype=torch.float32, device=device)
    trans0    = torch.tensor(sim['trans'][0], dtype=torch.float32, device=device)
    trans_end = torch.tensor(sim['trans'][-1], dtype=torch.float32, device=device)
    center_local = verts_o.mean(dim=0)
    pos0    = rot0 @ center_local + trans0
    pos_end = rot_end @ center_local + trans_end
    return float(torch.clamp(pos0[2] - pos_end[2], min=0.0).item())


def process_all(root_dir: str, output_dir: str, gpu_id: int = 0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device} for preprocessing")

    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)

    FIXED_OBJ_SAMPLES = 1000  # 항상 1000개만 사용

    for scene in tqdm(sorted(os.listdir(root_dir)), desc='Scenes'):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        for trial in sorted(os.listdir(scene_path)):
            trial_dir = os.path.join(scene_path, trial)
            if not os.path.isdir(trial_dir):
                continue

            try:
                sim_path    = os.path.join(trial_dir, 'sim_data.json')
                obj_path    = os.path.join(trial_dir, 'obj.obj')
                lhand_path  = os.path.join(trial_dir, 'lhand.obj')
                rhand_path  = os.path.join(trial_dir, 'rhand.obj')
                if not (os.path.exists(sim_path) and os.path.exists(obj_path)
                        and os.path.exists(lhand_path) and os.path.exists(rhand_path)):
                    print(f"[WARN] Missing files in {scene}/{trial}, skipping.")
                    continue

                # 1) sim 데이터 읽기
                with open(sim_path, 'r') as f:
                    sim = json.load(f)

                # 2) 원본 .obj 로드 (hand & object)
                mesh_l = trimesh.load(lhand_path, process=False)
                mesh_r = trimesh.load(rhand_path, process=False)
                mesh_o = trimesh.load(obj_path, process=False)

                # 3) 손 메쉬 → torch.Tensor
                verts_l = torch.from_numpy(mesh_l.vertices.astype(np.float32)).to(device)
                faces_l = torch.from_numpy(mesh_l.faces.astype(np.int64)).to(device)
                verts_r = torch.from_numpy(mesh_r.vertices.astype(np.float32)).to(device)
                faces_r = torch.from_numpy(mesh_r.faces.astype(np.int64)).to(device)

                # 4) 오브젝트 메쉬 → numpy 배열
                verts_o_np = mesh_o.vertices.astype(np.float32)  # (Vo_orig, 3)
                faces_o_np = mesh_o.faces.astype(np.int64)       # (Fo_orig, 3)

                # 5) 정확히 1000개 Poisson-disk 샘플링
                sampled_o_np = sample_exact_poisson_disk(
                    verts_o_np, faces_o_np, target_num=FIXED_OBJ_SAMPLES
                )
                # → sampled_o_np.shape == (1000, 3)인 NumPy array

                verts_o = torch.from_numpy(sampled_o_np.astype(np.float32)).to(device)

                # 6) Convex Hull로 face 재구성
                _, faces_o_recon = build_faces_from_sampled_points(verts_o)

                # 7) drop distance 계산 (원본 mesh_o의 평균 높이 사용)
                dist = compute_drop_distance(sim, torch.from_numpy(verts_o_np).to(device), device)
                print(f"[DEBUG] {scene}/{trial}: drop_distance={dist:.4f}")

                # 8) 결과 dict 준비
                out = {
                    'distance':        dist,
                    'lhand_num_verts': verts_l.shape[0],
                    'rhand_num_verts': verts_r.shape[0],
                    'obj_verts':       verts_o.cpu().numpy(),
                }
                print(f"[DEBUG] {scene}/{trial}: verts_l={verts_l.shape}, verts_r={verts_r.shape}, verts_o={verts_o.shape}")

                # 9) lhand → sampled-obj SDF
                face_verts_o = index_vertices_by_faces(
                    verts_o.unsqueeze(0).contiguous(),
                    faces_o_recon
                )
                d2_l2o, _, _ = point_to_mesh_distance(
                    verts_l.unsqueeze(0).contiguous(), face_verts_o
                )
                out['lhand_to_obj_sdf'] = d2_l2o.squeeze(0).sqrt().cpu().numpy().astype(np.float32)

                # 10) rhand → sampled-obj SDF
                d2_r2o, _, _ = point_to_mesh_distance(
                    verts_r.unsqueeze(0).contiguous(), face_verts_o
                )
                out['rhand_to_obj_sdf'] = d2_r2o.squeeze(0).sqrt().cpu().numpy().astype(np.float32)

                # 11) sampled-obj → lhand SDF
                face_verts_l = index_vertices_by_faces(
                    verts_l.unsqueeze(0).contiguous(), faces_l
                )
                d2_o2l, _, _ = point_to_mesh_distance(
                    verts_o.unsqueeze(0).contiguous(), face_verts_l
                )
                out['obj_to_lhand_sdf'] = d2_o2l.squeeze(0).sqrt().cpu().numpy().astype(np.float32)

                # 12) sampled-obj → rhand SDF
                face_verts_r = index_vertices_by_faces(
                    verts_r.unsqueeze(0).contiguous(), faces_r
                )
                d2_o2r, _, _ = point_to_mesh_distance(
                    verts_o.unsqueeze(0).contiguous(), face_verts_r
                )
                out['obj_to_rhand_sdf'] = d2_o2r.squeeze(0).sqrt().cpu().numpy().astype(np.float32)

                # 13) 결과 저장 (.npz)
                rel = os.path.relpath(trial_dir, root_dir)
                save_dir = os.path.join(output_dir, rel)
                os.makedirs(save_dir, exist_ok=True)
                np.savez_compressed(os.path.join(save_dir, 'cross_sdf.npz'), **out)
                print(f"[SAVED] {scene}/{trial}")

            except Exception as e:
                print(f"[ERROR] {scene}/{trial} failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Processing pipeline for cross-mesh SDF & drop distance'
    )
    parser.add_argument('--root_dir',   type=str, required=True,
                        help='Root directory with scene/trial folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed .npz files')
    parser.add_argument('--gpu_id',     type=int, default=0,
                        help='GPU ID to use')
    args = parser.parse_args()
    process_all(args.root_dir, args.output_dir, args.gpu_id)


if __name__ == '__main__':
    main()
