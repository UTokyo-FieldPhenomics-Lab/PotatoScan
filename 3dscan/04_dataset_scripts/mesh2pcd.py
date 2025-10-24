import os
import trimesh

from pathlib import Path

import numpy as np
import open3d as o3d

def mesh2pcd(mesh_path, points_num):
    # read and sample mesh by trimesh
    mesh_tri = trimesh.load(mesh_path, force='mesh')
    samples, face_idx, colors = trimesh.sample.sample_surface(mesh_tri, points_num, sample_color=True)

    o3d_rgb = colors[:,0:3] / 255

    # convert trimesh to open3d objects
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(samples)
    final_pcd.colors = o3d.utility.Vector3dVector(o3d_rgb)

    return final_pcd


if __name__ == "__main__":

    POINTS_NUM = [10000, 20000, 30000]

    ROOT = Path("/home/crest/w/hwang_Pro/datasets/3DPotatoTwin.source/2_SfM/")

    mesh_folder = ROOT / "1_mesh"
    pcd_folder = ROOT / "2_pcd"

    mesh_id_list = [i.name for i in mesh_folder.glob("**/*") if i.is_dir()]

    already_exist = [i.name for i in pcd_folder.glob("**/*") if i.is_dir()]

    for potato_idx in mesh_id_list:

        if potato_idx in already_exist:
            print(f"Exists [{potato_idx}], skip")
            continue
        else:
            print(f'Convert [{potato_idx}]')


        for pn in POINTS_NUM:

            mesh_path = mesh_folder / potato_idx / f"{potato_idx}.obj"

            sampled = mesh2pcd(mesh_path, pn)

            put_folder = pcd_folder / potato_idx
            if not put_folder.exists():
                put_folder.mkdir()

            pcd_path = put_folder / f"{potato_idx}_{pn}.ply"

            o3d.io.write_point_cloud(
                pcd_path,
                sampled
            )

            print(f" ---> Save to {os.path.abspath(pcd_path)}")