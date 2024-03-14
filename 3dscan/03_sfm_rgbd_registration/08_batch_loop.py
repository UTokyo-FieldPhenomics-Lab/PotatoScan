import pathlib
import numpy as np
import skimage
import plotly.graph_objects as go

from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib.path import Path as pltpath
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

import argparse


import open3d as o3d
import copy

from pin_segment import RgbdPinFetcher, SfMPinFetcher
from pin_center import find_pin_center
from icp_align import create_rotational_transform_matrix, paint_pcd_binary, color_based_icp, draw_registration_result

parser = argparse.ArgumentParser(description="Batch script for checking Transform Matrix")
parser.add_argument('-o', '--overwrite', default=False, action='store_true', help='overwrite existing matrix')
parser.add_argument('-i', '--id', default=None, help="specify potato id")


if __name__ == "__main__":
    dataset_root = pathlib.Path(r'/home/crest/w/hwang_Pro/datasets/3DPotatoTwin')
    pin_ref_folder = pathlib.Path(r'/home/crest/Documents/Github/PotatoScan/3dscan/03_sfm_rgbd_registration/pin_ref')

    args = parser.parse_args()

    # get the list of all ids
    id_folder = dataset_root / "2_sfm/2_pcd"
    pid_container = []
    for x in id_folder.glob('**/*'):
        if x.is_file():
            pid_container.append(
                x.name.replace('_30000.ply', '')
            )

    matrix_out_folder = dataset_root / "03_pair/01_tmatrix"
    mat_container = []
    for x in matrix_out_folder.glob('**/*'):
        if x.is_file():
            mat_container.append(x.stem)

    rgbd_fetcher = RgbdPinFetcher(dataset_root)
    sfm_fetcher = SfMPinFetcher(dataset_root, pin_ref_folder)

    # start looping
    if args.id is not None:
        if args.id not in pid_container:
            raise FileExistsError(f"Can not find pid=[{args.id}]")
        else:
            pid_container = [args.id]

    for pid in pid_container:
        print(f"\n====== {pid} ======")
        # skip the pid with existing matrix
        if pid in mat_container and not args.overwrite:
            continue

        rgbd_data = rgbd_fetcher.get(pid, visualize=True, show=False)
        sfm_data = sfm_fetcher.get(pid, visualize=True, show=False)
        
        sfm_pin_data = find_pin_center(
            sfm_data['pin_pcd'], sfm_data['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False
        )
        rgbd_pin_data = find_pin_center(
            rgbd_data['pin_pcd'], rgbd_data['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False
        )

        # show frame 1
        o3d.visualization.draw_geometries([
            #sfm & rgbd raw data
            sfm_data['pcd'], rgbd_data['pcd'],
            # pin segmetation result (with color strength)
            sfm_data['pin_pcd_strengthen'], rgbd_data['pin_pcd'],
            # sfm hsv intermediate visualization
            sfm_data['pcd_offset_colormap'],
            # regressed circle
            sfm_pin_data['circle_mesh'], rgbd_pin_data['circle_mesh'],
            # circle plane normal
            sfm_pin_data['vector_lineset'], rgbd_pin_data['vector_lineset'], 
        ], window_name=f"{pid} - preprocessing")

        ################
        # ICP aligment #
        ################
        imatrix =  create_rotational_transform_matrix(
            rgbd_pin_data['circle_center_3d'], rgbd_pin_data['vector'],
            sfm_pin_data['circle_center_3d'], sfm_pin_data['vector'],
        )

        sfm_pcd_bin = paint_pcd_binary(sfm_data['pcd'], sfm_data['pin_idx'])
        rgbd_pcd_bin = paint_pcd_binary(rgbd_data['pcd'], rgbd_data['pin_idx']) 

        tmatrix = color_based_icp(rgbd_pcd_bin, sfm_pcd_bin, imatrix, threshold=1)
        
        # visualize frame 2
        rgbd_temp, sfm_temp = draw_registration_result(rgbd_data['pcd'], sfm_data['pcd'], tmatrix, paint_color=False, offset=[0.1,0,0])
        rgbd_bin_temp, sfm_bin_temp = draw_registration_result(rgbd_pcd_bin, sfm_pcd_bin, tmatrix, paint_color=False, offset=[0.1,0.1,0])

        o3d.visualization.draw_geometries([
            # initial alignment
            sfm_data['pcd'], copy.deepcopy(rgbd_data['pcd']).transform(imatrix),
            # after alignment
            rgbd_temp, sfm_temp, 
            # rgbd_bin_temp, sfm_bin_temp, 
        ], window_name=f"{pid} - registration")

        print(tmatrix)

        break