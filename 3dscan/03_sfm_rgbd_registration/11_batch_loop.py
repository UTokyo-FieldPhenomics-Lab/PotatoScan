import pathlib

import argparse

import open3d as o3d
import copy
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import matplotlib

# # TkAgg
# print(matplotlib.get_backend())
matplotlib.use('QtAgg')
# # agg
# print(matplotlib.get_backend())

from scipy.signal import find_peaks

from PyQt5.QtWidgets import QApplication

from pin_segment import RgbdPinFetcher, SfMPinFetcher
import linear_algebra as util_la
import cross_align as util_ca
import pin_center as util_pc
import icp_align as util_ia
import qtmessagebox as util_qt
from jsonfile import dict2json

parser = argparse.ArgumentParser(description="Batch script for checking Transform Matrix")
parser.add_argument('-o', '--overwrite', default=False, action='store_true', help='overwrite existing matrix')
parser.add_argument('-i', '--id', default=None, help="specify potato id")
parser.add_argument('-d', '--rgbd_id', default=None, help="specify rgbd image, not the centered one")

if __name__ == "__main__":
    app = QApplication([])

    dataset_root = pathlib.Path(r'/home/crest/w/hwang_Pro/datasets/3DPotatoTwin')
    pin_ref_folder = pathlib.Path(r'/home/crest/Documents/Github/PotatoScan/3dscan/03_sfm_rgbd_registration/pin_ref')

    args = parser.parse_args()

    # get the list of all ids
    id_folder = dataset_root / "2_sfm/2_pcd"
    pid_container = []
    for x in id_folder.glob('**/*'):
        if not x.is_file():  # is folder
            pid_container.append(
                x.name
            )

    matrix_out_folder = dataset_root / "3_pair/tmatrix"
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
        # skip the pid with existing matrix
        if pid in mat_container and not args.overwrite:
            continue

        print(f"\n====== {pid} ======")

        rgbd_data = rgbd_fetcher.get(pid, img_id=args.rgbd_id, visualize=True, show=False)
        sfm_data = sfm_fetcher.get(pid, visualize=True, show=False)

        print(f"=> Find pin vector")
        
        sfm_pin_data = util_pc.find_pin_center(
            sfm_data['pin_pcd'], sfm_data['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False, label="sfm"
        )
        rgbd_pin_data = util_pc.find_pin_center(
            rgbd_data['pin_pcd'], rgbd_data['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False, label="rgbd"
        )

        # show frame 1
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        coord.scale(0.01, center=coord.get_center())
        o3d.visualization.draw_geometries([
            coord,
            #sfm & rgbd raw data
            sfm_data['pcd'], rgbd_data['pcd'],
            # pin segmetation result (with color strength)
            sfm_data['pin_pcd_strengthen'], rgbd_data['pin_pcd'],
            # sfm hsv intermediate visualization
            sfm_data['pcd_offset_colormap'],
            # regressed circle
            sfm_pin_data['circle_mesh'], rgbd_pin_data['circle_mesh'],
            # circle plane normal
            sfm_pin_data['vector_arrow'], rgbd_pin_data['vector_arrow'], 
        ], window_name=f"{pid} - pin segmentation for center and normal")

        ###############################
        # neighbour vector correction #
        ###############################
        search_radius = 0.03

        print(f"=> Find neighbor with {search_radius*100} cm radius")

        # find the pin neighbour of sfm
        sfm_nbr_data = util_ia.find_pin_nbr(sfm_data, sfm_pin_data, search_radius, visualize=True, label="sfm-pin")
        rgbd_nbr_data = util_ia.find_pin_nbr(rgbd_data, rgbd_pin_data, search_radius, visualize=True, label="rgbd-pin")

        # using the neighbour points to correct the vector
        # o3d.visualization.draw_geometries([
        #     # initial alignment
        #     sfm_nbr_data['nbr_pcd'], rgbd_nbr_data['nbr_pcd'],
        #     sfm_nbr_data['vector_arrow'], rgbd_nbr_data['vector_arrow'],
        # ], window_name=f"{pid} - pin neighbour of {search_radius * 100} cm")

        ###################
        # global aligment #
        ###################

        print(f"=> Rough alignment matrix by pin position")

        imatrix = util_la.create_rotational_transform_matrix(
            rgbd_pin_data['circle_center_3d'], rgbd_pin_data['vector'],
            sfm_pin_data['circle_center_3d'], sfm_pin_data['vector'],
        )
        # imatrix =  util_la.create_rotational_transform_matrix(
        #     rgbd_pin_data['circle_center_3d'], rgbd_nbr_data['nbr_vector'],
        #     sfm_pin_data['circle_center_3d'], sfm_nbr_data['nbr_vector'],
        # )

        # rotate rgbd according the imatrix to sfm coordinate
        # with _it suffix
        rgbd_data_it = {}
        rgbd_data_it['pcd'] = copy.deepcopy(rgbd_data['pcd']).transform(imatrix)
        rgbd_data_it['pin_idx'] = rgbd_data['pin_idx']
        rgbd_data_it['pin_pcd'] = rgbd_data_it['pcd'].select_by_index(rgbd_data['pin_idx'])

        rgbd_pin_data_it = util_pc.find_pin_center(
            rgbd_data_it['pin_pcd'], rgbd_data_it['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False
        )
        # find the pin neighbor of rgbd
        rgbd_nbr_data_it = util_ia.find_pin_nbr(rgbd_data_it, rgbd_pin_data_it, radius= search_radius)
        #===========================
        # align by latest n-u-v iter
        #===========================
        cross_buffer = 0.001
        print(f":: iterative optimize rotating around pin normal vector")
        angles, rmses, matrics = util_ca.iterative_nuv_rotate(
            source_pcd    = rgbd_nbr_data_it['nbr_pcd'],
            target_pcd    = sfm_nbr_data['nbr_pcd'],
            rotate_point  = sfm_pin_data['circle_center_3d'],
            normal_vector = sfm_pin_data['vector'],
            buffer        = cross_buffer
        )
        # find the global minimum incase 0 and 350 not identified as peak
        rmses_dup = np.append(rmses, rmses[0:3])

        # find the valley minimum
        peaks, _ = find_peaks(-rmses_dup, distance=1)
        # distance=1 -> 1 x 10 degree -> 10 degrees interval
        # distance=9 -> 9 x 10 degree -> 90 degrees interval
        peaks = peaks % len(rmses)
        peaks = np.unique(peaks)

        # sort peak by values (from minimum to largest)
        if len(peaks) != 1:
            peak_values = rmses[peaks]
            order = np.argsort(peak_values)
            peaks = peaks[order]

        peak_id = 0
        while True:
            best_idx = peaks[peak_id]
            best_angle = angles[best_idx]
            best_rot_matrix = matrics[best_idx]

            # print(f"Best_idx {best_idx}")

            peak_cm = cm['Pastel1']
            fig, ax = plt.subplots(1,1, figsize=(6,4))
            ax.plot(angles, rmses, label="RMSE(m)")
            for i, p in enumerate(peaks):
                if i == peak_id:
                    ax.axvline(x=angles[p], color='r', label="current choice")
                else:
                    ax.axvline(x=angles[p], color=peak_cm(i))
            ax.legend()
            ax.set_xlabel("Rotation degrees ($^{\circ}$)")
            ax.set_ylabel("RMSE")
            ax.set_title("Optimized errors after rotating around pin normal vector")
            plt.tight_layout()
            plt.show()
            plt.close()

            print(f'   Find the minimum differences {round(np.min(rmses), 7)} on angle {best_angle}')
            iimatrix = best_rot_matrix @ imatrix

            o3d.visualization.draw_geometries([
                # initial alignment
                sfm_data['pcd'], 
                # copy.deepcopy(rgbd_data['pcd']).transform(imatrix).paint_uniform_color([0.1,0.1,0.1]),
                # after iter rotation match
                copy.deepcopy(rgbd_data['pcd']).transform(iimatrix)
            ], window_name=f"{pid} - iterative rotating optimization")

            if len(peaks) == 1:
                break
            else:
                result =  util_qt.confirm_message('Confirmation', 'This peak is good?', no_text="Next")
                if result: # clWick yes
                    break
                else:
                    # looping peaks
                    peak_id = ( peak_id + 1 ) % len(peaks)
                    continue

        ########################
        # ICP + color aligment #
        ########################
                
        print(f"=> Class-based ICP detailed optimization")

        sfm_pcd_bin = util_ia.paint_pcd_binary(sfm_data['pcd'], sfm_data['pin_idx'])
        rgbd_pcd_bin = util_ia.paint_pcd_binary(rgbd_data['pcd'], rgbd_data['pin_idx']) 

        # o3d.visualization.draw_geometries([sfm_pcd_bin, rgbd_pcd_bin])

        # iteractive add iter
        icp_iter_num = 0
        icp_threshold = 0.001
        geometry_weight = 0.1

        first_iter = True
        while True:
            tmatrix, o3drmse = util_ia.color_based_icp(
                rgbd_pcd_bin, sfm_pcd_bin, iimatrix, 
                threshold=icp_threshold, max_iter=icp_iter_num, geometry_weight=geometry_weight,
                return_rmse=True)
            
            # visualize frame 2
            rgbd_temp, sfm_temp = util_ia.draw_registration_result(rgbd_data['pcd'], sfm_data['pcd'], tmatrix, paint_color=False, offset=[0.1,0,0])
            # rgbd_bin_temp, sfm_bin_temp = draw_registration_result(rgbd_pcd_bin, sfm_pcd_bin, tmatrix, paint_color=False, offset=[0.1,0.1,0])

            if not first_iter:
                o3d.visualization.draw_geometries([
                    # after alignment
                    rgbd_temp, sfm_temp, 
                    # rgbd_bin_temp, sfm_bin_temp, 
                ], window_name=f"{pid} - Class-based ICP refinement with {icp_iter_num} iters, {icp_threshold}m per step | shape weight {geometry_weight}, class weight {1-geometry_weight}")
            else:
                first_iter = False

            result = util_qt.iter_num_message("Need more ICP iteration?", f"Current is {icp_iter_num} iter(s), decrase if pin shifted")

            if result == 0 or result is None:
                break
            else:
                icp_iter_num = max(icp_iter_num + result, 0)

        print(f":: The computered transform matrix: \n{tmatrix}")

        result =  util_qt.confirm_message('Confirmation', 'Save the matrix file?')
        if result:
            source_pcd = copy.deepcopy(rgbd_data['pcd']).transform(tmatrix)
            rmse = util_la.compute_distance_rmse(source_pcd, sfm_data['pcd'])

            print(f"   Open3D calculated RMSE: {o3drmse}, Self calculated RMSE: {rmse}")

            matrix_file_path = matrix_out_folder / f"{pid}.json"

            output = {
                "rgbd_pcd_file": rgbd_data['pcd_rela_path'],
                "sfm_mesh_file": sfm_data['pcd_rela_path'],
                "T": tmatrix,
                "rms_minimum_distance": rmse,
                "open3d_inlier_rmse": o3drmse,
                "meta": {
                    "pin_segment": {
                        "sfm": {
                            "hsv_weight": sfm_data['hsv_weight'],
                            "hsv_index_denoise_threshold": sfm_data['stop_thresh'],
                            "hsv_index_denoised_volume": sfm_data['stop_hull_volume'], 
                            "center": sfm_pin_data['circle_center_3d'],
                            "radius(m)": sfm_pin_data['circle_radius'],
                            "normal_vector": sfm_pin_data['vector']
                        },
                        "rgbd": {
                            "center": rgbd_pin_data['circle_center_3d'],
                            "radius(m)": rgbd_pin_data['circle_radius'],
                            "normal_vector": rgbd_pin_data['vector']
                        },
                    },
                    "pin_neighbor": {
                        "search_radius(m)": search_radius,
                        "corss_buffer(m)": cross_buffer
                    },
                    "class_based_icp": {
                        "iter_num":icp_iter_num,
                        "iter_distance(m)": icp_threshold,
                        "geometry_weight": geometry_weight
                    }
                },
            }

            dict2json(output, matrix_file_path, indent=4, encoding='utf-8')