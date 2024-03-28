import pathlib

import argparse

import open3d as o3d
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

# # TkAgg
# print(matplotlib.get_backend())
matplotlib.use('QtAgg')
# # agg
# print(matplotlib.get_backend())

from scipy.signal import find_peaks

from PyQt5.QtWidgets import QApplication, QMessageBox

from pin_segment import RgbdPinFetcher, SfMPinFetcher
import linear_algebra as util_la
import cross_align as util_ca
import pin_center as util_pc
import icp_align as util_ia

parser = argparse.ArgumentParser(description="Batch script for checking Transform Matrix")
parser.add_argument('-o', '--overwrite', default=False, action='store_true', help='overwrite existing matrix')
parser.add_argument('-i', '--id', default=None, help="specify potato id")

def confirm_message(title, text):
    # 创建一个消息框
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle(title)
    msg.setText(text)
    # 添加按钮
    ok_button = msg.addButton('Yes', QMessageBox.AcceptRole)
    next_button = msg.addButton('Next', QMessageBox.RejectRole)
    # 显示消息框
    msg.exec_()
    # 判断用户点击了哪个按钮并返回
    if msg.clickedButton() == ok_button:
        return True
    elif msg.clickedButton() == next_button:
        return False
    else:
        return None


def iter_num_message(title, text):
    # 创建一个消息框
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle(title)
    msg.setText(text)
    # 添加按钮
    l_button = msg.addButton('-5', QMessageBox.AcceptRole)
    m_button = msg.addButton('OK', QMessageBox.RejectRole)
    r_button = msg.addButton('+5', QMessageBox.RejectRole)
    # 显示消息框
    msg.exec_()
    # 判断用户点击了哪个按钮并返回
    if msg.clickedButton() == l_button:
        return -1
    elif msg.clickedButton() == m_button:
        return 0
    elif msg.clickedButton() == r_button:
        return 1
    else:
        return None

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
        
        sfm_pin_data = util_pc.find_pin_center(
            sfm_data['pin_pcd'], sfm_data['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False
        )
        rgbd_pin_data = util_pc.find_pin_center(
            rgbd_data['pin_pcd'], rgbd_data['pcd'], 
            circle_color=[0,0,0], visualize=True, show=False
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
        ], window_name=f"{pid} - preprocessing")

        ###############################
        # neighbour vector correction #
        ###############################
        search_radius = 0.03

        # find the pin neighbour of sfm
        sfm_nbr_data = util_ia.find_pin_nbr(sfm_data, sfm_pin_data, search_radius, visualize=True)
        rgbd_nbr_data = util_ia.find_pin_nbr(rgbd_data, rgbd_pin_data, search_radius, visualize=True)

        # using the neighbour points to correct the vector
        # o3d.visualization.draw_geometries([
        #     # initial alignment
        #     sfm_nbr_data['nbr_pcd'], rgbd_nbr_data['nbr_pcd'],
        #     sfm_nbr_data['vector_arrow'], rgbd_nbr_data['vector_arrow'],
        # ], window_name=f"{pid} - pin neighbour of {search_radius * 100} cm")

        ###################
        # global aligment #
        ###################

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
        angles, rmses, matrics = util_ca.iterative_nuv_rotate(
            source_pcd    = rgbd_nbr_data_it['nbr_pcd'],
            target_pcd    = sfm_nbr_data['nbr_pcd'],
            rotate_point  = sfm_pin_data['circle_center_3d'],
            normal_vector = sfm_pin_data['vector']
        )

        # find the valley minimum
        peaks, _ = find_peaks(-rmses, distance=9)  # 10 degree one sample, 9 -> 90 degrees

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

            print(f"Best_idx {best_idx}")

            peak_cm = cm.get_cmap('Pastel1')
            fig = plt.figure()
            plt.plot(angles, rmses)
            for i, p in enumerate(peaks):
                if i == peak_id:
                    plt.axvline(x=angles[p], color='r', label="current choice")
                else:
                    plt.axvline(x=angles[p], color=peak_cm(i))
            plt.show()
            plt.close()

            print(f':: Iterative vector axis rotation\n   Find the minimum differences {round(np.min(rmses), 7)} on angle {best_angle}')
            iimatrix = best_rot_matrix @ imatrix

            o3d.visualization.draw_geometries([
                # initial alignment
                sfm_data['pcd'], copy.deepcopy(rgbd_data['pcd']).transform(imatrix),
                # after iter rotation match
                copy.deepcopy(rgbd_data['pcd']).transform(iimatrix).paint_uniform_color([1,0,0])
            ], window_name=f"{pid} - initial registration")

            if len(peaks) == 1:
                break
            else:
                result =  confirm_message('Confirmation', 'This peak is good?')
                if result: # clWick yes
                    break
                else:
                    # looping peaks
                    peak_id = ( peak_id + 1 ) % len(peaks)
                    print(peak_id)
                    continue

        ########################
        # ICP + color aligment #
        ########################

        sfm_pcd_bin = util_ia.paint_pcd_binary(sfm_data['pcd'], sfm_data['pin_idx'])
        rgbd_pcd_bin = util_ia.paint_pcd_binary(rgbd_data['pcd'], rgbd_data['pin_idx']) 

        # o3d.visualization.draw_geometries([sfm_pcd_bin, rgbd_pcd_bin])

        # iteractive add iter

        iter_num = 0
        while True:

            tmatrix = util_ia.color_based_icp(rgbd_pcd_bin, sfm_pcd_bin, iimatrix, threshold=0.002, max_iter=iter_num)
            
            # visualize frame 2
            rgbd_temp, sfm_temp = util_ia.draw_registration_result(rgbd_data['pcd'], sfm_data['pcd'], tmatrix, paint_color=False, offset=[0.1,0,0])
            # rgbd_bin_temp, sfm_bin_temp = draw_registration_result(rgbd_pcd_bin, sfm_pcd_bin, tmatrix, paint_color=False, offset=[0.1,0.1,0])

            o3d.visualization.draw_geometries([
                # after alignment
                rgbd_temp, sfm_temp, 
                # rgbd_bin_temp, sfm_bin_temp, 
            ], window_name=f"{pid} - registration")

            result = iter_num_message("Need more ICP iteration?", f"Current is [{iter_num}] iter(s), decrase if pin shifted")

            if result == 0 or result is None:
                break
            else:
                iter_num = max(iter_num + result * 5, 0)

        print(f":: The computered transform matrix: \n{tmatrix}")

        break