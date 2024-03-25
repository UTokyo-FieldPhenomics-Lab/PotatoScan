import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

import copy
import open3d as o3d

import linear_algebra as util_la
import pin_center as util_pc

##############################
# pin neighbour global align #
##############################

def find_pin_nbr(pcd_data_dict, pin_data_dict, radius, visualize=False):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_data_dict['pcd'])

    [k1, nbr_idx, _] = pcd_tree.search_radius_vector_3d(
        pin_data_dict['circle_center_3d'], radius
    )

    [k2, nbr_pin_idx, _] = pcd_tree.search_radius_vector_3d(
        pin_data_dict['circle_center_3d'], pin_data_dict['circle_radius']
    )

    nbr_no_pin_idx = np.setdiff1d(nbr_idx, nbr_pin_idx)

    nbr_pcd = pcd_data_dict['pcd'].select_by_index(nbr_no_pin_idx)

    # calculate the vector according to the new region.
    vector_normalized, bbox_center = util_pc.find_minimum_vector_of_bbox(nbr_pcd)
    # 矫正轴的方向
    vector_corrected = util_pc.correct_vector_direction(
        np.asarray(pcd_data_dict['pcd'].points), 
        vector_normalized, 
        pin_data_dict['circle_center_3d'], 
        np.asarray(pcd_data_dict['pcd'].colors)
    )
    
    results = {
        "nbr_pcd": nbr_pcd,
        "nbr_idx": nbr_no_pin_idx,
        "nbr_radius": radius,
        "nbr_vector": vector_corrected
    }

    if visualize:
        # 创建vector的箭头
        start_point = pin_data_dict['circle_center_3d']
        vector_arrow = util_pc.create_vector_arrow(start_point, vector_corrected, zoom=0.01, color=[1,1,0])

        results['vector_arrow'] = vector_arrow

    return results

def iter_rotation_angle(source_pcd, target_pcd, rotate_axis, rotate_point):
    angles = []
    distances = []
    rot_matrices = []
    for angle in range(1, 36):
        rot_matrix = util_la.rotation_matrix_around_vector(rotate_axis, rotate_point, angle*10)

        # rgbd -> to rotate
        source_rot_vector_pcd = copy.deepcopy(source_pcd).transform(rot_matrix)
        source_rot_vector_pcd.paint_uniform_color([1,0,0])

        dist = np.mean(
            source_rot_vector_pcd.compute_point_cloud_distance(target_pcd)
        )

        angles.append(angle*10)
        distances.append(dist)
        rot_matrices.append(rot_matrix)


    angles = np.asarray(angles)
    distances = np.asarray(distances)

    # find the minimum values
    best_angle = angles[np.argmin(distances)]
    best_rot_matrix = rot_matrices[np.argmin(distances)]

    plt.plot(angles, distances)
    plt.show()

    print(f':: Iterative vector axis rotation\n   Find the minimum differences {round(np.min(distances), 7)} on angle {best_angle}')

    return best_rot_matrix


############
# ICP part #
############
def draw_registration_result(source, target, transformation, paint_color=True, offset=[0,0,0], show=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if paint_color:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    # add offsets
    if offset != [0,0,0]:
        xyz = np.asarray(source_temp.points) + np.array(offset)
        source_temp.points = o3d.utility.Vector3dVector(xyz)

        xyz = np.asarray(target_temp.points) + np.array(offset)
        target_temp.points = o3d.utility.Vector3dVector(xyz)

    if show:
        o3d.visualization.draw_geometries([source_temp, target_temp])
    else:
        return source_temp, target_temp

def paint_pcd_binary(pcd, pin_idx):
    potato_temp = pcd.select_by_index(pin_idx, invert=True)
    pin_temp = pcd.select_by_index(pin_idx, invert=False)

    potato_temp.paint_uniform_color(np.array([0,0,1]))
    pin_temp.paint_uniform_color(np.array([1,0,0]))

    return potato_temp + pin_temp

def color_based_icp(
        source_binary_pcd, target_binary_pcd, initial_matrix, 
        voxel_size=0.001, geometry_weight=0.3, threshold=0.002, max_iter=2000
    ):
    """_summary_

    Parameters
    ----------
    source_binary_pcd : o3d.PointCloud
        rgbd point cloud
    target_binary_pcd : o3d.PointCloud
        sfm point cloud
    voxel_size : float, optional
        the size for voxel downsampling, by default 0.001
    color_weight : float, optional
        the weight of color vs geometry, from 0 to 1, by default 0.7
    threshold: float, optional
        the distance? threshold to stop ICP align iterations, by default 0.002 (2mm)


    Note
    ----
    By passing initial_matrix to `registration_colored_icp` gives a different (worse) results 
    than transforming first and then icp, and finally multiple two matrix together
    """
    source_binary_pcd_t = copy.deepcopy(source_binary_pcd).transform(initial_matrix)

    source_pcd_down = source_binary_pcd_t.voxel_down_sample(voxel_size)
    target_pcd_down = target_binary_pcd.voxel_down_sample(voxel_size)

    source_pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_pcd_down, target_pcd_down, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=geometry_weight), # weight of color, smaller means color more important
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    
    return result_icp.transformation @ initial_matrix

