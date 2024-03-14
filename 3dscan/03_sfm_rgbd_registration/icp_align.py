import numpy as np
from scipy.spatial.transform import Rotation

import copy
import open3d as o3d

##########################
# pin based global align #
##########################
        
def create_rotational_transform_matrix(P1, N1, P2, N2, rotation_point=None):
    # 将输入向量标准化
    N1 = N1 / np.linalg.norm(N1)
    N2 = N2 / np.linalg.norm(N2)
    
    # 计算旋转轴和旋转角度
    cross_product = np.cross(N1, N2)
    dot_product = np.dot(N1, N2)

    if rotation_point is None:
        rotation_point = P1
    
    # 当两个向量平行时，叉积为零
    if np.allclose(cross_product, 0):
        # 如果点积为-1，向量相反，需要绕垂直于这些向量的轴旋转180度
        if np.isclose(dot_product, -1):
            # 选择一个与N1垂直的轴作为旋转轴
            perp_vector = np.array([1, 0, 0]) if abs(N1[0]) < abs(N1[1]) else np.array([0, 1, 0])
            rotation_axis = np.cross(N1, perp_vector)
            rotation_angle = np.pi
        else:
            # 向量重合，无需旋转
            rotation_axis = [1, 0, 0]  # 默认轴
            rotation_angle = 0
    else:
        rotation_axis = cross_product
        rotation_angle = np.arccos(dot_product)
    
    # 使用scipy来构建旋转矩阵
    rotation_matrix = Rotation.from_rotvec(rotation_axis * rotation_angle).as_matrix()
    
    # 创建平移矩阵以将旋转点移至原点
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -rotation_point
    
    # 创建平移矩阵以将旋转点移回其原始位置
    translation_back = np.eye(4)
    translation_back[:3, 3] = rotation_point
    
    # 创建旋转矩阵的4x4版本
    rot_matrix_4x4 = np.eye(4)
    rot_matrix_4x4[:3, :3] = rotation_matrix
    
    # 组合变换：平移到原点，旋转，然后平移回去
    combined_transform = translation_back @ rot_matrix_4x4 @ translation_to_origin
    
    # 计算从P1到P2的平移向量，并将其添加到变换矩阵中
    translation_vector = P2 - P1
    combined_transform[:3, 3] += translation_vector
    
    return combined_transform


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
        voxel_size=0.001, geometry_weight=0.3, threshold=0.002
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
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    
    return result_icp.transformation @ initial_matrix

