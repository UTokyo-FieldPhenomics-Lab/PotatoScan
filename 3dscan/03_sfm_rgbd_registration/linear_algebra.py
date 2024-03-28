import numpy as np

#########################
# mv from pin_center.py #
#########################

def calculate_normal_uv(vector):
    # 归一化平面法线
    normal_vector = vector / np.linalg.norm(vector)
    
    # 计算平面的两个基向量
    if not np.allclose(normal_vector, [1, 0, 0]):
        u = np.cross(normal_vector, [1, 0, 0])
    else:
        u = np.cross(normal_vector, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vector, u)

    return u, v

def project_to_plane_vectorized(points, plane_point, plane_normal):
    u, v = calculate_normal_uv(plane_normal)
    
    # 计算从平面点到所有点的向量
    vecs = points - plane_point
    
    # 计算这些向量在平面法向量上的投影长度
    dists = np.dot(vecs, plane_normal).reshape(-1, 1)
    
    # 计算三维空间中的投影点
    proj_points_3d = points - dists * plane_normal
    
    # 计算二维平面上的坐标
    x_coords = np.dot(proj_points_3d - plane_point, u)
    y_coords = np.dot(proj_points_3d - plane_point, v)
    
    points_proj_2d = np.column_stack((x_coords, y_coords))
    
    return proj_points_3d, points_proj_2d, (u,v)

def convert_2d_to_3d(points_2d, plane_point, u, v):
    # 将二维坐标转换为三维坐标
    points_3d = plane_point + points_2d[:, 0, np.newaxis] * u + points_2d[:, 1, np.newaxis] * v
    return points_3d

def project_points_on_vector(points, vector, return_1d=False):
    """
    把点投影到给定向量上
    输入：points: nx3的三维点
         vector: 1x3的向量
    输出：投影后的点(在空间中的位置)
    """
    v_norm = np.sqrt(sum(vector**2))

    v3d = (np.dot(points, vector.reshape(3,1))/v_norm**2)*vector

    if return_1d:
        return np.sqrt(np.sum(v3d**2, axis=1))
    else:
        return v3d
    

########################
# mv from icp_align.py #
########################
    
# pin based global align
def create_rotational_transform_matrix(p1, n1, p2, n2, rotation_point=None):
    # 将输入向量标准化
    N1 = n1 / np.linalg.norm(n1)
    N2 = n2 / np.linalg.norm(n2)
    
    # 计算旋转轴和旋转角度
    v = np.cross(N1, N2)
    c = np.dot(N1, N2)
    s = np.linalg.norm(v)

    kmat = np.array([
        [0, -v[2], v[1]], 
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]
    ])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    if rotation_point is None:
        rotation_point = p1
    
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
    translation_vector = p2 - p1
    combined_transform[:3, 3] += translation_vector
    
    return combined_transform

def rotation_matrix_around_vector(axis, rotation_point, theta_deg):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    theta_rad = np.radians(theta_deg)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta_rad / 2.0)
    b, c, d = -axis * np.sin(theta_rad / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    rotation_matrix = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

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

    return combined_transform

#########################
# 09_icp_by_cross.ipynb #
#########################

def point_to_plane_distance(points, plane_point, plane_normal):
    # 计算点云到面的距离
    distances = np.abs(np.dot(points - plane_point, plane_normal)) / np.linalg.norm(plane_normal)
    return distances

def compute_distance_rmse(source_pcd, target_pcd):
    errors = np.asarray(
        source_pcd.compute_point_cloud_distance(target_pcd)
    )
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse