import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from circle_fit import hyperLSQ

import linear_algebra as util_la
    
def correct_vector_direction(points, vector, vector_root, colors=None):
    start_point = vector_root
    end_point = vector_root + vector/100
    # 把start_point & end_point 插入到头部
    root_on_top = np.vstack([start_point.T, end_point.T, points])

    root_on_top_projected = util_la.project_points_on_vector(
        points=root_on_top, 
        vector=vector, 
        return_1d=True
    )
    
    potato_min = root_on_top_projected[2:].min()
    potato_max = root_on_top_projected[2:].max()

    start = root_on_top_projected[0]
    end = root_on_top_projected[1]

    # find out st close to which
    dist2min = abs(start - potato_min)
    dist2max = abs(start - potato_max)

    print(f"st = {start} | ed = {end} | min = {potato_min} | max = {potato_max}")
    # =====debug=====
    # plot in 3d
    # root_on_top_projected_3d= project_points_on_vector(
    #     points=root_on_top, 
    #     vector=vector, 
    #     return_1d=False)
    # # colors = np.insert([[0,1,0]], 0, colors, axis=0)  # green = end
    # # colors = np.insert([[1,0,0]], 0, colors, axis=0)  # red = top
    # pts = o3d.geometry.PointCloud()
    # pts.points = o3d.utility.Vector3dVector(root_on_top_projected_3d[2:, :])
    # pts.colors = o3d.utility.Vector3dVector(colors)

    # direct_pts = o3d.geometry.PointCloud()
    # direct_pts.points = o3d.utility.Vector3dVector(root_on_top_projected_3d[:2, :] + np.array([0,0,0.005]))
    # direct_pts.colors = o3d.utility.Vector3dVector([[0,1,0], [1,0,0]])
    # o3d.visualization.draw_geometries([pts, direct_pts])

    # plot in 2d
    # fig,ax = plt.subplots(1,1, figsize=(4,2))
    # ax.scatter(x=root_on_top_projected[2:], y=np.zeros_like(root_on_top_projected[2:]), c=colors)
    # ax.scatter(x=root_on_top_projected[:2], y=[0.1, 0.1], c=[[1,0,0],[0,1,0]])
    # plt.axis('equal')
    # plt.show()
    # =====end debug=====

    if dist2min < dist2max:  # st close to min
        if start < end:
            #         st ==========> ed      
            #         |
            # -----|--o--------|-------->
            #      min         max  
            # 
            # in this case, normal need revert
            return -vector
        else:
            #  ed <== st    
            #         |
            # -----|--o--------|-------->
            #      min         max  
            # 
            # in this case, nomal no need revert
            return vector
    else: # st close to max
        if start < end:
            #             st ==========> ed      
            #              |
            # -----|-------o---|-------->
            #      min         max  
            # 
            # in this case, normal no need revert
            return vector
        else:
            #  ed <======= st    
            #              |
            # -----|-------o---|-------->
            #      min         max  
            #
            # in this case, normal need revert
            return -vector


def fit_circle_to_convex_hull(points_2d, show=False):
    """平面圆拟合"""

    # 获取凸包的顶点
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]

    # xc, yc, r, sigma = taubinSVD(hull_points)
    xc, yc, r, sigma = hyperLSQ(hull_points)

    if show:
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        ax.scatter(*points_2d.T, color='k', s=0.1, alpha=0.3)
        ax.scatter(*hull_points.T, color='r')
        ax.scatter([xc], [yc], color='green')
        plt.axis('equal')
        plt.show()

    return (xc, yc), r, sigma

def create_circle_mesh(center_2d, radius, plane_point, uv, num_segments=100, color=[1,0,0]):
    """visualize the regressed circlus of pin in open3d

    Parameters
    ----------
    center_2d : np.array[2x1]
        the x and y of circle center on the plane
    radius : float
        the radius of regressed circle
    plane_point : np.array[3x1]
        a point in the circle plane
    uv : np.array[2x3]
        two base vectors of the circular plane, 
        output (u, v) from `project_to_plane()` function
    num_segments : int, optional
        the edge number of circle, by default 100

    Returns
    -------
    _type_
        _description_
    """
    # 创建圆形的网格
    points = []
    indices = []
    
    # 中心点
    points.append(center_2d)
    # 计算圆周上的点
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        offset = np.array([np.cos(angle)*radius, np.sin(angle)*radius])
        points.append(center_2d + offset)
        if i != 0:
            # 每个三角形的索引
            indices.append([0, i, i+1])
    # 最后一个三角形的索引
    indices.append([0, num_segments, 1])

    # 转换为3d坐标
    points = util_la.convert_2d_to_3d(np.asarray(points), plane_point, uv[0], uv[1])
    
    # 创建三角网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(indices)
    mesh.compute_vertex_normals()

    mesh_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    num_lines = len(np.asarray(mesh_lineset.lines))
    line_colors = [color] * num_lines
    mesh_lineset.colors = o3d.utility.Vector3dVector(line_colors)
    
    return mesh_lineset

def create_vector_arrow(start_point, vector_normal, zoom=0.01, color=None):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=0.1 * zoom,
        cone_height=0.3 * zoom,
        cylinder_radius=0.05 * zoom,
        cylinder_height=0.7 * zoom
    )
    # arrow.scale(zoom, center=arrow.get_center())

    if color is not None:
        arrow = arrow.paint_uniform_color(color)


    # 确定箭头方向
    # 计算箭头的变换矩阵，使其从z轴对齐到向量方向
    # 这涉及到计算两个向量间的旋转矩阵
    z_axis = np.array([0, 0, 1])
    o= np.array([0, 0, 0])

    matrix = util_la.create_rotational_transform_matrix(o, z_axis, start_point, vector_normal)

    # vector = vector_normal * zoom
    # cross_prod = np.cross(z_axis, vector)
    # dot_prod = np.dot(z_axis, vector)
    # s = np.linalg.norm(cross_prod)
    # c = dot_prod
    # skew_symmetric = np.array([[0, -cross_prod[2], cross_prod[1]],
    #                         [cross_prod[2], 0, -cross_prod[0]],
    #                         [-cross_prod[1], cross_prod[0], 0]])
    # rotation_matrix = np.eye(3) + skew_symmetric + skew_symmetric.dot(skew_symmetric) * ((1 - c) / (s ** 2))

    # 构建4x4变换矩阵
    # transformation_matrix = np.eye(4)
    # transformation_matrix[:3, :3] = rotation_matrix
    # transformation_matrix[:3, 3] = start_point

    # 应用变换
    arrow = arrow.transform(matrix)

    return arrow

def find_minimum_vector_of_bbox(pcd):
    hull, hull_idx = pcd.compute_convex_hull()
    # get bounding box size
    hull_obb = hull.get_oriented_bounding_box()

    # 获取边界框的三个轴的长度
    extents = hull_obb.extent

    # 找到最短边和对应的索引
    min_extent_idx = np.argmin(extents)
    min_extent_length = extents[min_extent_idx]

    # 获取对应于最短边的轴
    _min_extent_vector = hull_obb.R[:, min_extent_idx]
    min_extent_vector_norm = _min_extent_vector / np.linalg.norm(_min_extent_vector)

    # 获取边界框的中心，作为平面上的一个点
    plane_point = hull_obb.center

    return min_extent_vector_norm, plane_point


def find_pin_center(pin_pcd, pcd, circle_color=[1,0,0], visualize=False, show=False):
    vector_normalized, plane_point = find_minimum_vector_of_bbox(pin_pcd)

    # 矫正轴的方向
    pin_vector = correct_vector_direction(
        np.asarray(pcd.points), vector_normalized, plane_point, np.asarray(pcd.colors)
    )
    print(f"pin vector from {vector_normalized} to {pin_vector}")

    # 投影到最短边对应的平面上
    points_3d = np.asarray(pin_pcd.points)
    points_proj_3d, points_proj_2d, uv = util_la.project_to_plane_vectorized(points_3d, plane_point, pin_vector)

    # 计算2D点集的凸包，并拟合圆
    circle_center_2d, circle_radius, sigma = fit_circle_to_convex_hull(points_proj_2d, show)

    # 将2D圆心转换回3D空间坐标
    circle_center_3d = util_la.convert_2d_to_3d(np.asarray([circle_center_2d]), plane_point, uv[0], uv[1])
    circle_center_3d = circle_center_3d[0]

    results = {
        "circle_center_3d": circle_center_3d,
        "circle_radius": circle_radius,
        "vector": pin_vector,
    }

    if visualize or show:

        # 将投影点列表转换为Open3D点云
        projected_cloud = o3d.geometry.PointCloud()
        projected_cloud.points = o3d.utility.Vector3dVector(points_proj_3d)
        projected_cloud = projected_cloud.paint_uniform_color([0,0,1])

        # 创建一个圆形网格
        circle_mesh = create_circle_mesh(circle_center_2d, circle_radius, plane_point, uv, color=circle_color)

        vector_arrow = create_vector_arrow(
            circle_center_3d, pin_vector, 
            zoom=0.01, color=circle_color)  # zoom to 1 cm
        # 创建vector的箭头
        # end_point = circle_center_3d + pin_vector / 100 # to 1 cm
        # lineset = o3d.geometry.LineSet()
        # # 设置点（两个点：起点和终点）
        # lineset.points = o3d.utility.Vector3dVector([circle_center_3d, end_point])
        # # 设置线（一条线从点0到点1）
        # lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
        # lineset.colors = o3d.utility.Vector3dVector([circle_color])


        if show:
            o3d.visualization.draw_geometries([circle_mesh, pin_pcd, projected_cloud, vector_arrow])

        results['projected_cloud'] = projected_cloud
        results['circle_mesh'] = circle_mesh
        results['vector_arrow'] = vector_arrow
        # results['vector_lineset'] = lineset

    return results