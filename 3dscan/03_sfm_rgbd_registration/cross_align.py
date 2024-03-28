import copy
import numpy as np
import matplotlib.pyplot as plt

import linear_algebra as util_la

def get_nbr_cross(pcd, plane_point, plane_normal, buffer, merge=True):
    points = np.asarray(pcd.points)
    u,v = util_la.calculate_normal_uv(plane_normal)

    dist_u = util_la.point_to_plane_distance(points, plane_point, u)
    u_id = np.where(dist_u < buffer)[0]
    dist_v = util_la.point_to_plane_distance(points, plane_point, v)
    v_id = np.where(dist_v < buffer)[0]

    if merge:
        uv_id = np.unique(np.concatenate((u_id, v_id),0))
        uv_pcd = pcd.select_by_index(uv_id)
        return uv_pcd, uv_id
    else:
        u_pcd = pcd.select_by_index(u_id)
        v_pcd = pcd.select_by_index(v_id)
        return u_pcd, v_pcd, u_id, v_id
    
def iterative_uv_rotate_one(source_pcd, target_pcd, rotate_point, rotate_vector, ax=None):
    """
    ax = None => no ploting
    """

    # target -> sfm
    _, target_cross_flatten, _ = util_la.project_to_plane_vectorized(
        np.asarray(target_pcd.points), rotate_point, rotate_vector
    )
    
    if ax is not None:
        ax.scatter(*target_cross_flatten.T, color='k',s=0.1, marker='.')

    rmses = []
    angles = []
    matrics = []
    ax_scatters = []

    for i, a in enumerate(range(-9, 10)):

        angle = a * 5

        # need to rotate around v axis to rotate on u
        rot_matrix = util_la.rotation_matrix_around_vector(rotate_vector, rotate_point, angle)

        color = plt.cm.get_cmap('winter_r')(i * 20)

        source_cross_rotate = copy.deepcopy(source_pcd).transform(rot_matrix)

        _, rgbd_cross_u_flatten, _ = util_la.project_to_plane_vectorized(
            np.asarray(source_cross_rotate.points), rotate_point, rotate_vector
        )

        rmse = util_la.compute_distance_rmse(target_pcd, source_cross_rotate)

        rmses.append(rmse)
        angles.append(angle)
        matrics.append(rot_matrix)

        # print(i, angle, rmse)

        if ax is not None:
            ax_scatters.append(
                ax.scatter(*rgbd_cross_u_flatten.T, color=color, s=0.1, marker='.', alpha=0.7)
            )


    # find the minimum rmse rotation angle
    rmses = np.asarray(rmses)

    best_idx = np.argmin(rmses)
    best_angle = angles[best_idx]
    best_matrix = matrics[best_idx]

    print(f"Minimum rMSE {rmses[best_idx]} at angle={best_angle}")

    if ax is not None:
        # change the selected color
        ax_scatters[best_idx].set_color('r')
        ax_scatters[best_idx].set_alpha(1)

        ax.set_aspect('equal')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([]) 

    out = {
        "rmses": rmses,
        "angles": np.asarray(angles),
        "matrics": matrics,
        "best_idx": best_idx,
        "best_angle": best_angle,
        "best_matrix": best_matrix
    }

    return out

def iterative_nuv_rotate(source_pcd, target_pcd, rotate_point, normal_vector):
    n = normal_vector
    u,v = util_la.calculate_normal_uv(n)

    sfm_cross_u, sfm_cross_v, sfm_u_id, sfm_v_id = get_nbr_cross(
        target_pcd, 
        plane_point=rotate_point,
        plane_normal=n,
        buffer=0.001,
        merge=False
    )

    rmses = []
    angles = []
    matrics = []

    for angle in range(1, 36):

        rot_matrix = util_la.rotation_matrix_around_vector(n, rotate_point, angle*10)
        # rgbd -> to rotate
        source_n_rotate_pcd = copy.deepcopy(source_pcd).transform(rot_matrix)

        rgbd_cross_u, rgbd_cross_v, rgbd_u_id, rgbd_v_id = get_nbr_cross(
            source_n_rotate_pcd, 
            rotate_point,
            normal_vector,
            buffer=0.001, merge=False
        )

        u_out = iterative_uv_rotate_one(rgbd_cross_u, sfm_cross_u, rotate_point, u)
        v_out = iterative_uv_rotate_one(rgbd_cross_v, sfm_cross_v, rotate_point, v)

        # plt.show()
        
        optimized_rotate_matrix = u_out['best_matrix'] @ v_out['best_matrix'] @ rot_matrix

        source_rotate_n_pcd = copy.deepcopy(source_pcd).transform(optimized_rotate_matrix)

        rmse = util_la.compute_distance_rmse(source_rotate_n_pcd, target_pcd)

        rmses.append(rmse)
        matrics.append(optimized_rotate_matrix)
        angles.append(angle * 10)

        print(f"- Rotate {angle * 10} | RMSE={rmse}")

    angles = np.asarray(angles)
    rmses = np.asarray(rmses)
    
    return angles, rmses, matrics