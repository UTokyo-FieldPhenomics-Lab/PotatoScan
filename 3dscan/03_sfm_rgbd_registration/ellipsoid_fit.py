# from https://github.com/aleksandrbazhin/ellipsoid_fit_python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_regularize(data, type="spherical", divs=10):
    limits = np.array([
        [min(data[:, 0]), max(data[:, 0])],
        [min(data[:, 1]), max(data[:, 1])],
        [min(data[:, 2]), max(data[:, 2])]])
        
    regularized = []

    if type == "cubic": # take mean from points in the cube
        
        X = np.linspace(*limits[0], num=divs)
        Y = np.linspace(*limits[1], num=divs)
        Z = np.linspace(*limits[2], num=divs)

        for i in range(divs-1):
            for j in range(divs-1):
                for k in range(divs-1):
                    points_in_sector = []
                    for point in data:
                        if (point[0] >= X[i] and point[0] < X[i+1] and
                                point[1] >= Y[j] and point[1] < Y[j+1] and
                                point[2] >= Z[k] and point[2] < Z[k+1]):
                            points_in_sector.append(point)
                    if len(points_in_sector) > 0:
                        regularized.append(np.mean(np.array(points_in_sector), axis=0))

    elif type == "spherical": #take mean from points in the sector
        divs_u = divs 
        divs_v = divs * 2

        center = np.array([
            0.5 * (limits[0, 0] + limits[0, 1]),
            0.5 * (limits[1, 0] + limits[1, 1]),
            0.5 * (limits[2, 0] + limits[2, 1])])
        d_c = data - center
    
        #spherical coordinates around center
        r_s = np.sqrt(d_c[:, 0]**2. + d_c[:, 1]**2. + d_c[:, 2]**2.)
        d_s = np.array([
            r_s,
            np.arccos(d_c[:, 2] / r_s),
            np.arctan2(d_c[:, 1], d_c[:, 0])]).T

        u = np.linspace(0, np.pi, num=divs_u)
        v = np.linspace(-np.pi, np.pi, num=divs_v)

        for i in range(divs_u - 1):
            for j in range(divs_v - 1):
                points_in_sector = []
                for k, point in enumerate(d_s):
                    if (point[1] >= u[i] and point[1] < u[i + 1] and
                            point[2] >= v[j] and point[2] < v[j + 1]):
                        points_in_sector.append(data[k])

                if len(points_in_sector) > 0:
                    regularized.append(np.mean(np.array(points_in_sector), axis=0))
# Other strategy of finding mean values in sectors
#                    p_sec = np.array(points_in_sector)
#                    R = np.mean(p_sec[:,0])
#                    U = (u[i] + u[i+1])*0.5
#                    V = (v[j] + v[j+1])*0.5
#                    x = R*math.sin(U)*math.cos(V)
#                    y = R*math.sin(U)*math.sin(V)
#                    z = R*math.cos(U)
#                    regularized.append(center + np.array([x,y,z]))
    return np.array(regularized)


# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)

def set_axes_equal(ax: plt.Axes):
    ax.set_box_aspect([1,1,1])
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    x, y, z = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])



# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v