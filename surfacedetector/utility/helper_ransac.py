"""This module will use RANSAC to estimate the plane of a point cloud.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import logging

from sklearn import linear_model




# Looks like this function needs to be compiled on first call or something similar.
# Below is a dummy calculation
# If I don't do this there is always a 23 ms overhead. Even with only 3 points!!
_, _, _, _ = np.linalg.lstsq(np.array([[0,0], [1, 0], [0, 1]]), np.array([0, 0, 0]), rcond=None)

def estimate_plane(pc:np.ndarray, stop_probability=0.95):
    """Will estimate a planes parameters using RANSAC
    Args:
        pc (np.ndarray): NX3 Point Cloud
        stop_probability (float, optional): sklearn.linear_model.See RANSACRegressor. Defaults to 0.95.
    Returns:
        tuple(ndarray, ndarray, RANSACRegressor): Returns a tuple of the centroid of the point cloud, unit normal, and the RANSAC object
    """
    x_train = pc[:, :2]
    y_train = pc[:, 2]
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), stop_probability=stop_probability)

    t0 = time.perf_counter()
    ransac.fit(x_train, y_train)
    t1 = time.perf_counter()
    ms  = (t1 - t0) * 1000

    logging.debug(f"Took {ms:.1f} ms to compute RANSAC for {pc.shape[0]:} points. # Trials: {ransac.n_trials_:}")

    z = lambda x,y: (ransac.estimator_.intercept_ + ransac.estimator_.coef_[0]*x + ransac.estimator_.coef_[1]*y)

    point_on_plane_0 = np.array([0, 0, z(0,0)])
    point_on_plane_1 = np.array([1, 0, z(1,0)])
    point_on_plane_2 = np.array([0, 1, z(0,1)])

    vec_a = point_on_plane_1 - point_on_plane_0
    vec_b = point_on_plane_2 - point_on_plane_0

    normal = np.cross(vec_a, vec_b)
    normal = normal / np.linalg.norm(normal, axis=0)
    centroid = np.mean(pc, axis=0)

    logging.debug("Normal: %r; Centroid: %r", normal, centroid)
    return normal, centroid, ransac


def plot_pc(pc, ax):
    normal, centroid, ransac = estimate_plane(pc)
    ax.plot3D(pc[:,0], pc[:,1], pc[:,2], 'or')
    ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=2.0)


def test_wheelchair():
    pc_1 = np.load("scratch/plane_0_0.npy")
    pc_2 = np.load("scratch/plane_0_1.npy")
    pc_3 = np.load("scratch/plane_0_2.npy")

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    plot_pc(pc_1, ax)
    plot_pc(pc_2, ax)
    plot_pc(pc_3, ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.view_init(-7,-68)
    plt.show()


def main():
    np.random.seed(1)
    slope_x = 0.5
    slope_y = 0.5
    intercept= 1.0
    noise = 0.5

    NUM_POINTS = 10

    X, Y = np.mgrid[0:NUM_POINTS, 0:NUM_POINTS]
    PC = np.column_stack((X.ravel(), Y.ravel()))

    Z = []
    for i in range(PC.shape[0]):
        Z.append(slope_x * PC[i,0] + slope_y * PC[i, 1] + intercept)
    Z = np.array(Z) + np.random.randn(PC.shape[0]) * noise

    PC = np.column_stack((PC[:,0], PC[:, 1], Z))

    normal, centroid, ransac = estimate_plane(PC)


    # # the plane equation
    z = lambda x,y: (ransac.estimator_.intercept_ + ransac.estimator_.coef_[0]*x + ransac.estimator_.coef_[1]*y)
    x,y = np.mgrid[0:NUM_POINTS,0:NUM_POINTS]

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot3D(PC[:,0], PC[:,1], PC[:,2], 'or')

    ax.plot_surface(x, y, z(x,y))
    ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=2.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([0, NUM_POINTS])
    ax.set_ylim([0, NUM_POINTS])
    ax.set_zlim([0, NUM_POINTS])
    ax.view_init(-7,-68)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_wheelchair()
    main()