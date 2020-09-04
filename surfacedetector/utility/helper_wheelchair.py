import time
import logging
import itertools

from shapely.geometry import Polygon
import numpy as np
from polylidar import HalfEdgeTriangulation

from surfacedetector.utility.helper_ransac import estimate_plane


def extract_geometric_plane(polygon: Polygon, plane_triangle_indices, tri_mesh: HalfEdgeTriangulation, normal: np.ndarray):
    """Will extract geometric details from the polygon and plane of interest
    Args:
        polygon (Polygon): Shapely Polygon of a flat surface
        plane_triangle_indices (ndarray uint64): Triangle indices of the plane in the mesh
        tri_mesh (HalfEdgeTriangulation): The mesh of the environment
        normal (np.ndarray): The surface normal that this plane was extracted on
    Returns:
        [type]: [description]
    """         
    # triangles:np.ndarray = np.asarray(tri_mesh.triangles)
    # vertices:np.ndarray = np.asarray(tri_mesh.vertices)
    # all_point_indices = triangles[plane_triangle_indices, :]
    # all_point_indices = np.reshape(all_point_indices, (np.prod(all_point_indices.shape), ))
    # all_point_indices = np.unique(all_point_indices)
    # all_points = vertices[all_point_indices, :]

    all_points = np.asarray(polygon.exterior.coords)
    # centroid = np.mean(all_points, axis=0) # polygon.centroid ?
    normal_ransac, centroid, _ = estimate_plane(all_points)

    return dict(point=centroid, normal=normal, all_points=all_points, area=polygon.area, normal_ransac=normal_ransac)


def analyze_planes(geometric_planes):
    """This will analyze all geometric planes that have been extracted and find the curb height
    Args:
        geometric_planes (List[dict]): A list of dicts representing the geometric planes
    Returns:
        float: Height of curb in meters
    """

    # This code will find the ground normal index, the index into geometric_planes
    # with the largest area of surfaces (e.g., the street and sidewalk)
    if len(geometric_planes) < 2:
        return 0.0
    max_area = 0.0
    ground_normal_index = 0
    mean_normal_ransac = np.array([0.0, 0.0, 0.0])
    for i, geometric_planes_for_normal in enumerate(geometric_planes):
        if len(geometric_planes_for_normal) > 1:
            total_area = 0.0
            total_normal_ransac = np.array([0.0, 0.0, 0.0])
            for j, plane in enumerate(geometric_planes_for_normal):
                logging.debug(
                    f"Plane {j} - Normal: {plane['normal']:}, Ransac Normal: {plane['normal_ransac']:}, Point: {plane['point']:}")
                # np.save(f'scratch/plane_{i}_{j}.npy', plane['all_points'])
                total_normal_ransac += plane['normal_ransac']
                total_area += plane['area']
            if total_area > max_area:
                max_area = total_area
                ground_normal_index = i
                mean_normal_ransac = total_normal_ransac / len(geometric_planes_for_normal)
                mean_normal_ransac = mean_normal_ransac / np.linalg.norm(mean_normal_ransac) * -1
    # This code will find the maximum orthogonal distance between any tow pair of surfaces with
    # the same normal
    max_orthogonal_distance = 0.0
    geometric_planes_for_normal = geometric_planes[ground_normal_index]
    for pair in itertools.combinations(range(len(geometric_planes_for_normal)), 2):
        # print(pair)
        first_plane = geometric_planes_for_normal[pair[0]]
        second_plane = geometric_planes_for_normal[pair[1]]
        orthoganal_distance = np.abs(mean_normal_ransac.dot(first_plane['point'] - second_plane['point']))
        if orthoganal_distance > max_orthogonal_distance:
            max_orthogonal_distance = orthoganal_distance

    logging.debug(f"Curb Height: {max_orthogonal_distance:}")

    return max_orthogonal_distance