import sys
import math
import time
import logging

import numpy as np
import cv2
import pyrealsense2 as rs

from shapely.geometry import Polygon
from descartes import PolygonPatch

M2TOCM2 = 10000
CMTOM = 0.01

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6",
                   "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A"]
ORANGE = [249, 115, 6]


def find_device_that_supports_advanced_mode(ctx, devices):
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                logging.info("Found device that supports advanced mode: %r", dev.get_info(rs.camera_info.name))
            return dev
    return None


def enable_advanced_mode(advnc_mode):
    """Attempts to enable advanced mode
    """
    # Loop until we successfully enable advanced mode
    while not advnc_mode.is_enabled():
        logging.info("Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        logging.info("Device disconnecting. Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        if dev is None:
            logging.error("Device did not reconnect! Exiting")
            sys.exit(1)
        advnc_mode = rs.rs400_advanced_mode(dev)
        logging.info("Advanced mode is %r", "enabled" if advnc_mode.is_enabled() else "disabled")

    return advnc_mode


def load_setting_file(ctx, devices, setting_file):
    """Loads a setting file

    Arguments:
        ctx {ctx} -- RS context
        devices {device} -- Realsense device
        setting_file {str} -- Path to settings file

    Returns:
        bool -- True if successful
    """
    dev = find_device_that_supports_advanced_mode(ctx, devices)
    if dev is None:
        logging.error("No device supports the advanced mode! Can not upload settings file: %r", setting_file)
        return None
    advnc_mode = rs.rs400_advanced_mode(dev)
    logging.info("Advanced mode is %r", "enabled" if advnc_mode.is_enabled() else "disabled")
    advnc_mode = enable_advanced_mode(advnc_mode)
    # Read settings file as a string
    with open(setting_file, 'r') as file:
        settings_json_str = file.read()
    advnc_mode.load_json(settings_json_str)
    return True


def rotation_matrix(x_theta=90):
    theta_rad = math.radians(x_theta)
    rotation_matrix = np.array([[1, 0, 0], [0, math.cos(theta_rad), -math.sin(theta_rad)],
                                [0, math.sin(theta_rad), math.cos(theta_rad)]])
    return rotation_matrix


def get_intrinsics(pipeline, stream=rs.stream.color):
    """Get intrinics for specified stream

    Arguments:
        pipeline {rs::pipeline} -- The pipeline that has been configured

    Keyword Arguments:
        stream {rs::stream::type} -- Stream Type (default: {rs.stream.color})

    Returns:
        rs::intrinsics -- The instrinsics object
    """
    streams = [stream_ for stream_ in pipeline.get_active_profile().get_streams() if stream_.stream_type() == stream]
    intrinsics = None
    if streams:
        intrinsics = streams[0].as_video_stream_profile().get_intrinsics()
    return intrinsics


def create_projection_matrix(intrinsics):
    fx, fy, ppx, ppy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    proj_mat = np.array([[fx, 0, ppx, 0], [0, fy, ppy, 0], [0, 0, 1, 0]])
    return proj_mat


def project_points_img(points, proj_mat, width, height):
    """Projects points into image given a projection matrix

    Arguments:
        points {ndarray} -- 3D points
        proj_mat {ndarray, 3X4} -- Projection Matrix
        width {int} -- width of image
        height {height} -- height of image

    Returns:
        ndarray -- pixels
    """
    pixels = proj_mat.dot(points)
    pixels = np.divide(pixels[:2, :], pixels[2, :]).transpose().astype(np.int)

    # Remove pixels that are outside the image
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height)
    # mask_x = (pixels[:, 0] < width) & (pixels[:, 0] > 0)
    # mask_y = (pixels[:, 1] < height) & (pixels[:, 1] > 0)

    # # Return the pixels and points that are inside the image
    # pixels = pixels[mask_x & mask_y]
    return pixels


def axis_angle_rm(axis=np.array([1, 0, 0]), angle=-1.57):
    """
    Create rotation matrix given an axis and angle
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    """
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis[0], axis[1], axis[2]
    rotation_matrix = np.array(
        [
            [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ])
    return rotation_matrix


def filter_zero(points_np):
    """
    Filter out all zero vectors (3D)
    """
    # TODO replace with cython or numba
    A = points_np[:, 0]
    B = points_np[:, 1]
    C = points_np[:, 2]
    t0 = time.time()
    mask = A == 0.0
    mask = mask & (B == 0.0)
    mask = mask & (C == 0.0)
    points_np = points_np[~mask]
    # print(f"Filtering Took {(time.time() - t0) * 1000:.1f} ms")
    return points_np


def get_patches(points, h, w, patches=[[450, 100, 10, 10], [450, 500, 10, 10]], min_valid_percent=0.75):
    """
    Get a list of point clouds from a list of region patches in the depth image
    """
    pc_image = points.reshape((h, w, 3))
    pc_patches = []
    for patch in patches:
        possible_patch = pc_image[patch[0]:patch[0] + patch[2], patch[1]:patch[1] + patch[3]]
        possible_patch = possible_patch.reshape(possible_patch.size // 3, 3)
        possible_patch = filter_zero(possible_patch)
        if possible_patch.shape[0] > min_valid_percent * (patch[2] * patch[3]):
            pc_patches.append(possible_patch)
    return pc_patches


def get_downsampled_patch(points, h, w, patch=[.75, 1.0, .10, 0.9], ds=[10, 10]):
    """
    Points (NX3) are extracted from a region in the depth image.
    Image height (h) and width (w) are given. Region is given by patch
    The points are then downsampled by ony selecting every ds point
    """
    t0 = time.time()
    pc_image = points.reshape((h, w, 3))
    ys = int(patch[0] * h)
    ye = int(patch[1] * h)
    xs = int(patch[2] * w)
    xe = int(patch[3] * w)
    patch = pc_image[ys:ye:ds[0], xs:xe:ds[1]]
    patch = patch.reshape(patch.size // 3, 3)
    patch = filter_zero(patch)

    # print(f"Downampled Patch: {(time.time() - t0) * 1000:.1f} ms")

    return patch


def get_patch(pc_image, h, w, patch=[.75, 1.0, .10, 0.9], ds=[2, 2]):
    ys = int(patch[0] * h)
    ye = int(patch[1] * h)
    xs = int(patch[2] * w)
    xe = int(patch[3] * w)
    patch = pc_image[ys:ye:ds[0], xs:xe:ds[1]]
    patch = patch.reshape(patch.size // 3, 3)
    patch = filter_zero(patch)
    return patch


def get_downsampled_patch_advanced(points, h, w, patch=[.50, 1.0, .10, 0.9], ds=[10, 10]):
    """Dowsample a patch semi-intelligently. Less downsampling near the top of the image

    Arguments:
        points {ndarray} -- NX3
        h {int} -- height
        w {width} -- width

    Keyword Arguments:
        patch {list} -- Normalized coordinate of image to downsample from (xmin, xmax, ymim, ymax) (default: {[.50, 1.0, .10, 0.9]})
        ds {list} -- Hom many row/cols to skip (default: {[[3, 4, 5], 10]})

    Returns:
        ndarray -- NX3 point cloud
    """
    t0 = time.time()
    ys = int(patch[0] * h)
    ye = int(patch[1] * h)
    xs = int(patch[2] * w)
    xe = int(patch[3] * w)
    pc_image = points.reshape((h, w, 3))

    v_spacing = ds[0]
    if isinstance(v_spacing, list):
        # advanced row spacing, greater row gaps the closer to the camera (bottom of image)
        h_actual = ye - ys
        partition = h_actual // len(v_spacing)  # number of partitions
        top = np.arange(0, partition, v_spacing[0])
        mid = np.arange(top[-1] + v_spacing[1], int(2 * partition), v_spacing[1])
        bottom = np.arange(mid[-1] + v_spacing[2], h_actual, v_spacing[2])
        row_indices = np.concatenate((top, mid, bottom)) + ys
    else:
        row_indices = np.arange(ys, ye, v_spacing)

    patch = pc_image[row_indices, xs:xe:ds[1]]
    patch = patch.reshape(patch.size // 3, 3)
    patch = filter_zero(patch)
    return patch


def rotate_points(points, rot):
    """
    Rotate 3D points given a provided rotation matrix
    """
    t0 = time.time()
    points_rot = points.transpose()
    points_rot = rot @ points_rot
    points_rot = points_rot.transpose()
    # print(f"Rotation Took {(time.time() - t0) * 1000:.1f} ms")
    return points_rot


def get_normal(points):
    points = points - np.mean(points, axis=0)
    u, s, vh = np.linalg.svd(points, compute_uv=True)
    # GET THE LAST ROW!!!!!!!NOT LAST COLUMN
    return vh[-1, :]


def calculate_plane_normal(patches):
    """
    Get normal of all the patches
    """
    normals = []
    for patch in patches:
        normal = get_normal(patch)
        normals.append(normal)
    # Taken naive mean of normals
    # TODO outlier removal
    normals = np.mean(np.array(normals), axis=0)
    return normals


def align_vector_to_zaxis(points, vector=np.array([0, 0, 1])):
    """
    Aligns z axis frame to chosen vector
    """
    # Shortcut from computing cross product of -Z axis X vector
    axis = np.cross(vector, np.array([0, 0, -1]))
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(-vector[2])

    rm = axis_angle_rm(axis, angle)
    points_rot = rotate_points(points, rm)
    return points_rot, rm


def get_point(pi, points):
    return [points[pi, 0], points[pi, 1], points[pi, 2]]


def filter_planes_and_holes(polygons, points, config):
    """Extracts the plane and obstacles returned from polylidar
    Will filter polygons according to: number of vertices and size
    Will also buffer (dilate) and simplify polygons

    Arguments:
        polygons {list[Polygons]} -- A list of polygons returned from polylidar
        points {ndarray} -- MX3 array
        config {dict} -- Configuration for filtering

    Returns:
        tuple -- A list of plane shapely polygons and a list of obstacle polygons
    """
    # filtering configuration
    pot_post = config['polygon']['postprocess']
    pot_filter = pot_post['filter']

    # will hold the plane(s) and obstacles found
    planes = []
    obstacles = []
    for poly in polygons:
        shell_coords = [get_point(pi, points) for pi in poly.shell]
        outline = Polygon(shell=shell_coords)

        outline = outline.buffer(distance=CMTOM*pot_post['buffer'])
        outline = outline.simplify(tolerance=CMTOM*pot_post['simplify'])
        area = outline.area * M2TOCM2
        if area >= pot_filter['plane_area']['min']:
            # Capture the polygon as well as its z height
            planes.append((outline, shell_coords[0][2]))

            for hole_poly in poly.holes:
                # Filter by number of obstacle vertices, removes noisy holes
                if len(hole_poly) > pot_filter['hole_vertices']['min']:
                    shell_coords = [get_point(pi, points) for pi in hole_poly]
                    outline = Polygon(shell=shell_coords)
                    area = outline.area * M2TOCM2
                    # filter by area
                    if area >= pot_filter['hole_area']['min'] and area < pot_filter['hole_area']['max']:
                        outline = outline.buffer(distance=CMTOM*pot_post['buffer'])
                        outline = outline.simplify(tolerance=CMTOM*pot_post['simplify'])
                        obstacles.append((outline, shell_coords[0][2]))
    return planes, obstacles


def plot_shapely_polys(polygons, ax, color='green'):
    for poly in polygons:
        outlinePatch = PolygonPatch(poly, ec=color, fill=False, linewidth=2)
        ax.add_patch(outlinePatch)


def get_pix_coordinates(pts, proj_mat, w, h):
    """Get Pixel coordinates of ndarray

    Arguments:
        pts {ndarray} -- 3D point clouds 3XN
        proj_mat {ndarray} -- 4X3 Projection Matrix
        w {int} -- width
        h {int} -- height

    Returns:
        ndarray -- Pixel coordinates
    """
    points_t = np.ones(shape=(4, pts.shape[1]))
    points_t[:3, :] = pts
    pixels = project_points_img(points_t, proj_mat, w, h)
    return pixels


def plot_opencv_polys(polygons, color_image, proj_mat, rot_mat, w, h, color=(0, 255, 0), thickness=2):
    for i, (poly, height) in enumerate(polygons):
        # Get 2D polygons and assign z component the height value of extracted plane
        pts = np.array(poly.exterior.coords)  # NX2
        pts = np.column_stack((pts, np.ones((pts.shape[0])) * height))  # NX3
        # Transform flat plane coordinate system to original cordinate system of depth frame
        pts = pts.transpose()  # 3XN
        pts = np.linalg.inv(rot_mat) @ pts

        # np.savetxt(f"polygon_{i}_cameraframe.txt", pts.transpose())
        # Project coordinates to image space
        pix_coords = get_pix_coordinates(pts, proj_mat, w, h)
        pix_coords = pix_coords.reshape((-1, 1, 2))
        cv2.polylines(color_image, [pix_coords], True, color, thickness=thickness)


def plot_planes_and_obstacles(planes, obstacles, proj_mat, rot_mat, color_image, config, thickness=2):
    """Plots the planes and obstacles (3D polygons) into the color image

    Arguments:
        planes {list(Polygons)} -- List of Shapely Polygon with height tuples
        obstacles {list[(polygon, height)]} -- List of tuples with polygon with height
        proj_mat {ndarray} -- Projection Matrix
        rot_mat {ndarray} -- Rotation Matrix
        color_image {ndarray} -- Color Image
        config {dict} -- Configuration
    """
    plot_opencv_polys(
        planes, color_image, proj_mat, rot_mat, config['color']['width'],
        config['color']['height'], color=(0, 255, 0), thickness=thickness)

    plot_opencv_polys(
        obstacles, color_image, proj_mat, rot_mat, config['color']['width'],
        config['color']['height'], color=ORANGE,  thickness=thickness)


def plot_polygons(polygons, points, ax):
    for poly in polygons:
        shell_coords = [get_point(pi, points) for pi in poly.shell]
        outline = Polygon(shell=shell_coords)
        outlinePatch = PolygonPatch(outline, ec='green', fill=False, linewidth=2)
        ax.add_patch(outlinePatch)

        for hole_poly in poly.holes:
            shell_coords = [get_point(pi, points) for pi in hole_poly]
            outline = Polygon(shell=shell_coords)
            outlinePatch = PolygonPatch(outline, ec='orange', fill=False, linewidth=2)
            ax.add_patch(outlinePatch)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
