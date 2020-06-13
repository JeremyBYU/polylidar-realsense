import logging
import sys
import argparse
from os import path
import time
import uuid
import math
import itertools


import numpy as np
import yaml
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

from polylidar import MatrixDouble, MatrixFloat, extract_point_cloud_from_float_depth, Polylidar3D
from fastga import GaussianAccumulatorS2, IcoCharts

from polylidar.polylidarutil.plane_filtering import filter_planes_and_holes
from grounddetector.helper import (align_vector_to_zaxis, get_downsampled_patch, calculate_plane_normal,
                                   filter_zero, plot_planes_and_obstacles, create_projection_matrix,
                                   get_intrinsics, get_downsampled_patch_advanced,
                                   load_setting_file, rotate_points)

from grounddetector.helper_mesh import create_meshes_cuda, create_meshes_cuda_with_o3d
from grounddetector.helper_updated import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh

logging.basicConfig(level=logging.INFO)


THIS_DIR = path.dirname(__file__)
CONFIG_DIR = path.join(THIS_DIR, "config")
ASSETS_DIR = path.join(THIS_DIR, '..', 'assets')
VID_DIR = path.join(ASSETS_DIR, 'videos')
PICS_DIR = path.join(ASSETS_DIR, 'pics')
DEFAULT_CONFIG_FILE = path.join(CONFIG_DIR, "default.yaml")

IDENTITY = np.identity(3)
IDENTITY_MAT = MatrixDouble(IDENTITY)


axis = o3d.geometry.TriangleMesh.create_coordinate_frame()


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def create_arrow(scale=1, cylinder_radius=None, **kwargs):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2
    cylinder_height = scale * 0.8
    cone_radius = cylinder_radius if cylinder_radius else scale / 10
    cylinder_radius = cylinder_radius if cylinder_radius else scale / 20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
                                                        cone_height=cone_height,
                                                        cylinder_radius=cylinder_radius,
                                                        cylinder_height=cylinder_height)
    return(mesh_frame)


def get_arrow(origin=[0, 0, 0], end=None, vec=None, **kwargs):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    # print(end)
    scale = 10
    beta = 0
    gamma = 0
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        mesh = create_arrow(scale, **kwargs)
        axis, angle = align_vector_to_another(b=vec / scale)
        if axis is None:
            axis_a = axis
        else:
            axis_a = axis * angle
            rotation_3x3 = mesh.get_rotation_matrix_from_axis_angle(axis_a)
    # mesh.transform(T)
    if axis is not None:
        mesh = mesh.rotate(rotation_3x3, center=False)
    mesh.translate(origin)
    return(mesh)


def create_pipeline(config):
    """Sets up the pipeline to extract depth and rgb frames

    Arguments:
        config {dict} -- A dictionary mapping for configuration. see default.yaml

    Returns:
        tuple -- pipeline, pointcloud, decimate, filters(list), colorizer (not used)
    """

    # Ensure device is connected
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        logging.error("No connected Intel Realsense Device!")
        sys.exit(1)

    if config['advanced']:
        logging.info("Attempting to enter advanced mode and upload JSON settings file")
        load_setting_file(ctx, devices, config['advanced'])

    # Configure streams
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.depth, config['depth']['width'],
        config['depth']['height'],
        rs.format.z16, config['depth']['framerate'])
    # other_stream, other_format = rs.stream.infrared, rs.format.y8
    rs_config.enable_stream(
        rs.stream.color, config['color']['width'],
        config['color']['height'],
        rs.format.rgb8, config['color']['framerate'])

    # Start streaming
    pipeline.start(rs_config)
    profile = pipeline.get_active_profile()

    # Processing blocks
    filters = []
    decimate = None
    align = rs.align(rs.stream.color)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    # Decimation
    if config.get("filters").get("decimation"):
        filt = config.get("filters").get("decimation")
        if filt.get('active', True):
            filt.pop('active', None)  # Remove active key before passing params
            decimate = rs.decimation_filter(**filt)

    # Spatial
    if config.get("filters").get("spatial"):
        filt = config.get("filters").get("spatial")
        if filt.get('active', True):
            filt.pop('active', None)  # Remove active key before passing params
            my_filter = rs.spatial_filter(**filt)
            filters.append(my_filter)

    # Temporal
    if config.get("filters").get("temporal"):
        filt = config.get("filters").get("temporal")
        if filt.get('active', True):
            filt.pop('active', None)  # Remove active key before passing params
            my_filter = rs.temporal_filter(**filt)
            filters.append(my_filter)

    process_modules = (align, depth_to_disparity, disparity_to_depth, decimate)
    # Create colorizer and point cloud
    # colorizer = rs.colorizer(2)
    pc = rs.pointcloud()

    intrinsics = get_intrinsics(pipeline, rs.stream.color)
    proj_mat = create_projection_matrix(intrinsics)

    return pipeline, pc, process_modules, filters, proj_mat


def get_frames(pipeline, pc, process_modules, filters, config):
    """Extracts frames from intel real sense pipline
    Applies filters and extracts point cloud
    Arguments:
        pipeline {rs::pipeline} -- RealSense Pipeline
        pc {rs::pointcloud} -- A class that can turn a depth frame into a point cloud (numpy array)
        process_modules {tuple} -- align, depth_to_disparity, disparity_to_depth, decimate
        filters {list[rs::filters]} -- List of filters to apply

    Returns:
        (rgb_image, depth_image, ndarray, meta) -- RGB Image, Depth Image (colorized), numpy points cloud, meta information
    """
    success, frames = pipeline.try_wait_for_frames(timeout_ms=5)
    if not success:
        return None, None, None
    # Get all the standard process modules
    (align, depth_to_disparity, disparity_to_depth, decimate) = process_modules

    # Align depth frame with color frame.  For later overlap
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Decimate, subsample image
    if decimate:
        depth_frame = decimate.process(depth_frame)
    # Depth to disparity
    depth_frame = depth_to_disparity.process(depth_frame)
    # Apply any filters we requested
    for f in filters:
        depth_frame = f.process(depth_frame)
    # Disparity back to depth
    depth_frame = disparity_to_depth.process(depth_frame)
    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    d_intrinsics_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                    [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                    [0, 0, 1]])
    # convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image, dict(h=h, w=w, intrinsics=d_intrinsics_matrix)


def get_polygon(depth_image: np.ndarray, config, ll_objects, h, w, intrinsics, **kwargs):
    """Extract polygons from point cloud

    Arguments:
        points {ndarray} -- NX3 numpy array
        config {dict} -- Configuration object
        h {int} -- height
        w {int} -- width

    Returns:
        tuple -- polygons, rotated downsample points, and rotation matrix
    """

    # 1. Convert depth frame to downsampled organized point cloud
    # 2. Create a smooth mesh from the organized point cloud (OrganizedPointFilters)
    #     1. You can skip smoothing if desired and only rely upon Intel Realsense SDK
    # 3. Estimate dominate plane normals in scene (FastGA)
    # 4. Extract polygons from mesh using dominant plane normals (Polylidar3D)

    alg_timings = dict()

    # 1. Create OPC
    stride = config['mesh']['stride']  # point cloud generation parameters
    depth_image = np.divide(depth_image, 1000.0, dtype=np.float32)
    points = extract_point_cloud_from_float_depth(MatrixFloat(
        depth_image), MatrixDouble(intrinsics), IDENTITY_MAT, stride=stride)
    new_shape = (int(depth_image.shape[0] / stride), int(depth_image.shape[1] / stride), 3)
    opc = np.asarray(points).reshape(new_shape)  # organized point cloud (will have NaNs!)

    # 2. Create Mesh and Smooth (all in one)
    # mesh, o3d_mesh, timings = create_meshes_cuda_with_o3d(opc, **config['mesh']['filter'])
    mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
    alg_timings.update(timings)
    # TODO add a commented line that only does mesh creation

    # 3. Estimate Dominate Plane Normals
    fga = config['fastga']
    avg_peaks, _, _, _, timings = extract_all_dominant_plane_normals(
        mesh, ga_=ll_objects['ga'], ico_chart_=ll_objects['ico'], **fga)
    alg_timings.update(timings)
    # print(avg_peaks)

    # 4. Extract Polygons from mesh
    planes, obstacles, timings = extract_planes_and_polygons_from_mesh(mesh, avg_peaks, pl_=ll_objects['pl'],
                                                                                       filter_polygons=True, optimized=True,
                                                                                       postprocess=config['polygon']['postprocess'])
    alg_timings.update(timings)

    return planes, obstacles, alg_timings
    # return planes, obstacles, timings, mesh, o3d_mesh, o3d_mesh_painted, arrow_o3d, all_poly_lines
    # return polygons, points_rot, rm


def valid_frames(color_image, depth_image, depth_min_valid=0.5):
    """Determines if returned color and depth images are valid for polygon extraction

    Arguments:
        color_image {ndarray} -- Color image
        depth_image {ndarray} -- Depth image

    Keyword Arguments:
        depth_min_valid {float} -- Minimum percentage of valid pixels in depth frame (default: {0.5})

    Returns:
        bool -- Whether frames are valid
    """
    count = np.count_nonzero(depth_image)
    pct = count / depth_image.size

    pass_depth = pct > depth_min_valid

    pass_all = pass_depth  # maybe others to come
    return pass_all


def colorize_images_open_cv(color_image, depth_image, config):
    """Colorizes and resizes images"""

    color_image_cv = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    depth_image_cv = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HOT)
    depth_image_cv = cv2.resize(depth_image_cv, (config['color']['width'], config['color']['height']))

    return color_image_cv, depth_image_cv


def capture(config, video=None):
    # Configure streams
    pipeline, pc, process_modules, filters, proj_mat = create_pipeline(config)
    logging.info("Pipeline Created")

    # Long lived objects. These are the object that hold all the algorithms for surface exraction.
    # They need to be long lived (objects) because they hold state (thread scheduler, image datastructures, etc.)
    ll_objects = dict()
    ll_objects['pl'] = Polylidar3D(**config['polylidar'])
    ll_objects['ga'] = GaussianAccumulatorS2(level=config['fastga']['level'])
    ll_objects['ico'] = IcoCharts(level=config['fastga']['level'])

    if video:
        frame_width = config['depth']['width'] * 2
        frame_height = config['depth']['height']
        out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    try:
        while True:
            t00 = time.time()
            color_image, depth_image, meta = get_frames(pipeline, pc, process_modules, filters, config)
            # color_image = cv2.resize(color_image, (config['color']['width']//2, config['color']['height']//2))
            t0 = time.time()
            if color_image is None or not valid_frames(color_image, depth_image, **config['polygon']['frameskip']):
                logging.debug("Invalid Frames")
                continue
            t1 = time.time()

            try:
                if config['show_polygon']:
                    # planes, obstacles, timings, mesh, o3d_mesh, o3d_mesh_painted, arrow_o3d, all_poly_lines = get_polygon(depth_image, config, ll_objects, **meta)
                    planes, obstacles, timings = get_polygon(depth_image, config, ll_objects, **meta)
                    # continue
                    # t2 = time.time()
                    # planes, obstacles = filter_planes_and_holes(polygons, points_rot, config['polygon']['postprocess'])
                    # t3 = time.time()
                    # Plot polygon in rgb frame
                    plot_planes_and_obstacles(planes, obstacles, proj_mat, None, color_image, config)

                t4 = time.time()
                # Show images
                if config.get("show_images"):
                    # Convert to open cv image types (BGR)
                    color_image_cv, depth_image_cv = colorize_images_open_cv(color_image, depth_image, config)
                    # Stack both images horizontally
                    images = np.hstack((color_image_cv, depth_image_cv))
                    cv2.imshow('RealSense Color/Depth (Aligned)', images)
                    if video:
                        out_vid.write(images)
                    res = cv2.waitKey(1)
                    if res == ord('p'):
                        uid = uuid.uuid4()
                        logging.info("Saving Picture: {}".format(uid))
                        cv2.imwrite(path.join(PICS_DIR, "{}_color.jpg".format(uid)), color_image_cv)
                        cv2.imwrite(path.join(PICS_DIR, "{}_stack.jpg".format(uid)), images)
                    if res == ord('m'):
                        plt.imshow(np.asarray(ll_objects['ico'].image))
                        plt.show()
                        all_lines = [line_mesh.cylinder_segments for line_mesh in all_poly_lines]
                        flatten = itertools.chain.from_iterable
                        all_lines = list(flatten(all_lines))

                        arrow_o3d = arrow_o3d.translate([0, 0, 1.3])
                        # import ipdb; ipdb.set_trace()

                        o3d.visualization.draw_geometries([axis, o3d_mesh_painted, arrow_o3d, *all_lines])
                # print(timings)
                logging.info(f"Get Frames: %.2f; Check Valid Frame: %.2f; Laplacian: %.2f; Bilateral: %.2f; Mesh: %.2f; FastGA: %.2f; Plane/Poly: %.2f; Filtering: %.2f",
                             (t0 - t00) * 1000, (t1 - t0) * 1000,  timings['t_laplacian'], timings['t_bilateral'], timings['t_mesh'], timings['t_fastga_total'], timings['t_polylidar_planepoly'], timings['t_polylidar_filter'])
            except Exception as e:
                logging.exception("Error!")

    finally:
        pipeline.stop()
    if video is not None:
        out_vid.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Captures ground plane and obstacles as polygons")
    parser.add_argument('-c', '--config', help="Configuration file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-v', '--video', help="Video file save path", default=None)
    args = parser.parse_args()
    with open(args.config) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            logging.exception("Error parsing yaml")

    # Run capture loop
    capture(config, args.video)


if __name__ == "__main__":

    main()
