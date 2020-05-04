import logging
import sys
import argparse
from os import path
import json
import time
import uuid
import traceback
import bisect
import math


import numpy as np
import yaml
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation as R
# from scipy.stats import describe
import open3d as o3d

from polylidar import extractPolygons, extract_uniform_mesh_from_float_depth, extract_polygons_from_mesh
from polylidarutil.plane_filtering import filter_planes_and_holes
from grounddetector.helper import (align_vector_to_zaxis, get_downsampled_patch, calculate_plane_normal,
                                   filter_zero, plot_planes_and_obstacles, create_projection_matrix,
                                   get_intrinsics, get_downsampled_patch_advanced,
                                   load_setting_file, rotate_points)

logging.basicConfig(level=logging.INFO)


THIS_DIR = path.dirname(__file__)
CONFIG_DIR = path.join(THIS_DIR, "config")
ASSETS_DIR = path.join(THIS_DIR, '..', 'assets')
VID_DIR = path.join(ASSETS_DIR, 'videos')
PICS_DIR = path.join(ASSETS_DIR, 'pics')
DEFAULT_CONFIG_FILE = path.join(CONFIG_DIR, "default.yaml")


# T265 to D400
H_t265_d400 = np.array([
    [1, 0, 0, 0],
    [0, -1.0, 0, 0],
    [0, 0, -1.0, 0],
    [0, 0, 0, 1]])

H_Standard_t265 = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]])

T265_ROTATION = []
T265_TIMES = []
MAX_POSES = 100


def callback_pose(frame):
    global T265_TRANSLATION, T265_ROTATION
    try:
        ts = frame.get_timestamp()
        domain = frame.frame_timestamp_domain
        pose = frame.as_pose_frame()
        data = pose.get_pose_data()
        t = data.translation
        r = data.rotation
        T265_ROTATION.append([r.x, r.y, r.z, r.w])
        T265_TIMES.append(int(ts))
        if len(T265_TIMES) >= MAX_POSES:
            T265_ROTATION.pop(0)
            T265_TIMES.pop(0)
    except Exception:
        logging.exception("Error in callback for pose")


def get_pose_index(ts_):
    ts = int(ts_)
    idx = bisect.bisect_left(T265_TIMES, ts, lo=0)
    return min(idx, len(T265_TIMES) - 1)


def get_pose_matrix(ts_):
    logging.debug("Get Pose at %r", int(ts_))
    idx = get_pose_index(ts_)
    quat = T265_ROTATION[idx]
    ts = T265_TIMES[idx]

    H_t265_W = R.from_quat(quat).as_matrix()
    extrinsic = H_t265_W @ H_t265_d400[:3, :3]
    logging.debug("Frame TimeStamp: %r; Pose TimeStamp %r", int(ts_), ts)
    return extrinsic


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
    dev_d400 = None
    dev_t265 = None
    for dev in devices:
        dev_name = dev.get_info(rs.camera_info.name)
        print("Found {}".format(dev_name))
        if "Intel RealSense D4" in dev_name:
            dev_d400 = dev
        elif "Intel RealSense T265" in dev_name:
            dev_t265 = dev

    if len(devices) != 2:
        logging.error("Need 2 connected Intel Realsense Devices!")
        sys.exit(1)

    # Configure Depth Sensor and create Pieline
    if config['advanced']:
        logging.info("Attempting to enter advanced mode and upload JSON settings file")
        load_setting_file(ctx, devices, config['advanced'])

    # Configure streams
    rs_config = rs.config()
    pipeline = rs.pipeline(ctx)
    rs_config.enable_stream(
        rs.stream.depth, config['depth']['width'],
        config['depth']['height'],
        rs.format.z16, config['depth']['framerate'])
    # other_stream, other_format = rs.stream.infrared, rs.format.y8
    rs_config.enable_stream(
        rs.stream.color, config['color']['width'],
        config['color']['height'],
        rs.format.rgb8, config['color']['framerate'])

    # Open T265
    if dev_t265:
        # Unable to open as a pipeline, must use sensors
        sensor_t265 = dev_t265.query_sensors()[0]
        profiles = sensor_t265.get_stream_profiles()
        pose_profile = [profile for profile in profiles if profile.stream_name() == 'Pose'][0]
        sensor_t265.open(pose_profile)
        sensor_t265.start(callback_pose)

    # Start streaming
    pipeline.start(rs_config)

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

    return pipeline, pc, process_modules, filters, proj_mat, sensor_t265


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
    success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
    if not success:
        return None, None, None, None
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
    depth_intrinsics = rs.video_stream_profile(
        depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    d_intrinsics_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                    [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                    [0, 0, 1]])

    # convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_ts = depth_frame.get_timestamp()

    # use point clouds to calculate mesh
    if config['mesh_creation'] == 'delaunay':
        # Create point cloud
        points = pc.calculate(depth_frame)
        # extract ONLY 3D points from point cloud (pc) data structure; in camera frame (+Z is forward)
        points = np.asanyarray(points.get_vertices(2))
    else:
        # use depth image to create uniform mesh grid
        points = None

    return color_image, depth_image, points, dict(h=h, w=w, ts=depth_ts, intrinsics=d_intrinsics_matrix)


def t265_frame_to_standard(rm):
    return H_Standard_t265[:3, :3] @ rm


def get_downsampled_point_cloud(points, depth_image, config, h, w, intrinsics):
    """Return downsampled Point Cloud """
    points_config = config['polygon']['pointcloud']  # point cloud generation parameters
    h_ = h
    w_ = w
    if config['mesh_creation'] == 'uniform':
        # convert uint16 to float32 in meters. extract point cloud and triangular mesh
        depth_image = np.divide(depth_image, 1000.0, dtype=np.float32)
        stride = points_config['ds'][0]
        points, triangles, halfedges = extract_uniform_mesh_from_float_depth(depth_image, intrinsics, stride=stride)
        points, triangles, halfedges = np.asarray(points), np.asarray(triangles), np.asarray(halfedges)
        points = points.reshape((int(points.shape[0] / 3), 3))

        h_, w_ = math.ceil(h / stride), math.ceil(w / stride)
        return points, triangles, halfedges, h_, w_
    else:
        points = get_downsampled_patch_advanced(points, h_, w_, patch=points_config['patch'], ds=points_config['ds'])
        return points, None, None, h_, w_


def get_polygon(points, depth_image, config, h, w, intrinsics, rm=None, **kwargs):
    """Extract polygons from point cloud

    Arguments:
        points {ndarray} -- NX3 numpy array
        config {dict} -- Configuration object
        h {int} -- height
        w {int} -- width

    Returns:
        tuple -- polygons, rotated downsample points, and rotation matrix
    """
    polylidar_kwargs = config['polygon']['polylidar']  # polylidar parameters

    t1 = time.perf_counter()
    points_ds, triangles, halfedges, h_, w_ = get_downsampled_point_cloud(points, depth_image, config, h, w, intrinsics)
    t2 = time.perf_counter()
    if points is None:
        points = points_ds
    if rm is None:
        ground_config = config['polygon']['ground_normal']  # ground normal parameters
        # Get downsample ground patch
        ground_patch = [get_downsampled_patch(points, h_, w_, patch=ground_config['patch'], ds=ground_config['ds'])]
        normal = calculate_plane_normal(ground_patch)
        if normal[2] < 0:
            normal = np.negative(normal)
        # Rotate Points
        points_rot, rm = align_vector_to_zaxis(points_ds, normal)
        points_rot = np.ascontiguousarray(points_rot)
    else:
        points_rot = np.ascontiguousarray(rotate_points(points_ds, rm))


    bottom_points = 1000
    z_values = points_rot[:-bottom_points, 2]
    height = np.median(z_values)
    threshold = 0.025
    mask = points_rot[:, 2] >  (height + threshold)
    points_rot[mask, 2] = points_rot[mask, 2] + 0.2
    point_rot_temp = np.ascontiguousarray(points_rot[~mask, :])    

    t3 = time.perf_counter()
    polygons = None
    if config['mesh_creation'] == 'uniform':
        # extract polygons from provided mesh
        pass
        polygons = extract_polygons_from_mesh(points_rot, triangles, halfedges, **polylidar_kwargs)
    else:
        # exracts polygon from point cloud.
        polygons = extractPolygons(point_rot_temp, **polylidar_kwargs)
    t4 = time.perf_counter()

    # logging.info("Get DS Point Cloud and Mesh: %f; Extract Polygons %f", (t2-t1) * 1000, (t4-t3) * 1000)
    return polygons, point_rot_temp, points_rot, rm


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


def downsample_pc(pc, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)


def colorize_depth(depth_image, config, vmin=500, vmax=1500, bgr=True):
    depth_image_cv = cv2.resize(depth_image, (config['color']['width'], config['color']['height']))
    normalized_depth = Normalize(vmin=vmin, vmax=vmax, clip=True)(depth_image_cv)
    depth_image_cv = (plt.cm.viridis(normalized_depth)[:,:, :3] * 255).astype(np.uint8)
    if bgr:
        depth_image_cv = cv2.cvtColor(depth_image_cv, cv2.COLOR_RGB2BGR)
    return depth_image_cv

def colorize_images_open_cv(color_image, depth_image, config):
    """Colorizes and resizes images"""
    color_image_cv = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    depth_image_cv = colorize_depth(depth_image, config)

    return color_image_cv, depth_image_cv


def capture(config, video=None):
    # Configure streams
    pipeline, pc, process_modules, filters, proj_mat, sensor_t265 = create_pipeline(config)

    if video:
        frame_width = config['depth']['width'] * 2
        frame_height = config['depth']['height']
        out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    try:
        while True:
            t00 = time.time()
            color_image, depth_image, points, meta = get_frames(pipeline, pc, process_modules, filters, config)
            # color_image = cv2.resize(color_image, (config['color']['width']//2, config['color']['height']//2))
            t0 = time.time()
            if color_image is None or not valid_frames(color_image, depth_image, **config['polygon']['frameskip']):
                logging.debug("Invalid Frames")
                continue
            t1 = time.time()

            backup_color = color_image.copy()

            try:
                if config['show_polygon']:
                    r_matrix = get_pose_matrix(meta['ts'])
                    r_matrix = t265_frame_to_standard(r_matrix)
                    polygons, points_rot, points_rot_all, rot_mat = get_polygon(points, depth_image, config, rm=None, **meta)
                    t2 = time.time()
                    planes, obstacles = filter_planes_and_holes(polygons, points_rot, config['polygon']['postprocess'])
                    t3 = time.time()
                    # Plot polygon in rgb frame
                    plot_planes_and_obstacles(planes, obstacles, proj_mat, rot_mat, color_image, config)

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
                        cv2.imwrite(path.join(PICS_DIR, "{}_color_nopoly.jpg".format(uid)), cv2.cvtColor(backup_color, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(path.join(PICS_DIR, "{}_depth.jpg".format(uid)), depth_image_cv)
                        cv2.imwrite(path.join(PICS_DIR, "{}_stack.jpg".format(uid)), images)
                        np.savetxt(path.join(PICS_DIR, "{}_points.txt".format(uid)), points_rot_all)
                        np.savetxt(path.join(PICS_DIR, "{}_depth_raw.txt".format(uid)), depth_image)

                        meta = dict(rm=r_matrix.tolist(), proj_mat=proj_mat.tolist(), w=config['color']['width'], h=config['color']['height'], rot_mat=rot_mat.tolist())
                        with open(path.join(PICS_DIR, "{}_meta.json".format(uid)), 'w') as fp:
                            json.dump(meta, fp, indent=4)


                    if res == ord('o'):
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_rot)
                        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        axis_frame = axis_frame.translate([0, 0, -0.7])
                        o3d.visualization.draw_geometries([pcd, axis_frame])

                logging.info("Get Frames: %.2f; Check Valid Frame: %.2f; Polygon Extraction: %.2f, Polygon Filtering: %.2f, Visualization: %.2f",
                             (t0 - t00) * 1000, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000)
            except Exception as e:
                logging.exception("Error!")

    finally:
        pipeline.stop()
    if video is not None:
        out_vid.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Captures ground plane and obstacles as polygons. Needs D4XX and T265!")
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
