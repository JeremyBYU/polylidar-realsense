import logging
import sys
import argparse
from os import path
import time
import uuid
import itertools


import numpy as np
import yaml
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import open3d as o3d
import pandas as pd

from polylidar import MatrixDouble, MatrixFloat, extract_point_cloud_from_float_depth, Polylidar3D
from fastgac import GaussianAccumulatorS2Beta, IcoCharts

from polylidar.polylidarutil.plane_filtering import filter_planes_and_holes
from surfacedetector.utility.helper import (plot_planes_and_obstacles, create_projection_matrix,
                                            get_intrinsics, load_setting_file, save_dict_to_json)


from surfacedetector.utility.helper_mesh import create_meshes_cuda, create_meshes_cuda_with_o3d, create_meshes
from surfacedetector.utility.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh
from surfacedetector.utility.helper_tracking import get_pose_matrix, cycle_pose_frames, callback_pose

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


def create_pipeline(config: dict):
    """Sets up the pipeline to extract depth and rgb frames

    Arguments:
        config {dict} -- A dictionary mapping for configuration. see default.yaml

    Returns:
        tuple -- pipeline, process modules, filters, t265 device (optional)
    """
    # Create pipeline and config for D4XX,L5XX
    pipeline = rs.pipeline()
    rs_config = rs.config()

    # IF t265 is enabled, need to handle seperately
    t265_dev = None
    t265_sensor = None
    t265_pipeline = rs.pipeline()
    t265_config = rs.config()

    if config['playback']['enabled']:
        # Load recorded bag file
        rs.config.enable_device_from_file(
            rs_config, config['playback']['file'], config['playback'].get('repeat', False))

        # This code is only activated if the user points to a T265 Recorded Bag File
        if config['tracking']['enabled']:
            rs.config.enable_device_from_file(
                t265_config, config['tracking']['playback']['file'], config['playback'].get('repeat', False))

            t265_config.enable_stream(rs.stream.pose)
            t265_pipeline.start(t265_config)
            profile_temp = t265_pipeline.get_active_profile()
            t265_dev = profile_temp.get_device()
            t265_playback = t265_dev.as_playback()
            t265_playback.set_real_time(False)

    else:
        # Ensure device is connected
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            logging.error("No connected Intel Realsense Device!")
            sys.exit(1)

        if config['advanced']:
            logging.info("Attempting to enter advanced mode and upload JSON settings file")
            load_setting_file(ctx, devices, config['advanced'])

        # Cycle through connected devices and print them
        for dev in devices:
            dev_name = dev.get_info(rs.camera_info.name)
            print("Found {}".format(dev_name))
            if "Intel RealSense D4" in dev_name:
                pass
            elif "Intel RealSense T265" in dev_name:
                t265_dev = dev
            elif "Intel RealSense L515" in dev_name:
                pass

        if config['tracking']['enabled']:
            if len(devices) != 2:
                logging.error("Need 2 connected Intel Realsense Devices!")
                sys.exit(1)
            if t265_dev is None:
                logging.error("Need Intel Realsense T265 Device!")
                sys.exit(1)

            if t265_dev:
                # Unable to open as a pipeline, must use sensors
                t265_sensor = t265_dev.query_sensors()[0]
                profiles = t265_sensor.get_stream_profiles()
                pose_profile = [profile for profile in profiles if profile.stream_name() == 'Pose'][0]
                t265_sensor.open(pose_profile)
                t265_sensor.start(callback_pose)
                logging.info("Started streaming Pose")

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

    depth_sensor = profile.get_device().first_depth_sensor()
    color_sensor = profile.get_device().first_color_sensor()

    depth_scale = depth_sensor.get_depth_scale()
    # depth_sensor.set_option(rs.option.global_time_enabled, 1.0)
    # color_sensor.set_option(rs.option.global_time_enabled, 1.0)

    if config['playback']['enabled']:
        dev = profile.get_device()
        playback = dev.as_playback()
        playback.set_real_time(False)

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

    intrinsics = get_intrinsics(pipeline, rs.stream.color)
    proj_mat = create_projection_matrix(intrinsics)

    sensor_meta = dict(depth_scale=depth_scale)
    config['sensor_meta'] = sensor_meta
    
    # Note that sensor must be saved so that it is not garbage collected
    t265_device = dict(pipeline=t265_pipeline, sensor=t265_sensor)

    return pipeline, process_modules, filters, proj_mat, t265_device


def get_frames(pipeline, t265_pipeline, process_modules, filters, config):
    """Extracts frames from intel real sense pipline
    Applies filters and extracts point cloud
    Arguments:
        pipeline {rs::pipeline} -- RealSense Pipeline
        t265_pipeline {rs::pipeline} -- Optional T265 Pipeline, can be None
        process_modules {tuple} -- align, depth_to_disparity, disparity_to_depth, decimate
        filters {list[rs::filters]} -- List of filters to apply

    Returns:
        (rgb_image, depth_image, ndarray, meta) -- RGB Image, Depth Image (colorized), numpy points cloud, meta information
    """
    if config['playback']['enabled']:
        frames = pipeline.wait_for_frames(timeout_ms=30)
        ts_depth = frames.get_timestamp()
        if t265_pipeline is not None:
            ts_t265 = cycle_pose_frames(t265_pipeline, ts_depth)

    else:
        success, frames = pipeline.try_wait_for_frames(timeout_ms=5)
        if not success:
            return None, None, None
    # Get all the standard process modules
    (align, depth_to_disparity, disparity_to_depth, decimate) = process_modules

    # Align depth frame with color frame.  For later overlap
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    ts_domain = depth_frame.get_frame_timestamp_domain()
    ts = depth_frame.get_timestamp()

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
    # dropped color frames will be reused, causing DOUBLE writes of polygons on the same
    # image buffer. Create a copy so this doesn't occur
    color_image = np.copy(np.asanyarray(color_frame.get_data()))
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_ts = depth_frame.get_timestamp()

    threshold = config['filters'].get('threshold')
    if threshold is not None and threshold['active']:
        mask = depth_image[:, :] > int(threshold['distance'] * (1 / config['sensor_meta']['depth_scale']))
        depth_image[mask] = 0

    meta = dict(h=h, w=w, intrinsics=d_intrinsics_matrix, ts=depth_ts)

    return color_image, depth_image, meta


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
    depth_scale = 1 / config['sensor_meta']['depth_scale']
    depth_image = np.divide(depth_image, depth_scale, dtype=np.float32)
    points = extract_point_cloud_from_float_depth(MatrixFloat(
        depth_image), MatrixDouble(intrinsics), IDENTITY_MAT, stride=stride)
    new_shape = (int(depth_image.shape[0] / stride), int(depth_image.shape[1] / stride), 3)
    opc = np.asarray(points).reshape(new_shape)  # organized point cloud (will have NaNs!)
    # print("OPC Shape: ", new_shape)
    # 2. Create Mesh and Smooth (all in one)
    # mesh, o3d_mesh, timings = create_meshes_cuda_with_o3d(opc, **config['mesh']['filter'])
    if config['mesh'].get('use_cuda'):
        mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
    else:
        mesh, timings = create_meshes(opc, **config['mesh']['filter'])
    alg_timings.update(timings)

    # 3. Estimate Dominate Plane Normals
    fga = config['fastgac']
    avg_peaks, _, _, _, timings = extract_all_dominant_plane_normals(
        mesh, ga_=ll_objects['ga'], ico_chart_=ll_objects['ico'], **fga)
    alg_timings.update(timings)
    # print(avg_peaks)

    # 4. Extract Polygons from mesh
    planes, obstacles, timings = extract_planes_and_polygons_from_mesh(mesh, avg_peaks, pl_=ll_objects['pl'],
                                                                       filter_polygons=True, optimized=True,
                                                                       postprocess=config['polygon']['postprocess'])
    alg_timings.update(timings)

    # Uncomment to save raw data if desired
    # np.save('L515_Depth.npy', depth_image)
    # np.save('L515_OPC.npy', opc)
    # save_dict_to_json('L515_meta.json', dict(depth_scale=depth_scale, intrinsics=intrinsics.tolist(),
    #                                          mesh=config['mesh'], fastgac=config['fastgac'],
    #                                          polylidar=config['polylidar'], postprocess=config['polygon']['postprocess']))

    # return planes, obstacles, alg_timings, o3d_mesh
    return planes, obstacles, alg_timings


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


def colorize_depth(depth_image, config, vmin=0.2, vmax=2.5, bgr=True):
    depth_scale = config['sensor_meta']['depth_scale']
    depth_image_cv = cv2.resize(depth_image, (config['color']['width'], config['color']['height']))
    normalized_depth = Normalize(vmin=int(vmin / depth_scale), vmax=int(vmax / depth_scale), clip=True)(depth_image_cv)
    depth_image_cv = (plt.cm.viridis(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
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
    pipeline, process_modules, filters, proj_mat, t265_device = create_pipeline(config)
    t265_pipeline = t265_device['pipeline']
    logging.info("Pipeline Created")

    # Long lived objects. These are the object that hold all the algorithms for surface exraction.
    # They need to be long lived (objects) because they hold state (thread scheduler, image datastructures, etc.)
    ll_objects = dict()
    ll_objects['pl'] = Polylidar3D(**config['polylidar'])
    ll_objects['ga'] = GaussianAccumulatorS2Beta(level=config['fastgac']['level'])
    ll_objects['ico'] = IcoCharts(level=config['fastgac']['level'])

    if video:
        frame_width = config['color']['width'] * 2
        frame_height = config['color']['height']
        out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    all_records = []
    counter = 0
    try:
        while True:
            t00 = time.perf_counter()
            try:
                color_image, depth_image, meta = get_frames(pipeline, t265_pipeline, process_modules, filters, config)
            except RuntimeError:
                # This only gets thrown when in playback mode from a recoded file when frames "run out"
                logging.info("Out of frames")
                break
            t0 = time.perf_counter()
            if color_image is None or not valid_frames(color_image, depth_image, **config['polygon']['frameskip']):
                logging.debug("Invalid Frames")
                continue
            t1 = time.perf_counter()
            counter += 1
            # if counter < 10:
            #     continue

            try:
                # Get 6DOF Pose at appropriate timestamp
                if config['tracking']['enabled']:
                    euler_t265 = get_pose_matrix(meta['ts'])
                    logging.info('euler_t265: %r', euler_t265)

                if config['show_polygon']:
                    # planes, obstacles, timings, o3d_mesh = get_polygon(depth_image, config, ll_objects, **meta)
                    planes, obstacles, timings = get_polygon(depth_image, config, ll_objects, **meta)
                    timings['t_get_frames'] = (t0 - t00) * 1000
                    timings['t_check_frames'] = (t1 - t0) * 1000
                    all_records.append(timings)

                    # Plot polygon in rgb frame
                    plot_planes_and_obstacles(planes, obstacles, proj_mat, None, color_image, config)

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
                        pass
                    to_save_frames = config['save'].get('frames')
                    if config['playback']['enabled'] and to_save_frames is not None and counter in to_save_frames:
                        logging.info("Saving Picture: {}".format(counter))
                        cv2.imwrite(path.join(PICS_DIR, "{}_color.jpg".format(counter)), color_image_cv)
                        cv2.imwrite(path.join(PICS_DIR, "{}_stack.jpg".format(counter)), images)

                logging.info(f"Frame %d; Get Frames: %.2f; Check Valid Frame: %.2f; Laplacian: %.2f; Bilateral: %.2f; Mesh: %.2f; FastGA: %.2f; Plane/Poly: %.2f; Filtering: %.2f",
                             counter, timings['t_get_frames'], timings['t_check_frames'], timings['t_laplacian'], timings['t_bilateral'], timings['t_mesh'], timings['t_fastga_total'],
                             timings['t_polylidar_planepoly'], timings['t_polylidar_filter'])
            except Exception as e:
                logging.exception("Error!")
    finally:
        pipeline.stop()
    if video is not None:
        out_vid.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame.from_records(all_records)
    print(df.mean())
    if config['save'].get('timings') != "":
        df.to_csv(config['save'].get('timings', 'data/timings.csv'))


def main():
    parser = argparse.ArgumentParser(description="Captures ground plane and obstacles as polygons")
    parser.add_argument('-c', '--config', help="Configuration file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-v', '--video', help="Video file save path", default=None)
    args = parser.parse_args()
    with open(args.config) as file:
        try:
            config = yaml.safe_load(file)
            # Run capture loop
            capture(config, args.video)
        except yaml.YAMLError as exc:
            logging.exception("Error parsing yaml")


if __name__ == "__main__":

    main()
