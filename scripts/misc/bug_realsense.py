import pyrealsense2 as rs
# rs.log_to_file(rs.log_severity.debug, 'lrs_nobug.log' )
import sys

ctx = rs.context()
devices = ctx.query_devices()
# Uncomment below to see eventual segfault with L515, no error with D4XX
dev = devices[0] # <-------- This is the specifc line that causes an error  
# dev_name = dev.get_info(rs.camera_info.name)
# print("Found {}".format(dev_name))


# Configure stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
# This will start a thread that will eventually segfault with L515, no error with D4XX
pipeline.start(config)
count = 0

while True:
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        print(frames)
    except:
        print("Frames didn't arrive in time")
        count += 1
        if count > 2:
            sys.exit()