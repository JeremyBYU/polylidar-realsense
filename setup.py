from setuptools import setup, find_packages
setup(
    name="surfacedetector",
    version="1.0.0",
    packages=['surfacedetector'],
    scripts=[],

    install_requires=['numpy', 'pyrealsense2', 'pyyaml', 'scipy>=1.4.0', 'pandas', 'open3d', 'fastgac', 'polylidar'],

    # metadata to display on PyPI
    author="Jeremy Castagno",
    author_email="jdcasta@umich.edu",
    description="Polylidar3D and Realsense",
    license="MIT",
    keywords="intel realsense point cloud polygon",
    url="https://github.com/JeremyBYU/polylidar-realsense",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/JeremyBYU/polylidar-realsense/issues",
    }
)