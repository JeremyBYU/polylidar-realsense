# Script was used to visualize and debug point clouds previously
txt2las.exe allpoints_cameraframe.txt -set_classification 1
txt2las.exe polygon_0_cameraframe.txt -set_classification 2
txt2las.exe polygon_1_cameraframe.txt -set_classification 3

lasmerge.exe -i *.las -o out.las 