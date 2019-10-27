# FusionNet
(Ongoing) Aiming to solve 3D multi-scale object detection problems, by generating dynamic anchors corresponding to 2D classification.
## Pipeline
<brInput 2D images, using 2D feature extractor generating multi-scale object bounding box, estimated depth, estimated rotation, object classification.\>
<brInput 3D PointCloud. Using SparseConvNet to generate 3D features.\>
<brCrop 3D features, within the estimated area based on the guidance of 2D result, augmenting 3D features with correspondence 2D features.\>
<brRefine the 3D BBox with 3D features.\>

a framework solve 2D and 3D features fusion challenges
## Some helpful sources
1 modified the 2D feature extractor based on the following:
6D-VNet: End-to-end 6DoF Vehicle Pose Estimation from Monocular RGB Images | Github: https://github.com/stevenwudi/6DVNET
