# Recurrent Pose Attention (RPAN)
Our Tensorflow implementation of Recurrent Pose Attention in Du et al.: "RPAN: An End-to-End Recurrent Pose-attention Network for Action Recognition in Videos".

Note that we are not associated with the original authors.

## Simple model
Our simple RPAN model in `model_simple.py` drops the parameter sharing method in Equation (2) of the paper. This is the version used in our submission for CVPR 2018 Moments in Time challenge.

## Shared model
We also attempt provide a model with the original parameter sharing scheme described. It can be found in `model_shared.py`.

## Pose Joint Maps
We provide an example of how to generate the joint maps in `data_handler.py`, see `gkern(.)` and `load_pose_map(.)`. Note that we use **Openpose format** (published as Cao et al.: "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields") throughout the project. If you are using a different pose detector, you will need to modify the code.

## Miscellaneous
Unlike the published paper, we use ResNet v2-50 to extract the convolutional cube. You can download our ResNet weights at http://cmlab.csie.ntu.edu.tw/~agethen/resnet_v2.npy .

For any feedback or questions, feel free to send a message to
> s [dot] agethen [at] gmail [dot] com.
