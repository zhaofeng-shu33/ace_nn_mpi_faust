# ace_nn on MPI-FAUST data
10 persons times 10 poses times 10 angles = 1000 images.

Step 1: angle picture generation. `person_pose_angle.jpg`
Step 2: image cropping. 10 npy files with 10 different poses.
For each npy file, a 3d array is stored. Its shape is [num_images, height, width].

The project is targeted to generate the hierachical structure for
10 poses.