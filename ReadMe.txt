Download the 11K hands datasets from here: https://github.com/mahmoudnafifi/11K-Hands and store it in a directory named Hands. I recommend using Pycharm but you may need to install the modules mentioned below manually after going to Files -> Settings.
The keypoints and descriptors are stored in separate files, as is the order of files as they appear during training. The reason for separating keypoints and descriptors is to reduce the size of file read and writes which caused memory errors during training, and they can be retrieved and used independently without conversion quite easily. For example, I have an implementation of plotting the keypoints on an image. In this visualisation, we don’t need the descriptors as only the coordinates are being marked. Similar to the LBP structure, the name of file in the order file corresponds to the keypoint and descriptor values in the respective files at corresponding line numbers. 

To run it, use the following:
Python 3.6
numpy 1.17.2
opencv-python 3.3.0.10

Use the Pycharm command line for running as it creates venv of its own it would be more robust to platform and compatibility errors.
run the sift_all.py with given file structure and optional command line argument as the image id (default is 0000000 which runs for entire dataset)
It will store features in Output directory in respective files.
run the compare_and_rank.py with test file id as argument and one optional argument for the number of matches to be displayed (default all matches displayed)

run find_and_show.py with command line argument name of image to visualize that image for keypoints.
