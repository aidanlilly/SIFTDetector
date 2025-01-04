# Real-Time Object Detection Using SIFT and Homography

This project demonstrates real-time object detection using **SIFT (Scale-Invariant Feature Transform)** and **Homography**. The script detects whether a specific object (such as a box or logo) appears in a video feed or webcam input. The object is identified by matching keypoints between a query image (the object) and frames from the video feed, followed by calculating a homography to determine if the object is present.

Below is a demonstration of the project in action:

![Object Detection Demo](demo_image.png)

*Image showcasing the matching process and detection of the object in a video frame.*

## Features
- **Keypoint Detection with SIFT**: Detects and extracts keypoints and descriptors from both the query image and video frames.
- **Homography Estimation**: Uses RANSAC to calculate the homography matrix, helping to align the query image with detected objects in the video frames.
- **Real-Time Detection**: Works with live webcam or prerecorded video files, detecting objects in each frame.
- **Visual Feedback**: Displays the detection result and keypoint matches on the video feed.
- **Object Detection Status**: Provides real-time feedback on whether the object is detected in the current frame.

## Dependencies

To ensure the project runs smoothly, the following dependencies are required:

- **Python 3.8+**: The script is compatible with Python version 3.8 or higher.
- **OpenCV**: For image and video processing, including feature detection and homography estimation.
- **NumPy**: For numerical operations related to arrays and matrices.

Install these dependencies using pip:
```bash
pip install opencv-python numpy
```
## Usage

Change the path of img1 in line 6 of **main.py** to the path to the intended query image. Next change the path in line 21 to the path to the intended video to detect the object in, including the webcam. To run the project, simply execute:
```bash
python main.py
```
