import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the query image (the box) and resize it to a fixed width while maintaining the aspect ratio
img1 = cv.imread('deoderant.jpeg', cv.IMREAD_GRAYSCALE)  # Query image
fixed_width = 600  # Set a fixed width for the image
aspect_ratio = img1.shape[1] / img1.shape[0]  # Maintain the aspect ratio
new_height = int(fixed_width / aspect_ratio)

# Resize the query image
img1 = cv.resize(img1, (fixed_width, new_height))

# Initiate SIFT detector with a limited number of keypoints to avoid too many matches
sift = cv.SIFT_create(nfeatures=500)  # Limit keypoints to 500 for efficiency

# Find the keypoints and descriptors with SIFT in the query image
kp1, des1 = sift.detectAndCompute(img1, None)
des1 = np.float32(des1)  # Ensure descriptors are in float32 type

# Start capturing from the webcam
cap = cv.VideoCapture('deovid.mp4')

while True:
    # Read a frame from the webcam
    ret, img2 = cap.read()
    if not ret:
        break  # If frame is not successfully read, exit the loop

    # Convert the webcam image to grayscale
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors in the webcam image
    kp2, des2 = sift.detectAndCompute(gray, None)
    des2 = np.float32(des2)  # Ensure descriptors are in float32 type

    # BFMatcher with knn matching (k=2)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches with a stricter ratio threshold
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    if len(good_matches) > 10:  # We need at least 10 good matches for a reliable transformation
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use RANSAC to find the homography (transformation matrix)
        homography, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

        # Mask is a binary array that indicates which matches are inliers (True) or outliers (False)
        matches_mask = mask.ravel().tolist()

        # Filter good matches based on inliers from RANSAC
        good_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i] == 1]

        # If enough good matches are found, indicate that the box is detected
        if len(good_matches) > 10:
            cv.putText(img2, "Yes, object in image", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(img2, "No object in image", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv.putText(img2, "No object in image", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the good matches
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the result
    cv.imshow("Box Detection", img3)

    # Exit the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv.destroyAllWindows()
