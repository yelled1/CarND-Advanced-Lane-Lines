import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cornersUnwarp(img, nx, ny, mtx, dist):
    if img == None: raise ValueError('Hell, Img is Empty or None!')
    undist = cv2.undistort(img, mtx, dist, None, mtx)   # 1) Undistort using mtx and dist
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)     # 2) Convert to grayscale
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None) #Drawing detected corners on an image:
    # Use cv2.calibrateCamera and cv2.undistort()
    # 3) Find the chessboard corners
    # 4) If corners found: 
    if ret == True:
        img = cv2.drawChessboardCorners(undist, (nx, ny), corners, ret) # a) draw corners
        offset = 100
        imgSz  = (gray.shape[1], gray.shape[0])
        src = np.float32([ corners[0], corners[nx-1], corners[-1],  corners[-nx] ])
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
        dst = np.float32([[offset,offset],[imgSz[0]-offset,offset],
                          [imgSz[0]-offset,imgSz[1]-offset], [offset,imgSz[1]-offset],])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        M = cv2.getPerspectiveTransform(src, dst)
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        warped = cv2.warpPerspective(undist, M, imgSz)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return warped, M

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "./calibration_wide/wide_dist_pickle.p", "rb" ) )
dist_pickle = pickle.load( open( "./calibration_wide/wide_dist_test_pickle.p", 'rb' ))
m = dist_pickle["mtx"]
d = dist_pickle["dist"]

x,y = 8,6 # the number of inside corners
Img = cv2.imread('./calibration_wide/test_image2.jpg') # Returns None if file NOT found
top_down, perspective_M = cornersUnwarp(Img, x, y, m, d)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(Img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
# Needs Get Into this a bit more from 'offset=100' onward
