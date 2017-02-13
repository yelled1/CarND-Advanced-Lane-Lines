import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
#%matplotlib qt

DIR = './camera_cal/'
PklF = DIR+"cameraCalibPickle.p"
x,y = 9,6

# Make a list of calibration images
images = glob.glob('./test_images/test*.jpg')
for idx, fname in enumerate(images):
    print(idx,":", fname)

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.undistort(img, mtx, dist, None, mtx) #dst = undistorted destination img
    cv2.imwrite(DIR+'test_undist1.jpg',dst) # distorted img, (camera) mtx, dist coeff, 


    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (x,y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (x,y), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(5) #500)
cv2.destroyAllWindows()

#U now have `objpoints` and `imgpoints` needed for camera calibation.  
#Run the below to calibrate, calc distortion coefficients, & test undistortion on an image!

#get_ipython().magic('matplotlib inline')
# Test undistortion on an image
img = cv2.imread(DIR+'test_image1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given objectpoints=[[[0,0,0]..[7,5,0]],..] and imagepoints=[corners .. corners]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
#dist = distortion coeffs # (camera) mtx=tranform 3D obj to 2d imgpoints
#position of the camera in the world: rotation & traslation vecs


# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open(PklF, "wb") )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
