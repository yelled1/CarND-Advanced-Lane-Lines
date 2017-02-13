import pickle, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved objpoints and imgpoints
DIR = './camera_cal/'
PklF = DIR+"cameraCalibPickle.p"
x,y = 9,6

dist_pickle = pickle.load( open( PklF, "rb" ) )
dist = dist_pickle["dist"]
mtx = dist_pickle["mtx"]

images = glob.glob('./test_images/test*.jpg')[:]
for idx, fnm in enumerate(images):
    print(idx, fnm)
    Img = cv2.imread(fnm) #DIR+'test_image1.jpg')
    undistort = cv2.undistort(Img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(Img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistort)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
