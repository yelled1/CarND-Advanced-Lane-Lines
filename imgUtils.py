import os, cv2, pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def drawLines(origImg, warpImg, left_fit, right_fit, Minv, dbg=0):
    ploty      = np.linspace(0, warpImg.shape[0]-1, warpImg.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero  = np.zeros_like(warpImg).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left   = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right  = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warpImg.shape[1], warpImg.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(origImg, 1, newwarp, 0.3, 0)
    if dbg: plt1Image(result)
    return result


def sliceImage(Img, slices=10):
    #Returns an array of horizontal slices
    origHeight = Img.shape[0]
    slicesList = []
    for i in range(0, slices): 
        ix = int(Img.shape[0]-(origHeight/float(slices)))
        slicesList.append(Img[ix:])               # Take the bottom slide
        Img = Img = Img[:-slicesList[i].shape[0]] # From the origImage, remove this slice
    # slice_list into np array (10, 72, 1280)
    return np.asarray(slicesList)


def colorMask(hsv, low, high):
    # Return mask from HSV 
    return cv2.inRange(hsv, low, high)

def applyColorMask(hsv, Img, low,high):
    # Apply color mask to image
    mask = cv2.inRange(hsv, low, high)
    return cv2.bitwise_and(Img, Img, mask= mask)

def addMask_yellowMask(Img):
    # Convert to HSV 
    Img = cv2.cvtColor(Img, cv2.COLOR_RGB2HSV)
    yellwHsvLow  = np.array([  0, 100, 100])
    yellwHsvHigh = np.array([ 80, 255, 255])
    whiteHsvLow  = np.array([  0,   0, 160])
    whiteHsvHigh = np.array([255, 255, 255])
    # Create the mask
    yellwMask = colorMask(Img, yellwHsvLow, yellwHsvHigh)
    whiteMask = colorMask(Img, whiteHsvLow, whiteHsvHigh)
    # Combine then into one by 'or'
    return cv2.bitwise_or(yellwMask, whiteMask)

def undistortImage(Img, PklF = './camera_cal/cameraCalibPickle.p'):
    dist_pickle = pickle.load( open( PklF, "rb" ) )
    dist = dist_pickle["dist"]
    mtx  = dist_pickle["mtx"]
    return cv2.undistort(Img, mtx, dist, None, mtx)

def loadImages(dirPath):
    #return np array of images within the 'dirPath' as an np.array RGB
    return np.array([cv2.imread(dirPath + image) for image in os.listdir(dirPath)])

def plt1Image(Img):
    if len(Img.shape) < 3: 
        plt.imshow(Img, cmap='gray')
    else:     plt.imshow(Img)
    plt.show()

def plt2Images(origImg, newImg, nTitle='New Img'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    f.tight_layout()
    ax1.imshow(origImg)
    ax1.set_title('Orig image', fontsize=25)
    if len(newImg.shape) < 3:
        ax2.imshow(newImg, cmap='gray')
    else: ax2.imshow(newImg)
    ax2.set_title(nTitle, fontsize=25)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

if __name__ == '__main__':
    #image  = mpimg.imread('signs_vehicles_xygrad.jpg')
    #image  = mpimg.imread('bridge_shadow.jpg')
    image  = mpimg.imread('./test_images/straight_lines1.jpg')

