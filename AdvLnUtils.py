import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imgUtils as iu
from moviepy.editor import VideoFileClip

def absSobelThresh(img, orient='x', sobel_kernel=3, thresh=(20, 255), dbg=0):
    if dbg: print('dirThresh=', orient, thresh)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ox, oy = 1,0
    if orient != 'x': ox, oy = 0,1 # (the 0, 1 at the end denotes y-direction)
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, ox, oy, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    gradBinary = np.zeros_like(scaled_sobel)
    gradBinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return gradBinary

def magThresh(img, sobel_kernel=3, mThresh=(20, 255), dbg=0):
    if dbg: print('magThresh=', mThresh)    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    magBinary = np.zeros_like(scaled_sobel)
    magBinary[(scaled_sobel >= mThresh[0]) & (scaled_sobel <= mThresh[1])] = 1
    return magBinary

def dirThresh(img, sobel_kernel=3, thresh=(0.7, np.pi/2), dbg=0):
    if dbg: print('dirThresh=', thresh)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    dirGradient = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dirBinary = np.zeros_like(dirGradient) # NOTE No np.int8 (8bit conversion)!
    dirBinary[(dirGradient >= thresh[0]) & (dirGradient <= thresh[1])] = 1
    return dirBinary

def hlsSelect(img, sel='S', thresh=(90, 255), dbg=0):
    if dbg: print('hlsSelect', sel, thresh)
    hlsSel={'H':0, 'L':1, 'S':2,}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) #float
    X = hls[:,:, hlsSel[sel.upper()]]  # Apply a threshold to the 1 of HLS channel
    binary = np.zeros_like(X)
    binary[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary

def combGradMagDir(Img, ksize=3, dbg=0):
    # Apply each of the thresholding functions
    gradx = absSobelThresh(Img, orient='x', sobel_kernel=ksize, thresh=(20, 255), dbg=dbg)
    grady = absSobelThresh(Img, orient='y', sobel_kernel=ksize, thresh=(20, 255), dbg=dbg)
    mag_binary = magThresh(Img, sobel_kernel=ksize, mThresh=(20, 255), dbg=dbg)
    dir_binary = dirThresh(Img, sobel_kernel=ksize, )#thresh=(.5, 1.)) #np.pi/1.2))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

def combGradMagDirWYmask(Img, ksize=3, dbg=0):
    # Apply each of the thresholding functions
    gradx = absSobelThresh(Img, orient='x', sobel_kernel=ksize, thresh=(20, 100), dbg=dbg)
    grady = absSobelThresh(Img, orient='y', sobel_kernel=ksize, thresh=(20, 100), dbg=dbg)
    mag_binary = magThresh(Img, sobel_kernel=ksize, mThresh=(35, 100), dbg=dbg)
    dir_binary = dirThresh(Img, sobel_kernel=ksize, thresh=(.7, 1.3)) #np.pi/1.2))
    combined = np.zeros_like(dir_binary)
    combined[ ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) & 
              ( iu.addMask_yellowMask(Img) == 1) ] = 1
    return combined

def pipeline(Img, s_thresh=(170, 255), sx_thresh=(20, 100), kernelSz=3, dbg=0):
    img = np.copy(Img)
    s_binary = hlsSelect(img, sel='S', thresh=s_thresh)
    l_layer = np.zeros_like(img)
    l_layer[:,:,1] = cv2.cvtColor(Img, cv2.COLOR_RGB2HLS)[:,:,1].astype(np.float)
    sxbinary = combGradMagDir(l_layer, ksize=kernelSz, thresh=sx_thresh, dbg=dbg)
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

def outlineRegionOfIntest(Img, vertices):
    #Outline the image defined by the polygon from 'vertices.' Beyond the outln is set to black=255
    mask = np.zeros_like(Img)
    # Setting 3 or 1 channel to fill the mask based on input Img: Color vs Gray
    if len(Img.shape) > 2:
        channelCount = Img.shape[2]  # i.e. 3 or 4 depending on your image
        ignoreMaskColor = (255,) * channelCount
    else: ignoreMaskColor = 255
    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignoreMaskColor)
    # Feturning the image only where mask pixels are nonzero
    maskedImg = cv2.bitwise_and(Img, mask)
    return maskedImg

def warp(Img):
    imgSz = (Img.shape[1], Img.shape[0])
    y_bot = Img.shape[0] #=720
    src = { 'tR': [730, 460],    'tL': [570, 460],
            'bR': [1180, y_bot], 'bL': [180,  y_bot] }
    vertices = np.array([[ src['bL'], src['tL'], src['tR'], src['bR'] ]], dtype=np.int32)
    regionOfInterest = outlineRegionOfIntest(Img, vertices)
    # src coordinates
    src = np.float32([ src['tR'], src['bR'], src['bL'], src['tL'] ], dtype=np.int32)

    dst = { 'tR': [980, 0],    'tL': [320, 0],
            'bR': [960, y_bot],'bL': [320, y_bot],}
    # Dst coordinates
    Dst = np.float32([dst['tR'], dst['bR'], dst['bL'], dst['tL'] ], dtype=np.int32)
    # perspective transform Calc
    M = cv2.getPerspectiveTransform(src, Dst)
    # the inverse matrix Calc (will be used in the last steps)
    Minv = cv2.getPerspectiveTransform(Dst, src)
    #Create waped image But keep the same size as input image
    warped = cv2.warpPerspective(regionOfInterest, M, imgSz, flags=cv2.INTER_LINEAR)  
    return warped, Minv

def combineGradientColor(Img): 
    ## Combine gradient thresholds & color space to better detect the lines
    rgbImg = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    hlsImg = hlsSelect(rgbImg, thresh=(90, 255)) 
    result = np.zeros_like(hlsImg)
    result[(combGradMagDirWYmask(rgbImg) == 1) | (hlsImg == 1) ] = 1
    return result

def onScrnCurvatureMeasrs(Img, left_curverad, right_curverad, dst_frm_center):
    CLR = (255,255,255)
    ## Print left & right Radius on each sides of the Image
    cv2.putText(Img, 'Left Radius', ( 50, 600), fontFace=5, fontScale=1.5, color=CLR, thickness=2)
    cv2.putText(Img, '{}m'.format(int(left_curverad)), (70, 650), 
                fontFace=5, fontScale=1.5, color=CLR,thickness=2)
    cv2.putText(Img, 'RightRadius', (1000, 600), fontFace=5, fontScale=1.5, color=CLR, thickness=2)
    cv2.putText(Img, '{}m'.format(int(right_curverad)), (1070, 650), 
                fontFace=5, fontScale=1.5, color=CLR, thickness=2)
    # Print distance from center on top center of the Image
    cv2.putText(Img, 'CenterOffSet', (530, 100), fontFace=5, fontScale=1.5, color=CLR, thickness=2)
    cv2.putText(Img, '{0:.2f}m'.format(dst_frm_center), (560, 160), 
                fontFace = 5, fontScale = 2, color=CLR, thickness = 2)
    return Img

def getPolynomialsCurve(Img, dbg=0):
    # Id the x & y positions of All nonzero pixels in the image
    nonzero  = Img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current  = leftx_base
    rightx_current = rightx_base
    
    margin = 100 # Set the width of the windows +/- margin
    minpix = 50  # Set minimum number of pixels found to recenter window
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds  = []
    right_lane_inds = []
    
    # Slice the image in 10 horizonally
    slices = iu.sliceImage(Img)
    window_height = np.int(slices[0].shape[0])
    
    for i in range(0, len(slices)):
        win_y_low = Img.shape[0] - (i+1)*window_height
        win_y_high = Img.shape[0] - i*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:  
            leftx_current  = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix: 
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    if dbg:
        # Generate x and y values for plotting
        ploty = np.linspace(0, Img.shape[0]-1, Img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    return left_fit, right_fit


def getLineCurvature(Img, left_fit, right_fit, dbg=0):
    ploty = np.linspace(0, Img.shape[0]-1, Img.shape[0])
    left_fitx  = left_fit[0]*ploty**2  + left_fit[1]*ploty  + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dim
    xm_per_pix = 3.7/700 # meters per pixel in x dim

    # Fit polynomials to x,y in Image space
    left_fit_cr  = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calc the new radius curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calc the center offset
    center_of_image_in_meters = (Img.shape[1] / 2) * xm_per_pix
    actual_centers_in_meters = np.mean([left_fitx, right_fitx]) * xm_per_pix
    dst_from_center = center_of_image_in_meters - actual_centers_in_meters
    
    # Transform radius of curvature is in meters
    return left_curverad, right_curverad, dst_from_center

def drawLines(origImg, warpdImg, left_fit, right_fit, Minv):
    ploty = np.linspace(0, warpdImg.shape[0]-1, warpdImg.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warpdImg).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warpdImg.shape[1], warpdImg.shape[0])) 
    
    # Combine the result w/ the original image
    result = cv2.addWeighted(origImg, 1, newwarp, 0.3, 0)
    return result

def processImg(image, dbg=0):
    undistImg = iu.undistortImage(image)
    filtrdImg = combineGradientColor(undistImg)
    warpedImg, Minv = warp(filtrdImg) #Bird Eye view
    # Get polynomial coeff fitting the curvature of the lane lines
    left_fit, right_fit = getPolynomialsCurve(warpedImg, dbg=dbg) 

    # Measure the curvature of the two lines, and get the distance from the center
    left_curvrad, right_curvrad, dst_from_center = getLineCurvature(warpedImg, left_fit, right_fit, dbg=dbg)
    if dbg: print(warpedImg.shape)
    # Draw the detected lines on the input image
    ImgWlines = drawLines(undistImg, warpedImg, left_fit, right_fit, Minv)
    #if dbg: iu.plt1Image(ImgWlines)
    # put the Curvature Measures on Screen
    ImgWlnsLbls = onScrnCurvatureMeasrs(ImgWlines, left_curvrad, right_curvrad, dst_from_center)
    return ImgWlnsLbls

def proccessVideo(inClipFnm, outClipFnm='./output_images/outPut.mp4'):
    inVclip = VideoFileClip(inClipFnm)
    outClip = inVclip.fl_image(processImg)
    outClip.write_videofile(outClipFnm, audio=False)    

if __name__ == '__main__':
    #image  = mpimg.imread('signs_vehicles_xygrad.jpg')
    #image  = mpimg.imread('bridge_shadow.jpg')
    #combin = warp(image)[0]
    #iu.plt2Images(image, combin)
    #iu.plt1Image( processImg( mpimg.imread('./test_images/straight_lines1.jpg'), 1))
    #proccessVideo("./project_video.mp4")
    #proccessVideo("./challenge_video.mp4", outClipFnm='./output_images/ChallengeOutPut.mp4')
    #proccessVideo("./harder_challenge_video.mp4", outClipFnm='./output_images/harderChallengeOutPut.mp4')
    xx = processImg( mpimg.imread('./test_images/straight_lines1.jpg'), dbg=1)
