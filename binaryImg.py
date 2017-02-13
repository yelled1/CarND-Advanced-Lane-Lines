import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from AdvLnUtils import combGradMagDir, pipeline

image  = mpimg.imread('./test_images/test1.jpg')
#combin = combGradMagDir(image, 5, )
#combin =pipeline(image)
combin = pipeline(image, s_thresh=(180, 200), sx_thresh=(40, 100), kernelSz=3, dbg=0)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combin, cmap='gray')
ax2.set_title('comb Grad', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
