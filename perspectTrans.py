import imp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import AdvLnUtils as AU
imp.reload(AU)

image  = mpimg.imread('./test_images/straight_lines2.jpg')
perspTrans = AU.warp(image)[0]
print(perspTrans.shape)
mpimg.imsave('./output_images/perspectTrans.jpg', perspTrans)
if 1:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(perspTrans)
    ax2.set_title('PerspectiveTransform', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('./output_images/perspectTrans2.jpeg', bbox_inches='tight')
    plt.show()
    plt.close()

