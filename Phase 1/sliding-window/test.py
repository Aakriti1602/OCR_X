from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
# image reading
image = cv2.imread("Img_1.jpg")
(winW, winH) = (128, 128)
i = image.copy()