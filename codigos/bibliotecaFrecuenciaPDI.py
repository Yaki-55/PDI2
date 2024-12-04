import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width/2, height/2)
    
    for i in range(0, lp_filter.shape[1]):
        for j in range(0, lp_filter.shape[0]):
            radius = max(1, math.sqrt(math.pow((i-centre[0]), 2.0)+ math.pow((j-centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1+math.pow((radius/d), (2*n)))
    
    return lp_filter

def butterworth_high_pass_filter(width, height, d, n):
    return 1 - (create_butterworth_low_pass_filter(width, height, d, n))

def create_gaussian_low_pass_filter(width, height, d, n):
    ##Future implements
    return None