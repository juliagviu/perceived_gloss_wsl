from mimetypes import init
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from hsluv import hsluv_to_rgb

# Tonemapper taken from [Serrano 21] 
def tonemapping(img, tm_fstop=1.5):
    tm_gamma = 2.2
    #tm_fstop = 1.5 # default parameter in Serrano21
    offset = 0.055

    inv_gamma = 1.0 / tm_gamma
    exposure = np.power(2, tm_fstop)

    img[..., :3] = np.where(img[..., :3] <= 0.0, 12.92 * img[..., :3], (1 + offset) * np.power(img[..., :3] * exposure, inv_gamma) - offset)
    # img[..., :3] = np.where(img[..., :3] <= 0.0031308, 12.92 * img[..., :3], (1 + offset) * np.power(img[..., :3] * exposure, inv_gamma) - offset)
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
    return img

def color_hsluv_gauss():

    H = random.uniform(0, 360)
    S = np.random.beta(2, 1, 1)[0] * 100
    L = random.gauss(50, 16)
    
    if L < 0:
        L = 0
    elif L > 100:
        L = 100

    return hsluv_to_rgb([H, S, L])




