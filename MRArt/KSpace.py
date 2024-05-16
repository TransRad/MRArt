# LIH -  Luxembourg Institute of Health
# Author: Georgia Kanli
# Date: 05/2024

import numpy as np
from MRArt import MRDOI
import random
import scipy
from numpy.fft import ifft2, ifftshift
from skimage.filters import window  


def add_motion_artifacts(ks2d=None, events=200, ratio=0.25, phase_direction=1, lines=None, motion_type=None):
# Introduce motion artifact in 2D k-space
#
# Output:
# motion_im2d - 2D kspace with noise
# motion_ks2d - 2D reconstructed image with noise
# lines - the numbers of the modified lines/columns
#
# Input:
# ks2d - 2D kspace
# events - how many events happened (find random the lines/column)
# ratio - ratio of kspace that shift [0 1]
# phase_direction - direction (1: horizontal else: vertical)
# lines: specific the numbers of the lines/columns that need to modify (if need it), when value is None the random lines/columns are selected
# motion_type - 1: rotation, 2: shift, None: select randomly 
    ks2d_copy = ks2d.copy()
    
    ks_new = ks2d_copy
    # specific lines/column 
    if lines is None:
        lines = np.zeros(events, dtype=int)
        for i in range(0, events):
            lines[i] = int(random.uniform(0, int(np.shape(ks2d_copy)[0])))
    
    # introduce motion artifact for many averages
    for ii in lines:  # int(step)
        motion_ratio = int(random.uniform(-20, 20))
        ks_motion = motion_im(ks2d, motion_ratio, motion_type)
        if phase_direction == 1:
            ks_new[:, ii] = ks2d_copy[:, ii] * (1-ratio) + ks_motion[:, ii] * ratio 
        else:
            ks_new[ii, :] = ks2d_copy[ii, :] * (1-ratio) + ks_motion[ii, :] * ratio 

    # Reconstructed image
    noisy_im = MRDOI.recon_corrected_kspace(ks_new)

    motion_im2d = noisy_im
    motion_ks2d = ks_new

    return motion_im2d, motion_ks2d, lines


def motion_im(ks2d, ratio, motion_type=None):
# Create motion artifact in 2D k-space in only one average
#
# Output:
# motion_ks2d - 2D reconstructed image with noise
#
# Input:
# ks2d - 2D kspace
# ratio - ratio of kspace that shift [0 1]
# motion_type - 1: rotation, 2: shift, None: select randomly 
    if motion_type is None:
        motion_type = int(random.uniform(0, 2))
    ks_motion = np.zeros(np.shape(ks2d), dtype=complex)   
    if motion_type == 1:
        pivot = np.around(np.shape(ks2d))
        pivot = pivot/2
        pivot = pivot.astype(int)
        ks_motion = rotateImage(ks2d, angle=ratio, pivot=pivot)
    else:
        axis = int(random.uniform(0, 2))
        ks_motion = shift_img_along_axis(ks2d, axis=axis, shift=ratio)
    
    return ks_motion


def rotateImage(ks2d, angle=10, pivot=[0,0]):
# Create rotation motion in whole image domain. 
#
# Output:
# motion_ks2d - 2D reconstructed image with noise
#
# Input:
# ks2d - 2D kspace
# angle - angle of the rotation
# pivot - center of the rotation
    img = ifftshift(ifft2(ifftshift(ks2d)))    # image domain
    # rotation
    im_motion = np.zeros(np.shape(img), dtype=complex)   
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = scipy.ndimage.rotate(imgP, angle, reshape=False, order= 0)
    im_motion = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    kspace = ifftshift(ifft2(ifftshift(im_motion))) # kspace domain
    return kspace


def shift_img_along_axis(ks2d, axis=0, shift=10, constant_values=0):
# Create shift motion in whole image domain. 
#
# Output:
# motion_ks2d - 2D reconstructed image with noise
#
# Input:
# ks2d - 2D kspace
# angle - angle of the rotation
# pivot - center of the rotation    
    img = ifftshift(ifft2(ifftshift(ks2d))) # image domain
    # shift
    intshift = int(shift)
    remain0 = abs(shift - int(shift))
    remain1 = 1-remain0 
    npad = int(np.ceil(abs(shift))) # ceil relative to 0. ( 0.5=> 1 and -0.5=> -1 )
    pad_arg = [(0,0)]*img.ndim
    pad_arg[axis] = (npad,npad)
    bigger_image = np.pad( img, pad_arg, 'constant', constant_values=constant_values)     
    part1 = remain1*bigger_image.take(np.arange(npad+intshift, npad+intshift+img.shape[axis]) ,axis)
    if remain0==0:
        shifted = part1
    else:
        if shift>0:
            part0 = remain0*bigger_image.take(np.arange(npad+intshift+1, npad+intshift+1+img.shape[axis]) ,axis) 
        else:
            part0 = remain0*bigger_image.take(np.arange(npad+intshift-1, npad+intshift-1+img.shape[axis]) ,axis) 

        shifted = part0 + part1        
    kspace = ifftshift(ifft2(ifftshift(shifted))) # kspace domain
    return kspace


def random_motion_level():   
# Extract random motion level. 
#
# Output:
# events - how many events happened (find random the lines/column)
# ratio - ratio of kspace that shift [0 1]
# phase_direction - direction (1: horizontal else: vertical)                                                                                           
    min_events = 20
    max_events = 100
    events = int(random.uniform(min_events, max_events))
    min_average = 1
    max_average = 3
    ratio = 1/int(random.uniform(min_average, max_average))
    phase_direction = random.randint(0, 1)
    return events, ratio, phase_direction


def add_gaussian_noise_artifacts(ks2d=None, mean=0, sigma_level=0.5):
# Add noise artifacts in the kspace
#
# Output:
# noisy_im2d - 2D kspace with noise
# noisy_ks2d - 2D reconstructed image with noise
#
# Input:
# ks2d - 2D kspace
# mean - Mean (“centre”) of the distribution.
# sigma_level - Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    sigma_level = sigma_level*1.5
    ks2d_copy = ks2d.copy()
    noisy_im2d = np.zeros(np.shape(ks2d_copy))
    noisy_ks2d = np.zeros(np.shape(ks2d_copy), dtype=complex)

    sz = np.shape(ks2d_copy)
    point = np.abs(ks2d_copy.mean())*100
    sigma = point*sigma_level
    gauss = np.random.normal(mean, sigma, sz)
    gauss = gauss.reshape(sz)
    noise = np.ones(np.shape(ks2d_copy), dtype=complex) * gauss
    ks_new = ks2d_copy + noise

    # Noisy reconstructed image
    noisy_im = MRDOI.recon_corrected_kspace(ks_new)

    noisy_im2d = noisy_im
    noisy_ks2d = ks_new

    return noisy_im2d, noisy_ks2d


def random_gaussian_noise_level():
# Extract random gaussian noise level 
#
# Output:
# mean - 0
# sigma_level - [0.2 0.3]
    min_rand = 0.2
    max_rand = 0.3
    mean = 0 # random.uniform(min_rand, max_rand)
    min_rand = 0.2
    max_rand = 0.3
    sigma_level = random.uniform(min_rand, max_rand)
    return mean, sigma_level


def add_blur_by_low_pass_filter_artifacts(ks2d=None, radius=0.5, x=0.1):
# Implement low pass filter in the kspace (# We don't need it now)
#
# Output:
# lpass_im2d - 2D kspace with low pass filter
# lpass_ks2d - 2D reconstructed image with low pass filter
# mask - 2D mask of the filter
#
# Input:
# ks2d - 2D kspace
# radius - the radius of the filter
# x - 
    radius = (np.exp((1-radius-0.1))-0.7)/1.2 #(1-radius)+0.2
    ks2d_copy = ks2d.copy()
    lpass_im2d = np.zeros(np.shape(ks2d_copy))
    lpass_ks2d = np.zeros(np.shape(ks2d_copy), dtype=complex)

    r_x = int(radius*np.shape(ks2d_copy)[0])
    r_y = int(radius*np.shape(ks2d_copy)[1])
    b1 = np.zeros(np.shape(ks2d_copy))
    b2 = window('hann', [r_x, r_y])
    mask = add_matrices_diff_size(b1,b2)+x
    
    ks2d_copy = ks2d_copy*mask

    # Noisy reconstructed image
    noisy_im = MRDOI.recon_corrected_kspace(ks2d_copy)

    lpass_im2d = noisy_im
    lpass_ks2d = ks2d_copy

    return lpass_im2d, lpass_ks2d, mask


def add_matrices_diff_size(b1=None,b2=None):
# Add matrices with different size
#
# Output:
# b1 - the final metric 
#
# Input:
# b1 - metric
# b2 - metric
    pos_v, pos_h = int(np.shape(b1)[0]/2-np.shape(b2)[0]/2), int(np.shape(b1)[1]/2-np.shape(b2)[1]/2)  # offset
    v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
    h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))
    v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
    h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))
    b1[v_range1, h_range1] += b2[v_range2, h_range2]
    return b1


def random_blur_by_low_pass_level():
# Extract random low pass filter level 
#
# Output:
# radius - the radius of the filter [0 1]
# x - 
    min_rand = 0.3
    max_rand = 0.9
    radius = random.uniform(min_rand, max_rand)
    min_rand = 0
    max_rand = 0.5  
    x = random.uniform(min_rand, max_rand)
    return radius, x



