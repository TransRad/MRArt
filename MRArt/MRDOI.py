# LIH -  Luxembourg Institute of Health
# Author: Georgia Kanli
# Date: 05/2024

import numpy as np
import os
import fnmatch
import glob
import pydicom as dicom


# Global 
global rootPathMRD
rootPathMRD = '/path/to/raw_data/'


def modify_root_mrd(rootPathMRDnew='None'):
# Modify the global variance rootPathMRD
#
# Input:
# rootPathMRD - rootPathMRD
    global rootPathMRD
    rootPathMRD = rootPathMRDnew


def open_dicom(subject_id=None, scan_id=None, path=None): # fix it
# Open Dicom file
#
# Output:
# im -  image domain
# dim - dimensions
# par - acquisition parameters: struct
#
# Input:
# filename: Path of the mrd file - string
    dcm_files = create_files(subject_id, scan_id, path)
    for i in range(0,np.shape(dcm_files)[0]):
        temp_im = dicom.dcmread(dcm_files[i]).pixel_array
        if i == 0:
            header = dicom.filereader.dcmread(dcm_files[i])
            im = np.zeros([np.shape(dcm_files)[0],np.shape(temp_im)[0],np.shape(temp_im)[1]], dtype=complex)
            par = header
        im[i,:,:] = temp_im
        
    dim = np.shape(im)
        
    return im, dim, par


def get_kspace(im=None): # fix it
# Get kspace domain from image domain (2D or 3D)
#
# Output:
# kspace - k-space domain 
#
# Input:
# im: image domain, 2D or 3D
    kspace = np.zeros(np.shape(im), dtype=complex)
    for i in range(0,np.shape(im)[0]):
        kspace[i,:,:] = inverse_recon(im[i,:,:], extra_shift=False)
    return kspace


def create_path(subject_id=None, scan_id=None):
# Create the MRD path
#
# Output:
# full_file_name - The MRD path: string
#
# Input:
# mouseID - The mouse ID
# scanID - The scan ID
#
# Example:
# create_path(1701,20626)
    full_file = str(rootPathMRD + str(int(subject_id)) + '/' + str(int(scan_id)) )
    return full_file


def create_files(subject_id=None, scan_id=None, path=None):
# Create the SUR path
#
# Output:
# mylist - list of the DICOM files
#
# Input:
# subject_id - The subject ID
# scanID - The scan ID
    if path is None:
        path = create_path(subject_id, scan_id)  
    full_folder = str(path + '/*.dcm')
    mylist = []
    for file in glob.glob(full_folder):
        mylist.append(file)
    return mylist


def recon_corrected_kspace(corrected_kspace=None):
# reconstruction of the kspace 2D
#
# Output:
# im - 2D image domain
#
# Input:
# corrected_kspace - 2D k-space domain
    im = np.fft.ifft(corrected_kspace, axis=1)
    im = np.fft.ifft(im, axis=0)
    im = np.fft.ifftshift(im)
    return im


def inverse_recon(im=None, extra_shift=False):
# inverse reconstruction of the image domain 2D
#
# Output:
# kspace - 2D k-space domain 
#
# Input:
# im - 2D image domain
# extra_shift: if ectra extra shift is needed for the FFT
    kspace = np.fft.fftshift(im)
    kspace = np.fft.fft2(kspace)
    kspace = np.fft.fftshift(kspace)
    if extra_shift is True:
        kspace = np.fft.fftshift(kspace)
    return kspace


if __name__ == '__main__':
    pass
