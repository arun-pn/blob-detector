# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 16:10:27 2025

@author: Arun
"""

from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io
from skimage import measure
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt



def blobs(filename):
    '''

    Parameters
    ----------
    filename : string
        Name of the image to be analysed.

    Returns
    -------
    tuple/list
        List containing number of blobs detected in the image and 
        corresponding intensities, reported as mean intensity, 
        overall intensity, percentage intensity, background intensity.

    '''
    fullpath = filename # name of the image file
    # fullpath = path_cwd+'\\'+filename # provide also the path (a good practice)
    
    image = io.imread(fullpath) # read an image
    if image.shape[2] == 3: # check if the image is RGB (three channel)
        image_gray = rgb2gray(image) # convert RGB to grayscale
    elif image.shape[2] == 4: # check if the image is RGBA (four channel)
        image_gray = rgb2gray(image[:, :, :3]) # convert RGBA to grayscale
    
    # Blob detection using Laplacian of Gaussian method
    blobs_log = blob_log(image_gray, min_sigma = 2, max_sigma=30, num_sigma=10, threshold=0.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    
    # Blob detection using Difference of Gaussian method
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
    # Blob detection using Determinant of Hessian method
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)
    
    # Plotting
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    def calculate_total_intensity(image_gray, blobs):
        '''

        Parameters
        ----------
        image_gray : ndarray
            A 2D array of shape (H, W) representing a grayscale image, where 
            H is the height (number of rows) and W is the width 
            (number of columns), and each element gives the intensity at 
            that pixel.
        blobs : ndarray
            Detected blobs, represented as a 2d array with each row 
            representing 2 coordinate values for a 2D image, plus 
            the sigma(s) (standard deviation of the kernel) used.

        Returns
        -------
        total_intensity : float
            Total intensity of all blobs detected in the given image.
            The total intensity is calculated as the sum of intensities of
            individual blobs.
        num_blobs : int
            Number of blobs detected in the given image.

        '''
        total_intensity = 0
        num_blobs = len(blobs)
        for blob in blobs:
            y, x, r = blob
            # Create a mask for each blob
            mask = np.zeros_like(image_gray, dtype=bool)
            rr, cc = np.ogrid[:image_gray.shape[0], :image_gray.shape[1]]
            mask = ((rr - y)**2 + (cc - x)**2 <= r**2)
            # Calculate the sum of pixel intensities inside the mask (blob)
            blob_intensity = np.sum(image_gray[mask])
            total_intensity += blob_intensity
        return total_intensity, num_blobs
    
    # Function to create a mask for the blobs
    # This is required to mask the blobs and extract background intensity
    def create_blob_mask(blobs, image_shape):
        '''

        Parameters
        ----------
        blobs : ndarray
            Detected blobs, represented as a 2d array with each row 
            representing 2 coordinate values for a 2D image, plus 
            the sigma(s) (standard deviation of the kernel) used.
        image_shape : tuple
            Shape of the gray scale image.

        Returns
        -------
        mask : ndarray
            A 2D Boolean array of shape (H, W), where True indicates pixels 
            inside one or more detected blobs, and False elsewhere.

        '''
        mask = np.zeros(image_shape, dtype=bool)
        for blob in blobs:
            y, x, r = blob
            rr, cc = np.ogrid[:image_shape[0], :image_shape[1]]
            blob_area = (rr - y) ** 2 + (cc - x) ** 2 <= r ** 2
            mask[blob_area] = True
        return mask
    
    # Function to calculate background intensity
    def calculate_background_intensity(image_gray, blobs):
        '''

        Parameters
        ----------
        image_gray : ndarray
            A 2D array of shape (H, W) representing a grayscale image, where 
            H is the height (number of rows) and W is the width 
            (number of columns), and each element gives the intensity at 
            that pixel.
        blobs : ndarray
            Detected blobs, represented as a 2d array with each row 
            representing 2 coordinate values for a 2D image, plus 
            the sigma(s) (standard deviation of the kernel) used.

        Returns
        -------
        background_intensity : float
            Background intensity of the image.
        background_mask : ndarray
            A 2D Boolean array of shape (H, W), where True indicates pixels 
            outside one or more detected blobs, and False elsewhere.

        '''
        blob_mask = create_blob_mask(blobs, image_gray.shape)
        background_mask = ~blob_mask
        background_intensity = np.sum(image_gray[background_mask])
        return background_intensity, background_mask
    
    # scalebar intensity and related filter
    # Thresholding to detect white regions
    threshold_value = threshold_otsu(image_gray)
    binary_image = image_gray > threshold_value

    # Label connected components in the binary image
    labeled_image = measure.label(binary_image)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_image)

    # Initialize masks for scale bar and text
    scale_bar_mask = np.zeros_like(image_gray)
    scale_text_mask = np.zeros_like(image_gray)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = width / height

        # Detect scale bar (large, rectangular regions)
        if aspect_ratio > 5 and region.area > 100:  # Adjust parameters
            scale_bar_mask[labeled_image == region.label] = 1
            
        # Detect text (smaller, irregular regions)
        elif 1.0 < aspect_ratio < 5 and region.area < 200:  # Adjust for text properties
            scale_text_mask[labeled_image == region.label] = 1

    # Quantify intensity for scale bar and text
    scale_bar_intensity = np.sum(image_gray * scale_bar_mask)
    scale_text_intensity = np.sum(image_gray * scale_text_mask)
    
    # Calculate and display the overall mean intensity for all blobs detected by each method
    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
    
        # Draw blobs on the image
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
        
    plt.tight_layout()
    plt.show()
    
    # Calculate total image intensity (including blobs and background)
    total_image_intensity = np.sum(image_gray)
    
    overall_intensities = []
    n_blobs = []
    mean_intensities = []
    bg_intensities = []
    percent_intensities = []
    percent_combined_intensities = []
    scalebar_threshold = 3573.0 # calculated from images with scale bar only
    scalebar_blobs = 25 # calculated from images with scale bar only
    
    # Iterate over the methods and calculate blob and background intensities
    for blobs, title in zip(blobs_list, titles):
        # Calculate the total intensity of blobs for the method
        if scale_bar_intensity > 0 and scale_text_intensity > 218: # images with scale bar and blobs
            total_blob_intensity = abs(calculate_total_intensity(image_gray, blobs)[0] - scalebar_threshold)
            blob_count = abs(calculate_total_intensity(image_gray, blobs)[1] - scalebar_blobs)
            mean_blob_intensity = total_blob_intensity / blob_count
        elif scale_bar_intensity > 0 and 213.0 < scale_text_intensity < 218: # images with scale bar only
            total_blob_intensity = 0.0
            mean_blob_intensity = 0.0
            blob_count = 0
        else: # images with blobs only
            total_blob_intensity = calculate_total_intensity(image_gray, blobs)[0]
            blob_count = calculate_total_intensity(image_gray, blobs)[1]
            mean_blob_intensity = total_blob_intensity / blob_count
        
        overall_intensities.append(total_blob_intensity)
        mean_intensities.append(mean_blob_intensity)
        n_blobs.append(blob_count)
        
        # Calculate the background intensity for the method
        background_intensity, background_mask = calculate_background_intensity(image_gray, blobs)
        bg_intensities.append(background_intensity)
            
        # Calculate the percentage intensity of blobs compared to total image intensity
        percentage_blob_intensity = (total_blob_intensity / total_image_intensity) * 100
        percent_intensities.append(percentage_blob_intensity)
            
        # Calculate the percentage of blob intensity relative to blob + background
        total_combined_intensity = total_blob_intensity + background_intensity
        percentage_blob_vs_background = (total_blob_intensity / total_combined_intensity) * 100
        percent_combined_intensities.append(percentage_blob_vs_background)
            
        # Print the results
        print(f"Method: {title}")
        print(f"Total Blob Intensity: {total_blob_intensity:.2f}")
        print(f"Background Intensity: {background_intensity:.2f}")
        print(f"Total Image Intensity: {total_image_intensity:.2f}")
        print(f"Percentage Blob Intensity (relative to total image): {percentage_blob_intensity:.2f}%")
        print(f"Percentage Blob Intensity (relative to blob + background): {percentage_blob_vs_background:.2f}%")
        print()
    return mean_intensities, overall_intensities, percent_intensities, bg_intensities, percent_combined_intensities, n_blobs

result = blobs('sample_image.tif') # function call, check result