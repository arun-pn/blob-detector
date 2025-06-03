# Blob Detection and Analysis
## Description
This Python script detects and analyzes circular features (blobs) in fluorescence microscopy images using the `scikit-image` library (Van Der Walt et al., 2014). It converts RGB images to grayscale using the `rgb2gray` function, which calculates pixel brightness based on human visual perception (Poynton, 1997).
Blobs are assumed to be bright, circular regions on a dark background. They are detected using the Laplacian of Gaussian (LoG) method (Lindeberg, 1993) via `skimage.feature.blob_log`. The script then creates binary masks and extracts properties like position, radius, and intensity for each detected blob.
This tool is useful for applications such as cell counting, biological image analysis, and other tasks involving circular region detection in grayscale images.

References
- Lindeberg, T. (1993). Detecting salient blob-like image structures and their scales with a scale-space primal sketch: A method for focus-of-attention. International Journal of Computer Vision, 11(3). https://doi.org/10.1007/BF01469346
- Poynton, C. (1997). Frequently Asked Questions about Color. https://poynton.ca/PDFs/ColorFAQ.pdf
- Van Der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). Scikit-image: Image processing in python. PeerJ, 2014(1). https://doi.org/10.7717/peerj.453

## Features
- Convert RGB images to grayscale
- Detect blobs using Laplacian of Gaussian (LoG) method
- Compare LoG with other methods such as Difference of Gaussian (DoG) and Determinant of Hessian (DoH)
- Generate binary masks for detected blobs
- Extract blob coordinates and radii
- Calculate accurate blob intensities
- Optional visualization of results

## Installation / Requirements
To install the required packages, run this command in your terminal or command prompt:

```bash
pip install -r requirements.txt
```
I have used Python version: Python 3.10.17

## Example
To understand the script, run the code 'democode.py' with the image 'sample_image.tif' placed in the same directory. You could also check the results with the 'scalebar_image.tif' image file.
