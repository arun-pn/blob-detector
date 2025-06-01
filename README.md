# Blob Detection and Analysis
## Description

This Python script detects and analyzes circular features (blobs) in fluorescence microscopy images using the `scikit-image` library. It converts RGB images to grayscale using the `rgb2gray` function, which calculates pixel brightness based on human visual perception.
Blobs are assumed to be bright, circular regions on a dark background. They are detected using the Laplacian of Gaussian (LoG) method via `skimage.feature.blob_log`. The script then creates binary masks and extracts properties like position, radius, and intensity for each detected blob.
This tool is useful for applications such as cell counting, biological image analysis, and other tasks involving circular region detection in grayscale images.
