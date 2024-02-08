# Image Blur Detection

This project provides a C++ application that utilizes OpenCV to detect blurriness in images through various methods including Laplacian variance, Radon Transform, and Sobel operator. It's designed to help in analyzing images for blurriness, which can be crucial for quality control in image processing applications.

## Prerequisites

To compile and run this application, you need to have the following installed:

- C++ Compiler (g++ recommended)
- OpenCV 4.x
- pkg-config

## Compilation

A Makefile is provided for easy compilation. You can compile the project using the following command:

```bash
make 
```

This will generate an executable named `image_blurr_detection`.

## Usage

After compilation, you can run the application as follows:

```bash
./image_blurr_detection 
```


The application expects images in a specific directory (`/path/to/image/*.png`). You need to modify the `directoryPath` variable in the `main` function to point to your directory containing the images you wish to analyze.

## Methods Implemented

1. **Laplacian Variance**: Detects blurriness by calculating the variance of the Laplacian of the image. A lower variance indicates a blurrier image.

2. **Radon Transform**: Utilizes the Radon Transform to detect blurriness by analyzing the projection of the image along various angles.

3. **Sobel Operator**: Computes the gradient magnitude of the image using the Sobel operator. A lower average gradient magnitude indicates a blurrier image.

## Output

The application will process each image in the specified directory, applying each of the three methods to determine if the image is blurry. It will display the original image with a label indicating the result of the blurriness detection for each method. Additionally, the duration taken for each method to process an image is outputted to the console.

## Note

This application is designed for educational and testing purposes. The thresholds for detecting blurriness might need adjustments based on your specific requirements and the characteristics of the images you are analyzing.