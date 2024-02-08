#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;

/**
 * @brief Detects blurriness in an image using Laplacian variance.
 *
 * @param image The input image.
 * @param threshold The threshold for blurriness.
 * @return True if the image is blurry, false otherwise.
 */
bool isImageBlurry(const Mat& image, double threshold = 100.0) {
    // Compute the Laplacian of the image
    Mat laplacian;
    Laplacian(image, laplacian, CV_64F);

    // Calculate the variance of the Laplacian
    Scalar mean, stddev;
    meanStdDev(laplacian, mean, stddev);

    double variance = stddev.val[0] * stddev.val[0];

    // Check if the image is blurry or not
    if (variance < threshold) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief Computes the Radon Transform of an input image.
 *
 * The Radon Transform is a mathematical transform used in image processing to
 * represent the projection of an image along a set of angles. This function
 * calculates the Radon Transform of the provided image for angles ranging from 0 to 179 degrees.
 *
 * @param image The input image for which the Radon Transform is computed.
 * @return A Mat representing the Radon Transform of the input image. The output
 *         matrix has dimensions (diagonalLength x 180), where diagonalLength is the
 *         length of the diagonal in the image.
 *
 * @details The Radon Transform is calculated using the formula:
 *          \f[
 *          R(\rho, \theta) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} I(x, y) \delta(x \cos \theta + y \sin \theta - \rho) \,dx\,dy
 *          \f]
 *          where:
 *          - \(I(x, y)\) is the intensity of the image at coordinates (x, y).
 *          - \(\rho\) is the distance parameter.
 *          - \(\theta\) is the angle parameter.
 *
 * @note The input image is assumed to be a grayscale image.
 *
 * @see https://en.wikipedia.org/wiki/Radon_transform
 *
 * @return A Mat representing the Radon Transform of the input image. The output
 *         matrix has dimensions (diagonalLength x 180), where diagonalLength is the
 *         length of the diagonal in the image.
 */
Mat computeRadonTransform(const Mat& image) {
    int diagonalLength = static_cast<int>(sqrt(pow(image.rows, 2) + pow(image.cols, 2)));
    Mat radonTransform = Mat::zeros(diagonalLength, 180, CV_64F);

    Point2f center(static_cast<float>(image.cols) / 2, static_cast<float>(image.rows) / 2);
    
    for (int angle = 0; angle < 180; ++angle) {
        Mat rotated;
        Mat rotationMatrix = getRotationMatrix2D(center, static_cast<double>(angle), 1.0);
        warpAffine(image, rotated, rotationMatrix, image.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

        for (int row = 0; row < diagonalLength; ++row) {
            double sum = 0.0;
            for (int col = 0; col < image.cols; ++col) {
                int y = static_cast<int>(center.y + (row - diagonalLength / 2) * sin(CV_PI * angle / 180.0));
                int x = static_cast<int>(center.x + (row - diagonalLength / 2) * cos(CV_PI * angle / 180.0));
                
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    sum += rotated.at<uchar>(y, x);
                }
            }
            radonTransform.at<double>(row, angle) = sum;
        }
    }

    return radonTransform;
}

/**
 * @brief Calculates the gradient magnitude of an input grayscale image.
 *
 * This function applies the Sobel operator to compute the gradient in the x and y directions.
 * The gradient magnitude is then calculated as the square root of the sum of squares of the
 * gradients in the x and y directions.
 *
 * @param gray Input grayscale image.
 * @return The gradient magnitude image.
 *
 * @note The input image should be a single-channel grayscale image (CV_8U or CV_64F).
 */
Mat calculateGradientMagnitude(const Mat& gray) {

    Mat gradientX, gradientY;
    Sobel(gray, gradientX, CV_64F, 1, 0, 3);
    Sobel(gray, gradientY, CV_64F, 0, 1, 3);

    Mat gradientMagnitude;
    magnitude(gradientX, gradientY, gradientMagnitude);

    return gradientMagnitude;
}

// Function to analyze gradient-based blur
bool isBlurred(const Mat& gradientMagnitude, double& threshold) {
    Scalar mean, stddev;
    meanStdDev(gradientMagnitude, mean, stddev);

    double meanGradient = mean.val[0];

    // Adjust the threshold based on experimentation
    return meanGradient < threshold;
}

int main() {

    // Specify the directory containing images
    String directoryPath("/path/to/image/*.png");

    // Use cv::glob to get a list of image files in the directory
    vector<String> imageFiles;
    glob(directoryPath, imageFiles, true);

    // Loop through each image file in the directory
    for (const auto& imageFile : imageFiles) {
        
        // Read the image
        Mat image = imread(imageFile, IMREAD_GRAYSCALE);

        if (image.empty()) {
            cerr << "Error: Could not read the image " << imageFile << endl;
            continue;  // Skip to the next image if reading fails
        }

        //================================ Laplacian ===================================================== 
        auto start_time = std::chrono::high_resolution_clock::now();

        // Detect motion blur with Laplacian
        bool isBlurry = isImageBlurry(image);

        // Further actions based on the result...
        if (isBlurry) {
            cout << "The image based on Laplacian is blurry." << endl;
        } else {
            cout << "The image based on Laplacian is not blurry." << endl;
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Output the duration
        std::cout << "Duration of Laplacian: " << duration.count() << " milliseconds" << std::endl;
        // Display the result on the image
        if (isBlurry) {
            putText(image, "Laplacian Blurry", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        } else {
            putText(image, "Laplacian Not Blurry", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        // Display the original image
        imshow("Original Image", image);
        waitKey(0);

        //================================= Radon Transform =================================================== 

        // Read the image
        image = imread(imageFile, IMREAD_GRAYSCALE);

        if (image.empty()) {
            cerr << "Error: Could not read the image " << imageFile << endl;
            continue;  // Skip to the next image if reading fails
        }

        start_time = std::chrono::high_resolution_clock::now();
        
        // Detect motion blur with Radon Transform
        Mat radonTransform = computeRadonTransform(image);

        // Sum along rows to get the projection
        Mat projection;
        reduce(radonTransform, projection, 0, REDUCE_SUM);

        // Find the angle with the maximum projection
        Point maxLoc;
        minMaxLoc(projection, nullptr, nullptr, nullptr, &maxLoc);

        double blurThreshold = 1000.0;  // You may need to adjust this threshold based on your images

        // Check if the image is blurry or not
        if (projection.at<double>(maxLoc.y, maxLoc.x) > blurThreshold) {
            cout << "The image based on Radon Transform is blurry." << endl;
        } else {
            cout << "The image based on Radon Transform is not blurry." << endl;
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Output the duration
        std::cout << "Duration of Radon Transform: " << duration.count() << " milliseconds" << std::endl;

        // Display the result on the image
        if (isBlurry) {
            putText(image, "Radon Transform Blurry", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        } else {
            putText(image, "Radon Transform Not Blurry", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        // Display the original image
        imshow("Original Image", image);
        waitKey(0);

        //=================================== Sobel ====================================================== 

        image = imread(imageFile, IMREAD_GRAYSCALE);

        if (image.empty()) {
            cerr << "Error: Could not read the image " << imageFile << endl;
            continue;  // Skip to the next image if reading fails
        }
        start_time = std::chrono::high_resolution_clock::now();

        // Detect motion blur with Sobel
        Mat gradientMagnitude = calculateGradientMagnitude(image);

        // Define a threshold for blurriness
        double threshold3 = 30.0;  // Adjust this threshold based on experimentation

        // Check if the image is blurry or not
        if (isBlurred(gradientMagnitude, threshold3)) {
            cout << "The image based on Sobel is blurry." << endl;
        } else {
            cout << "The image based on Sobel is not blurry." << endl;
        }

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Output the duration
        std::cout << "Duration of Sobel: " << duration.count() << " milliseconds" << std::endl;

        // Display the result on the image
        if (isBlurry) {
            putText(image, "Sobel Blurry", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        } else {
            putText(image, "Sobel Not Blurry", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        // Display the original image
        imshow("Original Image", image);
        waitKey(0);
    //================================================================================================ 
    }
    return 0;
}
