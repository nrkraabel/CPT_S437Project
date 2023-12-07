#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

bool isNondescript(const cv::Mat& image) {
      cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); //grayscale

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    double variance = stddev.val[0] * stddev.val[0]; // Variance

    const double varianceThreshold = 9.7; // threshold for variance I looked this online and just used what bing said
    return variance < varianceThreshold; 
}

int main() {
    std::ifstream inFile("image_data.csv"); 
    std::string imagePath;
    int nondescriptCount = 0;

    while (getline(inFile, imagePath)) {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (isNondescript(image)) {
            nondescriptCount++;
        }
    }

    std::cout << "Number of nondescript images: " << nondescriptCount << std::endl;
    return 0;
}
