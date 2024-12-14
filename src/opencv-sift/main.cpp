#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


// Namespace
using std::chrono::high_resolution_clock;


int main(int argc, char** argv) {
    // Check input arguments
    if (argc != 3) {
        throw std::invalid_argument("Usage: <path to image 1> <path to image 2>");
    }
    
    // Parse the inpur arguments
    std::string path1 = argv[1];
    std::string path2 = argv[2];
    
    // Load images
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        throw std::runtime_error("Failed to load one or both images. Please check the file paths.");
    }

    // Step 1: Detect keypoints and compute descriptors using SIFT (initialize variables)
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;

    // Begin timing
    auto start_time = std::chrono::high_resolution_clock::now();
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // Step 2: Match descriptors using BFMatcher with L2 norm
    matcher.match(descriptors1, descriptors2, matches);

    // Step 3: Sort matches by distance (best matches first)
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
        return a.distance < b.distance;
    });

    // Step 4: Draw the top 50 matches
    const int numGoodMatches = 50;
    std::vector<cv::DMatch> goodMatches(matches.begin(), matches.begin() + std::min((size_t)numGoodMatches, matches.size()));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // End timing

    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches);

    // Step 5: Show results
    // cv::imshow("Matches", imgMatches);
    cv::imwrite("Matches.png", imgMatches);
    std::cout << "Execution time: " << elapsed << " milliseconds.\n";

    cv::waitKey(0);
    return 0;
}
