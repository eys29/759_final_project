#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"
#include <chrono> // Include the chrono library
int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png)\n";
        return 0;
    }
    Image a(argv[1]), b(argv[2]);
    a = a.channels == 1 ? a : rgb_to_grayscale(a);
    b = b.channels == 1 ? b : rgb_to_grayscale(b);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<sift::Keypoint> kps_a = sift::find_keypoints_and_descriptors(a);
    std::vector<sift::Keypoint> kps_b = sift::find_keypoints_and_descriptors(b);
    //std::pair allows mix of heterogenous types in c++
    //returns a list of pairs where first int is i:index of keypoint in a and second int
    // is j: index of keypoint in b
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps_a, kps_b);
    Image result = sift::draw_matches(a, b, kps_a, kps_b, matches);
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // Now can save result
    result.save("result.jpg");
    // Output results
    std::cout << "Found " << matches.size() << " feature matches. Output image is saved as result.jpg\n";
    std::cout << "Execution time: " << elapsed << " milliseconds.\n";
    return 0;
}
