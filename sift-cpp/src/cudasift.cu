#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"
#include <chrono> 

#include <cuda.h>

// compile
// nvcc cudasift.cu sift.cpp image.cpp -Xcompiler -O3 -Xcompiler -O3 -std c++17 -Xcompiler -fopenmp -o cudasift --expt-relaxed-constexpr

// CUDA Kernel for distance computation
__global__ void compute_distances(std::array<uint8_t, 128> desc_a[], std::array<uint8_t, 128> desc_b[], float* distances, int num_a, int num_b, int dim) {
    int idx_a = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_b = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_a < num_a && idx_b < num_b) {
        float dist = 0.0f;
        for (int i = 0; i < dim; i++) {
            int di = (int)desc_a[idx_a][i] - (int)desc_b[idx_b][i];
            dist += di * di;
        }
        distances[idx_a * num_b + idx_b] = sqrtf(dist);
    }
}

// Match features based on distances
__global__ void match_features(float* distances, int* matches, int num_a, int num_b, float thresh_relative, float thresh_absolute) {
    int idx_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_a < num_a) {
        float best_dist = 1e10f;
        float second_best_dist = 1e10f;
        int best_idx = -1;

        for (int idx_b = 0; idx_b < num_b; idx_b++) {
            float dist = distances[idx_a * num_b + idx_b];
            if (dist < best_dist) {
                second_best_dist = best_dist;
                best_dist = dist;
                best_idx = idx_b;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
            }
        }

        // Apply Lowe's ratio test
        if (best_dist < thresh_relative * second_best_dist && best_dist < thresh_absolute) {
            matches[idx_a] = best_idx;
        } else {
            matches[idx_a] = -1; // No match
        }
    }
}

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png)\n";
        return 0;
    }
    // keep on cpu
    Image a(argv[1]), b(argv[2]);
    a = a.channels == 1 ? a : rgb_to_grayscale(a);
    b = b.channels == 1 ? b : rgb_to_grayscale(b);

    // Start timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // keep on cpu
    std::vector<sift::Keypoint> kps_a = sift::find_keypoints_and_descriptors(a);
    std::vector<sift::Keypoint> kps_b = sift::find_keypoints_and_descriptors(b);
    
    int num_a = kps_a.size();
    int num_b = kps_b.size();
    int dim = 128; // Descriptor size

    // Allocate and copy descriptors to device
    std::array<uint8_t, 128> desc_a[num_a];
    std::array<uint8_t, 128> desc_b[num_b];
    cudaMalloc((void **)&desc_a, num_a * sizeof(std::array<uint8_t, 128>));
    cudaMalloc((void **)&desc_b, num_b * sizeof(std::array<uint8_t, 128>));

    std::array<uint8_t, 128> h_desc_a[num_a];
    std::array<uint8_t, 128> h_desc_b[num_b];
    //uint8_t* h_desc_a = (uint8_t *)malloc(num_a * dim * sizeof(uint8_t));
    //uint8_t* h_desc_b = (uint8_t *)malloc(num_b * dim * sizeof(uint8_t));

    for (int i = 0; i < num_a; i++)
        memcpy(&h_desc_a[i], (void *)&kps_a[i].descriptor, sizeof(std::array<uint8_t, 128>));

    for (int i = 0; i < num_b; i++)
        memcpy(&h_desc_b[i], (void *)&kps_b[i].descriptor, sizeof(std::array<uint8_t, 128>));

    cudaMemcpy(desc_a, h_desc_a, num_a * sizeof(std::array<uint8_t, 128>), cudaMemcpyHostToDevice);
    cudaMemcpy(desc_b, h_desc_b, num_b * sizeof(std::array<uint8_t, 128>), cudaMemcpyHostToDevice);

    // Allocate memory for distances and matches on device
    float* distances;
    int* matches;
    cudaMalloc((void **)&distances, num_a * num_b * sizeof(float));
    cudaMalloc((void **)&matches, num_a * sizeof(int));

    // Launch distance computation kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_a + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_b + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_distances<<<numBlocks, threadsPerBlock>>>(desc_a, desc_b, distances, num_a, num_b, dim);
    cudaDeviceSynchronize();

    // Launch matching kernel
    dim3 threadsPerMatchBlock(256);
    dim3 numMatchBlocks((num_a + threadsPerMatchBlock.x - 1) / threadsPerMatchBlock.x);

    match_features<<<numMatchBlocks, threadsPerMatchBlock>>>(distances, matches, num_a, num_b, 0.7f, 350.0f);
    cudaDeviceSynchronize();

    //Timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy matches back to host
    int* h_matches = (int *)malloc(sizeof(int) * num_a);
    cudaMemcpy(h_matches, matches, num_a * sizeof(int), cudaMemcpyDeviceToHost);

    // Post-process matches
    std::vector<std::pair<int, int>> final_matches;
    for (int i = 0; i < num_a; i++) {
        if (h_matches[i] >= 0) {
            final_matches.push_back({i, h_matches[i]});
        }
    }
    
    
    Image result = sift::draw_matches(a, b, kps_a, kps_b, final_matches);
    result.save("result.jpg");
    
    // Output results
    std::cout << "Found " << final_matches.size() << " feature matches. Output image is saved as result.jpg\n";
    std::cout << "Execution time: " << elapsed << " milliseconds.\n";

    cudaFree(desc_a);
    cudaFree(desc_b);
    cudaFree(distances);
    cudaFree(matches);
    //free(h_desc_a);
    //free(h_desc_b);
    free(h_matches);


    return 0;
}
