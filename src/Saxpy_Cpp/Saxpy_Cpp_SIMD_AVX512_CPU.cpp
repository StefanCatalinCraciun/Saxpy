#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring> 



void simd_addition_multiply(float* a, float* b, float* result, float c, long long size) {
    __m512* xmm_a = (__m512*)a;
    __m512* xmm_b = (__m512*)b;
    __m512* xmm_result = (__m512*)result;
    __m512 xmm_c = _mm512_set1_ps(c);

    long long num_simd_ops = size / 16;

    for (long long i = 0; i < num_simd_ops; i++) {
        xmm_result[i] = _mm512_fmadd_ps(xmm_a[i], xmm_c, xmm_b[i]);
    }

    long long remainder = size % 16;
    for (long long i = size - remainder; i < size; i++) {
        result[i] = a[i] * c + b[i];
    }
}


int main() {
    std::cout << std::fixed << std::setprecision(6);

    int n = 30;
    int iterations = 1;
    float c = {3.1415f};

    double timings[n][iterations];

    for (int z = 1; z <= n; z++) {

        long long size = 1 << z; // equivalent to 2^n
        //std:: cout <<size<<" ";
        //std:: cout <<"\n";

        for (int it = 0; it < iterations; it++) {


        float* a = (float*)aligned_alloc(64, size * sizeof(float));
        float* b = (float*)aligned_alloc(64, size * sizeof(float));
        float* result = (float*)aligned_alloc(64, size * sizeof(float));

        

        

        for (long long i = 0; i < size; i++) {
            a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/10);
            b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/10);
        }

        auto start = std::chrono::high_resolution_clock::now();
        simd_addition_multiply(a, b, result, c, size);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::micro> elapsed = end - start;
        //std::cout << "Execution time: " << elapsed.count() << " ms" << std::endl;
        timings[z-1][it] = elapsed.count(); // store elapsed time in 2D array
        //std::cout << "Result: " << result[0] << " ... " << result[size-1] << std::endl;

        _mm_free(a);
        _mm_free(b);
        _mm_free(result);
        }
    }

    double Average_Time[n];
    memset(Average_Time, 0, n*sizeof(double));

    // Print the 2D array of timings
    for (int z = 0; z < n; z++) {
        std::cout << "z = " << z+1 << ": ";
        for (int it = 0; it < iterations; it++) {
            Average_Time[z] =Average_Time[z] + timings[z][it];

            std::cout << timings[z][it] << " ";

        //Average_Time[z] = Average_Time[z]/2;
    }
    std::cout << std::endl;
    }

    for (int z = 0; z < n; z++) {
    std::cout << "Average time for size "<<z<<": " << Average_Time[z]/iterations << " microseconds" << std::endl;
}

    std:: cout << sizeof(Average_Time)/sizeof(double);



    // Open the output file
    std::ofstream outfile("Benchmark_Results/Saxpy_Cpp_SIMD_AVX512_CPU.csv");

    // Write the array values to the file
    for (int z = 0; z < n; z++) {
        //long long size = 1 << z;
        outfile << Average_Time[z]/iterations << ",";
    }

    // Close the output file
    outfile.close();

    return 0;
}
