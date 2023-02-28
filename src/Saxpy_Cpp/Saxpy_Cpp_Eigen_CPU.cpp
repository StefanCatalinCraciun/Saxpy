#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring> 
#include <Eigen/Dense>

void saxpy(Eigen::VectorXd& x, Eigen::VectorXd& y, Eigen::VectorXd& z, float a) {
    z = a * x + y;
}

void saxpy1(Eigen::VectorXd& x, Eigen::VectorXd& y, Eigen::VectorXd& z, float a, long long size) {
    for (long long i = 0; i < size; i++) {
        z[i] = a * x[i] + y[i];
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    int n = 30;
    int iterations = 1;
    float a = {3.1415f};

    double timings[n][iterations];

    for (int elements = 1; elements <= n; elements++) {

        long long size = 1 << elements; // equivalent to 2^n
        //std:: cout <<size<<" ";
        //std:: cout <<"\n";

        for (int it = 0; it < iterations; it++) {

            Eigen::VectorXd x(size), y(size), z(size);

            for (long long i = 0; i < size; i++) {
                x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/10);
                y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/10);
            }

            auto start = std::chrono::high_resolution_clock::now();
            //saxpy(x, y, z, a);
            saxpy1(x, y, z, a, size);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::micro> elapsed = end - start;
            timings[elements-1][it] = elapsed.count(); // store elapsed time in 2D array

        
        }
    }

    double Average_Time[n];
    memset(Average_Time, 0, n*sizeof(double));

    // Print the 2D array of timings
    for (int z = 0; z < n; z++) {
        std::cout << "z = " << z+1 << ": ";
        for (int it = 0; it < iterations; it++) {
            Average_Time[z] += timings[z][it];
            std::cout << timings[z][it] << " ";
        }
        std::cout << std::endl;
    }

    for (int z = 0; z < n; z++) {
        std::cout << "Average time for size "<<z<<": " << Average_Time[z]/iterations << " microseconds" << std::endl;
    }

    // Open the output file
    std::ofstream outfile("Benchmark_Results/Saxpy_Cpp_Eigen_CPU.csv");

    // Write the array values to the file
    for (int z = 0; z < n; z++) {
        outfile << Average_Time[z]/iterations << ",";
    }

    // Close the output file
    outfile.close();

    return 0;
}
