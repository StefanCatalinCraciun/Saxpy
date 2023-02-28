#####################################################################################
##################### Saxpy Julia CUBLAS GPU #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 29 # No. of elements in array =2^ekements
#iterations = 1 ; @btime uses 20 iterations by default and gives back the median


#####################################################################################
#####################################################################################
#####################################################################################

using BenchmarkTools
using Statistics
using DelimitedFiles
using Printf
using CUDA
using CUDA.CUBLAS


#N = 536_870_912
N = 2 .^ (1:elements)
const a = Float32(3.1415)
global x = y = CUDA.Array{Float32}(undef, )

# Preallocate array to store benchmark times
times = zeros(Float64, length(N))

global counter = 1
global n = 0

for size in N
    
    global x = CUDA.rand(Float32, size)
    global y = CUDA.rand(Float32, size)

    global n = Int64(size)
    #@btime saxpy(a,x,y) #just prints the output
    time = @benchmark CUDA.@sync CUBLAS.axpy!(n, a, x, y)
    times[counter] = mean(time.times)
    #println(mean(time.times))

    @show counter
    global counter += 1
end

# Save the results to a CSV file
writedlm("/home/stefancc/Desktop/HPC/saxpy/Benchmark_Results/Saxpy_Julia_CUBLAS_GPU.csv",  times, ',')

for t in times
    @printf("%.3f\n", t)
end

x = nothing
y = nothing
GC.gc(true)
CUDA.reclaim()
CUDA.memory_status()

println(CUDA.memory_status())