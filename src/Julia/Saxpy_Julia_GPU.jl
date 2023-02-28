#####################################################################################
##################### Saxpy Julia GPU #####################################
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

function saxpy(a,x,y)
    y .= a .* x .+ y 
end

#N = 536_870_912
N = 2 .^ (1:elements)
const a = Float32(3.1415)
global x = y = CUDA.Array{Float32}(undef, )

# Preallocate array to store benchmark times
times = zeros(Float64, length(N))

global counter = 1

for size in N
    
    global x = CUDA.rand(Float32, size)
    global y = CUDA.rand(Float32, size)
        
    #@btime saxpy(a,x,y) #just prints the output
    time = @benchmark CUDA.@sync saxpy(a,x,y)
    times[counter] = mean(time.times)
    #println(mean(time.times))

    @show counter
    global counter += 1
end

# Save the results to a CSV file
#writedlm("/home/stefancc/Desktop/HPC/saxpy/Benchmark_Results/Saxpy_Julia_GPU.csv",  times, ',')

for t in times
    @printf("%.3f\n", t)
end

x = nothing
y = nothing
GC.gc(true)
CUDA.reclaim()
CUDA.memory_status()

println(CUDA.memory_status())

global y = 10*rand(Float32, size
global y = 10*rand(Float32, size
global y = 10*rand(Float32, sizeze)
global y = 10*rand(Float32, size