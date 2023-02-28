using CUDA
using BenchmarkTools

CUDA.versioninfo()

function axpy(a,x,y)
    y .= a .* x .+ y 
end

const dim = 536_870_912

const a = Float32(3.1415)


x = ones(Float32, dim)
y = ones(Float32, dim)

typeof=(x)
sizeof(x)
@btime axpy(a,x,y)
varinfo()

x = ones(Float32, dim)
y = ones(Float32, dim)

typeof=(x)
sizeof(x)
@btime y .= a .* x .+ y 
varinfo()

cpu_time1 = 398.654 #ms
cpu_time2 = 505.881 #ms




x = CUDA.ones(Float32, dim)
y = CUDA.ones(Float32, dim)

x = ones(Float32, dim)
y = ones(Float32, dim)

x_gpu = CuArray(x)
y_gpu = CuArray(y)

x = Array(x_gpu)
y = Array(y_gpu)


typeof=(x)
sizeof(x)
@btime CUDA.@sync axpy(a,x,y)
varinfo()

CUDA.memory_status()

x = nothing
y = nothing
x_gpu = nothing
y_gpu = nothing
GC.gc(true)

CUDA.reclaim()


CUDA.memory_status()

x = CUDA.ones(Float32, dim)
y = CUDA.ones(Float32, dim)

typeof=(x)
sizeof(x)
@btime CUDA.@sync y .= a .* x .+ y 
varinfo()

gpu_time1 = 34.690 #ms
gpu_time2 = 34.968 #ms

performance = cpu_time1 / gpu_time1

#vscodedisplay(x)





###################################

using CUDA
using BenchmarkTools

CUDA.versioninfo()

function axpy(a,x,y)
    y .= a .* x .+ y 
end

const dim = 536_870_912

const a = Float32(3.1415)

x = CUDA.ones(Float32, dim)
y = CUDA.ones(Float32, dim)

CUDA.memory_status()

nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
)

nblocks = cld(dim, nthreads)

function saxpy_gpu_kernel!(a,x,y)
    # calculate index number
    i = (blockIdx().x - 1)* blockDim().x + threadIdx().x
    # calculate saxpy
    if i <= length(y)
        # eliminate array bounds check
        @inbounds y[i] = a * x[i] + y[i]
    end
    return nothing
end

# launch cuda kernel
@btime CUDA.@sync @cuda(
    threads = nthreads,
    blocks = nblocks,
    saxpy_gpu_kernel!(a,x,y)
)


cpu_time = 398.654 #ms
gpu_time = 34.690 #ms
kernel_time = 33.276

cpu_time / gpu_time
cpu_time / kernel_time

gpu_time / kernel_time

###########################################################

# using CUBLAS

using CUDA
using CUDA.CUBLAS
using BenchmarkTools

const dim = 536_870_912

const a = Float32(3.1415)

x = CUDA.ones(Float32, dim)
y = CUDA.ones(Float32, dim)

CUDA.memory_status()

@btime CUDA.@sync CUBLAS.axpy!(dim, a, x, y)


cpu_time = 201.367 #ms
gpu_time = 20.579 #ms
kernel_time = 19.963
cublas_time = 19.714

cpu_time / gpu_time
cpu_time / kernel_time

gpu_time / kernel_time
gpu_time / cublas_time

