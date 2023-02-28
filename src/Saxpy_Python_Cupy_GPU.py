#####################################################################################
##################### Saxpy Python Numpy CPU #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 28 # No. of elements in array =2^ekements
iterations = 1

#####################################################################################
#####################################################################################
#####################################################################################

#Import modules
import numpy as np
import cupy as cp
import timeit
np.set_printoptions(suppress = True) # deactivates scientific number format

### For shared memory
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)

def saxpy(size):
    x = 10*cp.random.rand(size).astype(cp.float32)
    y = 10*cp.random.rand(size).astype(cp.float32)
    print (x[0])
    print (y[0])
    start = timeit.default_timer()
    y = a * x + y
    end = timeit.default_timer()
    #print('sssssssssssssssssssss',x.size)
    #print(y[0])
    del x 
    del y
    cp._default_memory_pool.free_all_blocks()

    return (end-start)

N = 2** np.arange(1,elements+1,1)
a = 3.1415
Time=np.empty([iterations,elements])

mempool = cp.get_default_memory_pool()
#pinned_mempool = cp.get_default_pinned_memory_pool() 
mempool.used_bytes()      
mempool.total_bytes()            
#pinned_mempool.n_free_blocks()  


for it in range(iterations):
    counter = 0
    for size in N:

        #pinned_mempool = cp.get_default_pinned_memory_pool()
        #print(mempool.used_bytes() )     
        #print(mempool.total_bytes())       
        #print(pinned_mempool.n_free_blocks())   
        #print(mempool.n_free_blocks())  
        
        Time[it,counter] = (saxpy(size))
        print(counter)
        counter = counter + 1
    print(it)
print("Done loop")
# Sanity check
#print(cp.allclose(a*x+y,z))
        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("/home/stefancc/Desktop/HPC/saxpy/Benchmark_Results/Saxpy_Python_Cupy_GPU.csv", Time_average, delimiter=",")