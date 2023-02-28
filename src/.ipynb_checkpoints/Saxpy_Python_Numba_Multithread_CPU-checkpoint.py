#####################################################################################
##################### Saxpy Python Numba Multithreading CPU #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 30 # No. of elements in array =2^ekements
iterations = 30

#####################################################################################
#####################################################################################
#####################################################################################

#Import modules
import numpy as np
import time
from numba import jit
import numba
np.set_printoptions(suppress = True) # deactivates scientific number format

@jit(nopython=True,parallel=True,fastmath=True)
def saxpy_numba(a, x, y):
    size = x.shape[0]
    z = np.empty(size)
    for i in numba.prange(size):
        z[i] = a * x[i] + y[i]
    return z

N = 2** np.arange(1,elements+1,1)
a = 3.1415
Time=np.empty([iterations,elements])

for it in range(iterations):
    counter = 0
    for size in N:
        x = 10*np.random.rand(size).astype(np.float32)
        y = 10*np.random.rand(size).astype(np.float32)
        z = np.empty(size)

        start = time.time()
        
        #numba.set_num_threads(16)
        z = saxpy_numba(a, x, y)

        end = time.time()
        
        Time[it,counter] = (end - start)*1000
        #print(counter)
        counter = counter + 1
    print(it)

# Sanity check
print(np.allclose(a*x+y,z))
        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("/home/stefancc/Desktop/HPC/saxpy/Benchmark_Results/Saxpy_Python_Numba_Multithreading_CPU16.csv", Time_average, delimiter=",")

#print(saxpy_numba.parallel_diagnostics())

print("Number of threads used: ", numba.get_num_threads())
print("Threading layer used: ", numba.threading_layer())
