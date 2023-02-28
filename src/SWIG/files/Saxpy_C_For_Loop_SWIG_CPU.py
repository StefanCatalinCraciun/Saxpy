#####################################################################################
##################### Saxpy Python For Loop CPU #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 20 # No. of elements in array =2^ekements
iterations = 1


#####################################################################################
#####################################################################################
#####################################################################################

#Import modules
import numpy as np
from simple import saxpy
import time
np.set_printoptions(suppress = True) # deactivates scientific number format
'''
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

        #C.saxpy(x,y,z,a)

        #for i in range(size):
            #z[i] = a * x[i] + y[i]

        end = time.time()
        
        Time[it,counter] = (end - start)*1000
        print(counter)
        counter = counter + 1
print("Done loop")
# Sanity check
print(np.allclose(a*x+y,z))
        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
#np.savetxt("/home/stefancc/Desktop/HPC/saxpy/Benchmark_Results/Saxpy_C_For_Loop_SWIG_CPU.csv", Time_average, delimiter=",")

'''
