#####################################################################################
##################### Saxpy Julia For Loop CPU #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 20 # No. of elements in array =2^ekements
iterations = 1


#####################################################################################
#####################################################################################
#####################################################################################

function saxpy(a,x,y)
    y .= a .* x .+ y 
end


elements = 20 
iterations = 1
using BenchmarkTools
N = 2 .^ (1:elements)
const a = Float32(3.1415)
x = y = nothing
for it in 1:iterations
    counter = 1
    for size in N

        println(size)

        x = ones(Float32, Int32(size))
        y = ones(Float32, Int32(size))

        println(typeof(y))
 
        @btime saxpy(a,x,y)
      
        counter += 1
    end
    
end


