/* Need to use 1D index for accessing array elements */

void saxpy(double* x, int sizex, double* y, int sizey, double* z, int sizez, float a){
    for (int i = 0; i < sizex; i++) {
        z[i] = a * x[i] + y[i];
    }
}


