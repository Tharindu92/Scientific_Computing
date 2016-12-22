#include <stdlib.h>
#include <cuda.h>

#define NUM_BLOCK  30  // Number of thread blocks
#define NUM_THREAD  8  // Number of threads per block


int main(void){
	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	int size = 1000000000;
	float *sumHost, *sumDev;  // Pointer to host & device arrays
	float *vector1;
	vector1 = (float*) malloc(size * sizeof(float));
	float *vector2;
	vector2 = (float*) malloc(size * sizeof(float));
	for(j=0; j < size; j++){
		*(vector1+j) = floatGen();
		*(vector2+j) = floatGen();
	}
}

float floatGen(){
	float num ;
	num = 1.0 * random() / RAND_MAX + 1.0;
	return num;
}
