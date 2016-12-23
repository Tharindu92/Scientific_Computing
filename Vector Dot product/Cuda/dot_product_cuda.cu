#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>

#define NUM_THREAD  256  // Number of thread blocks
#define print(x) printf("%d",x)

double doubleGen();
__global__ void dotProduct_CUDA(double *sum, long size, double *vector1, double *vector2){
	long idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	if(idx < size){
		//printf("Before idx%d : %lf\n",idx,sum[idx]);
		sum[idx] = (vector2[idx]) * (vector1[idx]);
		//printf("Vector1 %lf\n",*(vector1+idx));
		//printf("Vector2 %lf\n",vector2[idx]);
		//printf("After idx%d : %lf\n",idx,sum[idx]);
	}
}

int main(void){
	long num = 1000000000;
	int num_block = (num + NUM_THREAD - 1)/(NUM_THREAD);
	dim3 dimGrid(num_block,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	size_t size = num*sizeof(double);  //Array memory size.
	double *sumHost, *sumDev;  // Pointer to host & device arrays
	double *vector1, *vector1_device;
	vector1 = (double*) malloc(size);
	double *vector2, *vector2_device;
	vector2 = (double*) malloc(size);
	double dotProduct = 0;
	int tid,j;	
	srand(time(NULL));
	for(j=0; j < num; j++){
		*(vector1+j) = doubleGen();
		*(vector2+j) = doubleGen();
	}
	print(38);
	sumHost = (double *)malloc(size); print(39); //  Allocate array on host
	
	cudaMalloc((void **) &sumDev, size);  print(42);// Allocate array on device
	cudaMalloc((void **) &vector1_device, size);  // Allocate array on device
	cudaMalloc((void **) &vector2_device, size);  // Allocate array on device
	print(45);
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	cudaMemcpy(vector1_device, vector1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(vector2_device, vector2, size, cudaMemcpyHostToDevice);

	// Do calculation on device
	dotProduct_CUDA <<<num_block, NUM_THREAD>>> (sumDev, num, vector1_device, vector2_device); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<num; tid++){
		dotProduct += sumHost[tid];
		printf("Sum Host %lf\n",sumHost[tid]);
	}
	// Print results
	printf("Cuda Result = %lf\n",dotProduct);
	dotProduct = 0;
	for (j=0; j< num; j++) {
		dotProduct += (*(vector1+j)) * (*(vector2+j));
		//printf("Vector 1 : %lf\n",vector1[j]);
	}
	
	/*for (j=0; j< num; j++) {
//		dotProduct += (*(vector1+j)) * (*(vector1+j));
		printf("Vector 2 : %lf\n",vector2[j]);
	}*/

	printf("Serial Result = %lf\n",dotProduct);
	// Cleanup
	free(sumHost); 
	free(vector1);
	free(vector2);	
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);	
}

double doubleGen(){
	double num ;
	num = 1.0 * random() / RAND_MAX + 1.0;
	return num;
}
