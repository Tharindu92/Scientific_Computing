// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi

#include <stdio.h>
#include <cuda.h>
#include <getopt.h>
#include <stdlib.h>

#define NBIN 10000000  // Number of bins
#define NUM_BLOCK  30  // Number of thread blocks
#define NUM_THREAD  8  // Number of threads per block
#define PI 3.1415926535  // known value of pi
int tid;
float pi = 0;
double pi_d = 0;

// Kernel that executes on the CUDA device
__global__ void cal_pi(float *sum, int nbin, float step, int nthreads, int nblocks) {
	int i;
	float x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
}

__global__ void cal_pi_d(double *sum, int nbin, double step, int nthreads, int nblocks) {
	int i;
	double x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
}

// Main routine that executes on the host
int main(int argc, char **argv) {
	int dp = 0;
	int c;
	while((c = getopt(argc, argv, "d")) != -1){
		switch(c){
			case 'd':
				dp = 1;
				printf("Run with double presision\n");
				break;
			default:
				dp = 0;
				printf("Run with single presision\n");
				break;
		}
	}
	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	if(!dp){
		float *sumHost, *sumDev;  // Pointer to host & device arrays
		float step = 1.0/NBIN;  // Step size
		size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
		sumHost = (float *)malloc(size);  //  Allocate array on host
		cudaMalloc((void **) &sumDev, size);  // Allocate array on device
		// Initialize array in device to 0
		cudaMemset(sumDev, 0, size);
		// Do calculation on device
		cal_pi <<<dimGrid, dimBlock>>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
		// Retrieve result from device and store it in host array
		cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
		for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
			pi += sumHost[tid];
		pi *= step;

		// Print results
		printf("PI = %.10f\n",pi);
		printf("Error = %.10f\n",abs(PI-pi));

		// Cleanup
		free(sumHost);
		cudaFree(sumDev);
	}else{
		double *sumHost, *sumDev;  // Pointer to host & device arrays
		double step = 1.0/NBIN;  // Step size
		size_t size = NUM_BLOCK*NUM_THREAD*sizeof(double);  //Array memory size
		sumHost = (double *)malloc(size);  //  Allocate array on host
		cudaMalloc((void **) &sumDev, size);  // Allocate array on device
		// Initialize array in device to 0
		cudaMemset(sumDev, 0, size);
		// Do calculation on device
		cal_pi_d <<<dimGrid, dimBlock>>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
		// Retrieve result from device and store it in host array
		cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
		for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
			pi_d += sumHost[tid];
		pi_d *= step;

		// Print results
		printf("PI = %.10lf\n",pi_d);
		printf("Error = %.10lf\n",abs(PI-pi_d));

		// Cleanup
		free(sumHost);
		cudaFree(sumDev);
	}


	return 0;
}
