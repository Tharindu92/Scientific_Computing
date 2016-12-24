// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi

#include <stdio.h>
#include <cuda.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define TRIALS_PER_THREAD 1118481
#define BLOCKS  30  // Number of thread BLOCKS
#define THREADS  8  // Number of threads per BLOCKS
#define NBIN 268435440  // Number of bins
#define PI 3.1415926535  // known value of pi

int tid;
float pi = 0;
double pi_d = 0;


// Kernel that executes on the CUDA device for Pi-Mystery
__global__ void cal_pi(float *sum, int nbin, float step, int nthreads, int nBLOCKS) {
	int i;
	float x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the BLOCKS
	for (i=idx; i< nbin; i+=nthreads*nBLOCKS) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
}

__global__ void cal_pi_d(double *sum, int nbin, double step, int nthreads, int nBLOCKS) {
	int i;
	double x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the BLOCKS
	for (i=idx; i< nbin; i+=nthreads*nBLOCKS) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
}

// Kernel that executes on the CUDA device for Pi-Curand
__global__ void gpu_monte_carlo(float *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}

__global__ void gpu_monte_carlo_d(double *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	double x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (double) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle = 0;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

double host_monte_carlo_d(long trials) {
	double x, y;
	long points_in_circle = 0;
	for(long i = 0; i < trials; i++) {
		x = rand() / (double) RAND_MAX;
		y = rand() / (double) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

//Calculate pi using Pi Mystery
void pi_mystry(int dp){
	clock_t start, stop;
	dim3 dimGrid(BLOCKS,1,1);  // Grid dimensions
	dim3 dimBLOCKS(THREADS,1,1);  // BLOCKS dimensions
	if(!dp){	//single precision
		printf("Run with single precision\n");
		float *sumHost, *sumDev;  // Pointer to host & device arrays
		float step = 1.0/NBIN;  // Step size
		size_t size = BLOCKS*THREADS*sizeof(float);  //Array memory size
		sumHost = (float *)malloc(size);  //  Allocate array on host
		printf("Pi Mystery # of trials per thread = %d, # of BLOCKS = %d, # of threads/BLOCKS = %d, # of bins = %d\n", TRIALS_PER_THREAD,
			BLOCKS, THREADS,NBIN);
		//Star timing
		start = clock();
		cudaMalloc((void **) &sumDev, size);  // Allocate array on device
		// Initialize array in device to 0
		cudaMemset(sumDev, 0, size);
		// Do calculation on device
		cal_pi <<<dimGrid, dimBLOCKS>>> (sumDev, NBIN, step, THREADS, BLOCKS); // call CUDA kernel
		// Retrieve result from device and store it in host array
		cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
		for(tid=0; tid<THREADS*BLOCKS; tid++)
			pi += sumHost[tid];
		pi *= step;

		stop = clock();
		//Stop timing
		//print time
		printf("GPU pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);
		// Print results
		printf("Pi-mystery's CUDA estimate of PI = %f [error of %f]\n", pi, pi - PI);

		// Cleanup
		free(sumHost);
		cudaFree(sumDev);
	}else{	//double precision
		printf("Run with double precision\n");
		double *sumHost, *sumDev;  // Pointer to host & device arrays
		double step = 1.0/NBIN;  // Step size
		size_t size = BLOCKS*THREADS*sizeof(double);  //Array memory size
		sumHost = (double *)malloc(size);  //  Allocate array on host
		start = clock();
		cudaMalloc((void **) &sumDev, size);  // Allocate array on device
		// Initialize array in device to 0
		cudaMemset(sumDev, 0, size);
		// Do calculation on device
		cal_pi_d <<<dimGrid, dimBLOCKS>>> (sumDev, NBIN, step, THREADS, BLOCKS); // call CUDA kernel
		// Retrieve result from device and store it in host array
		cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
		for(tid=0; tid<THREADS*BLOCKS; tid++)
			pi_d += sumHost[tid];
		pi_d *= step;

		stop = clock();
		//Stop timing
		//print time
		printf("GPU pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);
		// Print results
		printf("Pi-mystery's CUDA estimate of PI = %f [error of %f]\n", pi_d, pi_d - PI);

		// Cleanup
		free(sumHost);
		cudaFree(sumDev);
	}
}

//Calculate pi using crand
void pi_curand(int dp){
	clock_t start, stop;
	if(!dp){
		float host[BLOCKS * THREADS];
		float *dev;
		curandState *devStates;

		printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
	BLOCKS, THREADS);

		start = clock();

		cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

		cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

		gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

		cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results

		float pi_gpu;
		for(int i = 0; i < BLOCKS * THREADS; i++) {
			pi_gpu += host[i];
		}

		pi_gpu /= (BLOCKS * THREADS);

		stop = clock();

		printf("GPU pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();
		float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
		stop = clock();
		printf("CPU pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
		cudaFree(dev);
		cudaFree(devStates);
	}else{
		double host[BLOCKS * THREADS];
			double *dev;
			curandState *devStates;

			printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
		BLOCKS, THREADS);

			start = clock();

			cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(double)); // allocate device mem. for counts

			cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

			gpu_monte_carlo_d<<<BLOCKS, THREADS>>>(dev, devStates);

			cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost); // return results

			double pi_gpu;
			for(int i = 0; i < BLOCKS * THREADS; i++) {
				pi_gpu += host[i];
			}

			pi_gpu /= (BLOCKS * THREADS);

			stop = clock();

			printf("GPU pi calculated in %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

			start = clock();
			double pi_cpu = host_monte_carlo_d(BLOCKS * THREADS * TRIALS_PER_THREAD);
			stop = clock();
			printf("CPU pi calculated in %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

			printf("CUDA estimate of PI = %lf [error of %lf]\n", pi_gpu, pi_gpu - PI);
			printf("CPU estimate of PI = %lf [error of %lf]\n", pi_cpu, pi_cpu - PI);
			cudaFree(dev);
			cudaFree(devStates);
	}
}

// Main routine that executes on the host
int main(int argc, char **argv) {
	int dp = 0;
	int c;
	//Select the precision used according to user arguments
	while((c = getopt(argc, argv, "d")) != -1){
		switch(c){
			case 'd':
				dp = 1;
				break;
			default:
				dp = 0;
				break;
		}
	}

	pi_mystry(dp);
	pi_curand(dp);

	return 0;
}

