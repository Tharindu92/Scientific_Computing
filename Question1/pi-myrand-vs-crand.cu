// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi

#include <stdio.h>
#include <cuda.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535

int tid;
float pi = 0;
double pi_d = 0;


// Generate random number from CPU and send to GPU
__device__ float my_rand(unsigned int *seed) {
	unsigned long a = 16807;  // constants for random number generator
        unsigned long m = 2147483647;   // 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

        return ((float)x)/m;
}

__device__ double my_rand_d(unsigned int *seed) {
	unsigned long a = 16807;  // constants for random number generator
        unsigned long m = 2147483647;   // 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

        return ((double)x)/m;
}
// Kernel that executes on the CUDA device for Pi-Myrand
__global__ void gpu_monte_carlo_myrand(float *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	unsigned int seed =  tid + 1;  // starting number in random sequence

	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = my_rand(&seed);
		y = my_rand(&seed);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}

__global__ void gpu_monte_carlo_myrand_d(double *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	double x, y;

	unsigned int seed =  tid + 1;  // starting number in random sequence

	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = my_rand_d(&seed);
		y = my_rand_d(&seed);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (double) TRIALS_PER_THREAD; // return estimate of pi
}

// Kernel that executes on the CUDA device for Pi-Curand
__global__ void gpu_monte_carlo_curand(float *estimate, curandState *states) {
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

__global__ void gpu_monte_carlo_curand_d(double *estimate, curandState *states) {
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

//Calculate pi using crand and myrand
void pi_rand(int dp){
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

		gpu_monte_carlo_curand<<<BLOCKS, THREADS>>>(dev, devStates);

		cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results

		float pi_gpu_cu;
		for(int i = 0; i < BLOCKS * THREADS; i++) {
			pi_gpu_cu += host[i];
		}

		pi_gpu_cu /= (BLOCKS * THREADS);

		stop = clock();
		cudaFree(dev);
		cudaFree(devStates);

		printf("GPU pi calculated with CuRand %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();

		cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

		gpu_monte_carlo_myrand<<<BLOCKS, THREADS>>>(dev);

		cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results

		float pi_gpu_my;
		for(int i = 0; i < BLOCKS * THREADS; i++) {
			pi_gpu_my += host[i];
		}

		pi_gpu_my /= (BLOCKS * THREADS);

		stop = clock();

		printf("GPU pi calculated with MyRand %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();
		float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
		stop = clock();
		printf("CPU pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		printf("CUDA estimate of PI with CuRand = %f [error of %f]\n", pi_gpu_cu, pi_gpu_cu - PI);
		printf("CUDA estimate of PI with MyRand = %f [error of %f]\n", pi_gpu_my, pi_gpu_my - PI);
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

			gpu_monte_carlo_curand_d<<<BLOCKS, THREADS>>>(dev, devStates);

			cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost); // return results

			double pi_gpu_cu;
			for(int i = 0; i < BLOCKS * THREADS; i++) {
				pi_gpu_cu += host[i];
			}

			pi_gpu_cu /= (BLOCKS * THREADS);

			stop = clock();

			printf("GPU pi calculated with CuRand %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

			cudaFree(dev);
			cudaFree(devStates);

			start = clock();

			cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(double)); // allocate device mem. for counts

			gpu_monte_carlo_myrand_d<<<BLOCKS, THREADS>>>(dev);

			cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost); // return results

			double pi_gpu_my;
			for(int i = 0; i < BLOCKS * THREADS; i++) {
				pi_gpu_my += host[i];
			}

			pi_gpu_my /= (BLOCKS * THREADS);

			stop = clock();

			printf("GPU pi calculated with MyRand %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

			start = clock();
			double pi_cpu = host_monte_carlo_d(BLOCKS * THREADS * TRIALS_PER_THREAD);
			stop = clock();
			printf("CPU pi calculated in %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

			printf("CUDA estimate of PI with CuRand = %lf [error of %lf]\n", pi_gpu_cu, pi_gpu_cu - PI);
			printf("CUDA estimate of PI with MyRand = %lf [error of %lf]\n", pi_gpu_my, pi_gpu_my - PI);
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

	pi_rand(dp);

	return 0;
}

