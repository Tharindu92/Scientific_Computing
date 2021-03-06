// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/PiMyRandom.cu

// Written by Barry Wilkinson, UNC-Charlotte. PiMyRandom.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi

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

__global__ void gpu_monte_carlo(float *estimate) {
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

__global__ void gpu_monte_carlo_d(double *estimate) {
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
		x = (double)rand() /  RAND_MAX;
		y = (double)rand() /  RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0);
	}
	return (double)(4.0 * points_in_circle / trials);
}

int main(int argc, char **argv) {
	int dp = 0;
	int c;
	while((c = getopt(argc, argv, "d")) != -1){
		switch(c){
			case 'd':
				dp = 1;
				printf("Run with double presision\n");
				c = -1;
				break;
			default:
				dp = 0;
				printf("Run with single presision\n");
				break;
		}
	}
	clock_t start, stop;
	if(!dp){
		float host[BLOCKS * THREADS];
		float *dev;


		printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
	BLOCKS, THREADS);

		start = clock();

		cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

		gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev);

		cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results

		float pi_gpu;
		for(int i = 0; i < BLOCKS * THREADS; i++) {
			pi_gpu += host[i];
		}

		pi_gpu /= (BLOCKS * THREADS);

		stop = clock();

		printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

		start = clock();
		float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
		stop = clock();
		printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

		printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	}else{
		double host[BLOCKS * THREADS];
		double *dev;

		printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
	BLOCKS, THREADS);

		start = clock();

		cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(double)); // allocate device mem. for counts

		gpu_monte_carlo_d<<<BLOCKS, THREADS>>>(dev);

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
	}
	
	return 0;
}
