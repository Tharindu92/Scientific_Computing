// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <omp.h>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi

/*
 * \Run CuRand
 */
//run on GPU
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

//run on CPU without threads
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

//run on cpu with threads
float para_monte_carlo(long trials, int thread_count) {
	float x, y;
	long points_in_circle = 0;
	#pragma omp parallel num_threads(thread_count) private(x,y)
	{
		#pragma omp for schedule(static) reduction(+:points_in_circle)
		for(long i = 0; i < trials; i++) {
			x = rand() / (float) RAND_MAX;
			y = rand() / (float) RAND_MAX;
			points_in_circle += (x*x + y*y <= 1.0f);
		}
	}

	return 4.0f * points_in_circle / trials;
}

double para_monte_carlo_d(long trials, int thread_count) {
	double x, y;
	long points_in_circle = 0;
	#pragma omp parallel num_threads(thread_count) private(x,y)
	{
		#pragma omp for schedule(static) reduction(+:points_in_circle)
		for(long i = 0; i < trials; i++) {
			x = rand() / (double) RAND_MAX;
			y = rand() / (double) RAND_MAX;
			points_in_circle += (x*x + y*y <= 1.0f);
		}
	}
	return 4.0f * points_in_circle / trials;
}

/*
 * pi-curand-thrust
 */

struct estimate_pi :
    public thrust::unary_function<unsigned int, float>
{
  __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 8192; // samples per thread

    unsigned int seed = thread_id;

    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      float x = curand_uniform(&s);
      float y = curand_uniform(&s);

      // measure distance from the origin
      float dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

struct estimate_pi_d :
    public thrust::unary_function<unsigned int, double>
{
  __device__
  double operator()(unsigned int thread_id)
  {
    double sum = 0;
    unsigned int N = 8192; // samples per thread

    unsigned int seed = thread_id;

    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      double x = curand_uniform(&s);
      double y = curand_uniform(&s);

      // measure distance from the origin
      double dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

int main(int argc, char **argv) {
	int dp = 0;
	int c;
	int thread_count = 2;
	while((c = getopt(argc, argv, "dn:")) != -1){
		switch(c){
			case 'd':
				dp = 1;
				break;
			case 'n':
				thread_count = atoi(optarg);
				if(thread_count > 8 || thread_count < 2){
					printf("Invalid Number of threads\nThread number is set to 2\n");
					thread_count = 2;
				}
				break;
			default:
				dp = 0;
				break;
		}
	}
	clock_t start, stop;
	int M = 32768;
	if(!dp){
		printf("Run with single precision\n");
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

		printf("GPU CuRand pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();
		float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
		stop = clock();
		printf("CPU pi calculated in %lf s.\n", (double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();
		float pi_para = para_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD, thread_count);
		stop = clock();
		printf("CPU with %d threads pi calculated in %lf s.\n", thread_count,(double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();
		float estimate = thrust::transform_reduce(
			          thrust::counting_iterator<int>(0),
			          thrust::counting_iterator<int>(M),
			          estimate_pi(),
			          0.0f,
			          thrust::plus<float>());
			    estimate /= M;
		stop = clock();
		printf("CUDA Thrust CuRand pi calculated in %lf s.\n",(double)(stop-start)/CLOCKS_PER_SEC);

		printf("CUDA CuRand estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
		printf("CPU with %d threads estimate of PI = %f [error of %f]\n", thread_count, pi_para, pi_para - PI);
		printf("CUDA Thrust CuRand estimate of PI = %f [error of %f]\n", estimate, estimate - PI);

	}else{
		printf("Run with double precision\n");
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

		printf("GPU CuRand pi calculated in %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

		start = clock();
		double pi_cpu = host_monte_carlo_d(BLOCKS * THREADS * TRIALS_PER_THREAD);
		stop = clock();
		printf("CPU pi calculated in %lf s.\n", (stop-start)/(double)CLOCKS_PER_SEC);

		start = clock();
		double pi_para = para_monte_carlo_d(BLOCKS * THREADS * TRIALS_PER_THREAD, thread_count);
		stop = clock();
		printf("CPU with %d threads pi calculated in %lf s.\n", thread_count,(double)(stop-start)/CLOCKS_PER_SEC);

		start = clock();
		double estimate = thrust::transform_reduce(
			          thrust::counting_iterator<int>(0),
			          thrust::counting_iterator<int>(M),
			          estimate_pi_d(),
			          0.0f,
			          thrust::plus<double>());
			    estimate /= M;
		stop = clock();
		printf("CUDA Thrust CuRand pi calculated in %lf s.\n",(double)(stop-start)/CLOCKS_PER_SEC);


		printf("CUDA CuRand estimate of PI = %lf [error of %lf]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %lf [error of %lf]\n", pi_cpu, pi_cpu - PI);
		printf("CPU with %d threads estimate of PI = %lf [error of %lf]\n", thread_count, pi_para, pi_para - PI);
		printf("CUDA Thrust CuRand estimate of PI = %lf [error of %lf]\n", estimate, estimate - PI);
	}

	return 0;
}

