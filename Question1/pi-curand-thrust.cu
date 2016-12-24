// Source: http://docs.nvidia.com/cuda/curand/index.html

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>
#include <getopt.h>

// we could vary M & N to find the perf sweet spot

struct estimate_pi : 
    public thrust::unary_function<unsigned int, float>
{
  __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 10000; // samples per thread

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
    unsigned int N = 10000; // samples per thread

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
  // use 30K independent seeds
  int M = 30000;
  if(!dp){
	  float estimate = thrust::transform_reduce(
	          thrust::counting_iterator<int>(0),
	          thrust::counting_iterator<int>(M),
	          estimate_pi(),
	          0.0f,
	          thrust::plus<float>());
	    estimate /= M;

	    std::cout << std::setprecision(4);
	    std::cout << "pi is approximately ";
	    std::cout << estimate << std::endl;
  }else{
	  double estimate = thrust::transform_reduce(
	          thrust::counting_iterator<int>(0),
	          thrust::counting_iterator<int>(M),
	          estimate_pi_d(),
	          0.0f,
	          thrust::plus<double>());
	    estimate /= M;

	    std::cout << std::setprecision(4);
	    std::cout << "pi is approximately ";
	    std::cout << estimate << std::endl;
  }

  return 0;
}

