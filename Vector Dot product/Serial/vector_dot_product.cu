#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda.h>

#define NUM_THREAD  256  // Number of thread blocks
#define print(x) printf("%d",x)

/*
 * Define function parameters
 */
float dotProduct_float_serial(float vector1[], float vector2[], int size);
float dotProduct_float_parallel(float vector1[], float vector2[], int size, int thread_count);
float dotProduct_float_cuda(float* vector1, float* vector2, int num);
double dotProduct_double_serial(double vector1[], double vector2[], int size);
double dotProduct_double_parallel(double vector1[], double vector2[], int size, int thread_count);
double dotProduct_double_cuda(double* vector1, double* vector2, int num);
double doubleGen();
float floatGen();
int operations(int size, int parallel, int serial, int cuda, int verify, int thread_count);
void print_results_float( int size, float sum1, double time_spent);
void print_results_double( int size, double sum1, double time_spent);

//Functions run on GPU
__global__ void dotProduct_CUDA_double(double *sum, int size, double *vector1, double *vector2){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	if(idx < size){
		sum[idx] = (vector2[idx]) * (vector1[idx]);
	}
}

__global__ void dotProduct_CUDA_float(float *sum, int size, float *vector1, float *vector2){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	if(idx < size){
		sum[idx] = (vector2[idx]) * (vector1[idx]);
	}
}

int main(int argc, char **argv){
	int parallel = 0;
	int serial = 0;
	int cuda = 0;
	int verify = 0;
	int thread_count = 2;
	int c;
	int size = 10000000;
	while((c = getopt(argc, argv, "scp:vn:")) != -1){
		switch(c){
			case 'p':
				parallel = 1;
				thread_count = atoi(optarg);
				if(thread_count > 8 || thread_count < 2){
					printf("Invalid Number of threads\nThread number is set to 2\n");
					thread_count = 2;
				}
				break;
			case 'n':
				size = atoi(optarg);
				if(size > 1000000000 || size < 1){
					printf("Invalid Number of threads\nThread number is set to 2\n");
					size = 10000000;
				}
				break;
			case 's':
				serial = 1;
				break;
			case 'c':
				cuda = 1;
				break;
			case 'v':
				verify = 1;
				break;
			case '?':
				if(optopt == 'p'){
					printf("Number of threads missing\nThread number set to 8");
					thread_count = 8;
				}else{
				printf("Unknown option selected\nProgram exited\n");
				return 0;
				}
				break;
			default:
				abort();

		}
	}

	operations(size, parallel, serial, cuda, verify, thread_count);
}

int operations(int size, int parallel, int serial, int cuda, int verify, int thread_count){
	int j;
	clock_t begin, end;
	double time_spent_serial, time_spent_parallel, time_spent_cuda;
	float sum1_serial,sum1_parallel,sum1_cuda;
	double sum2_serial,sum2_parallel,sum2_cuda;
	srand(time(NULL));
	
	float *vector1;
	vector1 = (float*) malloc(size * sizeof(float));
	float *vector2;
	vector2 = (float*) malloc(size * sizeof(float));
	for(j=0; j < size; j++){
		*(vector1+j) = floatGen();
		*(vector2+j) = floatGen();
	}
	printf("===================================================================\n");
	//fprintf( "===================================================================\n");
	printf("\tVector Initialization is completed\n");
	if(serial || verify){
		printf("Run Serial\n");
		begin = clock();
		sum1_serial = dotProduct_float_serial(vector1,vector2,size);
		end = clock();
		time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, sum1_serial, time_spent_serial);
	}

	if(parallel){
		printf("Run Parallel\n");
		begin = clock();
		sum1_parallel = dotProduct_float_parallel(vector1,vector2,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, sum1_parallel, time_spent_parallel);
	}

	if(cuda){
		printf("Run CUDA\n");
		begin = clock();
		sum1_cuda = dotProduct_float_cuda(vector1,vector2,size);
		end = clock();
		time_spent_cuda = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, sum1_parallel, time_spent_cuda);
	}

	if(verify){
		printf("===============================Single Precision====================================\n");
		double confident = sum1_serial/10000;
		if(parallel){
			if(abs(sum1_serial - sum1_parallel) > confident ){
				printf("======================Paralle vs Serial=================================\n");
				printf("Significant difference between parallal with %d threads and serial\n", thread_count);
				
			}else{
				printf("======================Paralle vs Serial=================================\n");
				printf("No significant difference between parallal with %d threads and serial\n", thread_count);
			}	
			printf("Answer for parallel OMP threads with thread count of %d : %f\n", thread_count, sum1_parallel);
		}

		if(cuda){
			if(abs(sum1_serial - sum1_cuda) > confident ){
				printf("======================Cuda vs Serial=================================\n");
				printf("Significant difference between cuda and serial\n");
			}else{
				printf("======================Cuda vs Serial=================================\n");
				printf("No significant difference between cuda and serial\n");
			}
			printf("Answer for cuda : %f\n", sum1_cuda);	
		}

		printf("Answer for serial : %f\n",sum1_serial);
	}

	free(vector1);
	free(vector2);
	double *vector3;
	vector3 = (double*) malloc(size * sizeof(double));
	double *vector4;
	vector4 = (double*) malloc(size * sizeof(double));
	for(j=0; j < size; j++){
		*(vector3+j) = doubleGen();
		*(vector4+j) = doubleGen();
	}
	if(serial || verify){
		printf("Run Serial\n");
		begin = clock();		
		sum2_serial = dotProduct_double_serial(vector3,vector4,size);
		end = clock();
		time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double( size, sum2_serial, time_spent_serial);
	}

	if(parallel){
		printf("Run Parallel\n");
		begin = clock();		
		sum2_parallel = dotProduct_double_parallel(vector3,vector4,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double( size, sum2_parallel, time_spent_parallel);
	}

	if(cuda){
		printf("Run CUDA\n");
		begin = clock();
		sum2_cuda = dotProduct_double_cuda(vector3,vector4,size);
		end = clock();
		time_spent_cuda = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, sum1_parallel, time_spent_cuda);
	}

	if(verify){
		printf("===============================Double Precision====================================\n");
		double confident = sum1_serial/10000;	//1% of error
		if(parallel){
			if(abs(sum2_serial - sum2_parallel) > confident ){
				printf("======================Paralle vs Serial=================================\n");
				printf("Significant difference between parallal with %d threads and serial\n", thread_count);
				
			}else{
				printf("======================Paralle vs Serial=================================\n");
				printf("No significant difference between parallal with %d threads and serial\n", thread_count);
			}	
			printf("Answer for parallel OMP threads with thread count of %d : %f\n", thread_count, sum2_parallel);
		}

		if(cuda){
			if(abs(sum2_serial - sum2_cuda) > confident ){
				printf("======================Cuda vs Serial=================================\n");
				printf("Significant difference between cuda and serial\n");
			}else{
				printf("======================Cuda vs Serial=================================\n");
				printf("No significant difference between cuda and serial\n");
			}
			printf("Answer for cuda : %lf\n", sum2_cuda);	
		}

		printf("Answer for serial : %lf\n",sum2_serial);
	}
	free(vector3);
	free(vector4);
	return 1;
}

void print_results_float( int size, float sum1, double time_spent){
	printf("Single Precision Time Spent : %lf\n\n",time_spent);
}

void print_results_double( int size, double sum1, double time_spent){
	printf("Double Precision Time Spent : %lf\n\n",time_spent);
}

/*
 * vector dot product sequential
 */
float dotProduct_float_serial(float* vector1, float* vector2, int size){
	float sum = 0.0;
	int i;
	for(i=0; i < size; i++){
		sum += (*(vector1+i)) * (*(vector2+i));
	}
	return sum;
}

double dotProduct_double_serial(double* vector1, double* vector2, int size){
	double sum = 0.0;
	int i;
	for(i=0; i < size; i++){
		sum += (*(vector1+i)) * (*(vector2+i));
	}
	return sum;
}

/*
 * Vector dot product OMP threads
 */

float dotProduct_float_parallel(float* vector1, float* vector2, int size, int thread_count){
	float sum = 0.f;
	int i;
	#pragma omp parallel num_threads(thread_count)
	{
		#pragma omp for schedule(static) reduction(+:sum)
		for(i=0; i < size; i++){
			sum += (*(vector1+i)) * (*(vector2+i));
		}	
	}
	
	return sum;
}

double dotProduct_double_parallel(double* vector1, double* vector2, int size, int thread_count){
	double sum = 0.0;
	int i;
	#pragma omp parallel num_threads(thread_count)
	{
		#pragma omp for schedule(static) reduction(+:sum)
		for(i=0; i < size; i++){
			sum += (*(vector1+i)) * (*(vector2+i));
		}
	}
	return sum;
}

/*
 * Vector dot product CUDA
 */

float dotProduct_float_cuda(float* vector1, float* vector2, int num){
	int num_block = (num + NUM_THREAD - 1)/(NUM_THREAD);
	size_t size = num*sizeof(float);  //Array memory size.
	float *sumHost, *sumDev;  // Pointer to host & device arrays
	float *vector1_device;
	float *vector2_device;
	float dotProduct = 0;
	int tid;	
	sumHost = (float *)malloc(size); //  Allocate array on host
	
	cudaMalloc((void **) &sumDev, size); // Allocate array on device
	cudaMalloc((void **) &vector1_device, size);  // Allocate array on device
	cudaMalloc((void **) &vector2_device, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	cudaMemcpy(vector1_device, vector1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(vector2_device, vector2, size, cudaMemcpyHostToDevice);

	// Do calculation on device
	dotProduct_CUDA_float <<<num_block, NUM_THREAD>>> (sumDev, num, vector1_device, vector2_device); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<num; tid++){
		dotProduct += sumHost[tid];
	}	
	// Cleanup
	free(sumHost); 
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);
	return dotProduct;
}

double dotProduct_double_cuda(double* vector1, double* vector2, int num){
	int num_block = (num + NUM_THREAD - 1)/(NUM_THREAD);
	size_t size = num*sizeof(double);  //Array memory size.
	double *sumHost, *sumDev;  // Pointer to host & device arrays
	double *vector1_device;
	double *vector2_device;
	double dotProduct = 0;
	int tid;	
	sumHost = (double *)malloc(size); //  Allocate array on host
	
	cudaMalloc((void **) &sumDev, size); // Allocate array on device
	cudaMalloc((void **) &vector1_device, size);  // Allocate array on device
	cudaMalloc((void **) &vector2_device, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	cudaMemcpy(vector1_device, vector1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(vector2_device, vector2, size, cudaMemcpyHostToDevice);

	// Do calculation on device
	dotProduct_CUDA_double <<<num_block, NUM_THREAD>>> (sumDev, num, vector1_device, vector2_device); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<num; tid++){
		dotProduct += sumHost[tid];
	}
	// Cleanup
	free(sumHost); 
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);
	return dotProduct;
}

/*
 * Random Number generation between 1 - 2
 */
float floatGen(){
	float num ;
	num = 1.0 * random() / RAND_MAX + 1.0;
	return num;
}

double doubleGen(){
	double num;
	num = 1.0 * random() / RAND_MAX + 1.0;
	return num;	
}
