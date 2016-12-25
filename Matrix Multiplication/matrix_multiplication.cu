#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda.h>

#define NUM_THREAD  256  // Number of thread blocks
#define print(x) printf("%d",x)

float *matrixMul_float_serial(float vector1[], float vector2[], int size);
float *matrixMul_float_parallel(float vector1[], float vector2[], int size, int thread_count);
float *matrixMul_float_cuda(float* vector1, float* vector2, int num);
double *matrixMul_double_serial(double vector1[], double vector2[], int size);
double *matrixMul_double_parallel(double vector1[], double vector2[], int size, int thread_count);
double *matrixMul_double_cuda(double* vector1, double* vector2, int num);
double doubleGen();
float floatGen();
void operations(int size, int parallel, int serial, int cuda, int verify, int thread_count);
void print_results_float( int size, double time_spent);
void print_results_double( int size, double time_spent);
double verifyVectord(double *vector1, double *vector2, int size);
float verifyVectorf(float *vector1, float *vector2, int size);

__global__ void matMul_CUDA_double(double *sum, int size, double *vector1, double *vector2){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	int k;	
	if(idx < size*size){
		for(k=0; k< size; k++){
			sum[idx] += (*(vector1+(idx-(idx % size)+k))) * (*(vector2+(k*size+(idx % size))));
		}
	}
}

__global__ void matMul_CUDA_float(float *sum, int size, float *vector1, float *vector2){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
	int k;	
	if(idx < size*size){
		for(k=0; k< size; k++){
			sum[idx] += (*(vector1+(idx-(idx % size)+k))) * (*(vector2+(k*size+(idx % size))));
		}
	}
}

int main(int argc, char **argv){
	int parallel = 0;
	int serial = 0;
	int cuda = 0;
	int verify = 0;
	int thread_count = 2;
	int c;
	int size=10;
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
				if(size > 100000 || size < 1){
					size = 10;
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

void operations(int size, int parallel, int serial, int cuda, int verify, int thread_count){
	int j;
	int t_size;
	clock_t begin, end;
	double time_spent_serial, time_spent_parallel, time_spent_cuda;
	srand(time(NULL));
	t_size = size*size;
	float *vector1;
	vector1 = (float*) malloc(t_size * sizeof(float));
	float *vector2;
	vector2 = (float*) malloc(t_size * sizeof(float));
	float *ans_fserial;
	float *ans_fparallel;
	float *ans_fcuda;

	for(j=0; j < t_size; j++){
		*(vector1+j) = floatGen();
		*(vector2+j) = floatGen();
	}

	printf("===================================================================\n");
	printf("\tVector Initialization is completed\n");

	if(serial || verify){
		printf("Run Serial\n");
		begin = clock();
		ans_fserial = matrixMul_float_serial(vector1,vector2,size);
		end = clock();
		time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, time_spent_serial);
	}

	if(parallel){
		printf("Run Parallel\n");
		begin = clock();
		ans_fparallel = matrixMul_float_parallel(vector1,vector2,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, time_spent_parallel);
	}

	if(cuda){
		printf("Run CUDA\n");
		begin = clock();
		ans_fcuda = matrixMul_float_cuda(vector1,vector2,size);
		end = clock();
		time_spent_cuda = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float( size, time_spent_cuda);
	}

	if(verify){
		float error;
		double confident = 4*size / 10000;
		printf("===============================Single Precision====================================\n");
		if(parallel){
			error = verifyVectorf(ans_fserial, ans_fparallel, t_size);
			if(error > confident ){
				printf("======================Paralle vs Serial=================================\n");
				printf("Significant difference between parallal with %d threads and serial\n", thread_count);
			}else{
				printf("======================Paralle vs Serial=================================\n");
				printf("No significant difference between parallal with %d threads and serial\n", thread_count);
			}	
			printf("Error : %.20f\n",error);
		}

		if(cuda){
			error = verifyVectorf(ans_fserial, ans_fcuda, t_size);
			if(error > confident ){
				printf("======================Cuda vs Serial=================================\n");
				printf("Significant difference between cuda and serial\n");
			}else{
				printf("======================Cuda vs Serial=================================\n");
				printf("No significant difference between cuda and serial\n");
			}
			printf("Error : %.20f\n",error);	
		}
	}
	free(vector1); 
	free(vector2); 
	if(serial || verify)
		free(ans_fserial); 
	if(parallel)
		free(ans_fparallel);
	if(cuda)
		free(ans_fcuda);
	double *vector3;
	vector3 = (double*) malloc(t_size * sizeof(double));
	double *vector4;
	vector4 = (double*) malloc(t_size * sizeof(double));
	double *ans_dserial;
	double *ans_dparallel;
	double *ans_dcuda;

	for(j=0; j < t_size; j++){
		*(vector3+j) = doubleGen();
		*(vector4+j) = doubleGen();
	}

	if(serial || verify){
		printf("Run Serial\n");
		begin = clock();		
		ans_dserial = matrixMul_double_serial(vector3,vector4,size);
		end = clock();
		time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double( size, time_spent_serial);
	}

	if(parallel){
		printf("Run Parallel\n");
		begin = clock();		
		ans_dparallel = matrixMul_double_parallel(vector3,vector4,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double( size, time_spent_parallel);
	}

	if(cuda){
		printf("Run CUDA\n");
		begin = clock();
		ans_dcuda = matrixMul_double_cuda(vector3,vector4,size);
		end = clock();
		time_spent_cuda = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double( size, time_spent_cuda);
	}

	if(verify){
		double error_d;
		double confident_d = 4*size / 10000;
		
		printf("===============================Double Precision====================================\n");
		if(parallel){
			error_d = verifyVectord(ans_dserial, ans_dparallel, t_size);
			if(error_d > confident_d ){
				printf("======================Paralle vs Serial=================================\n");
				printf("Significant difference between parallal with %d threads and serial\n", thread_count);
			}else{
				printf("======================Paralle vs Serial=================================\n");
				printf("No significant difference between parallal with %d threads and serial\n", thread_count);
			}	
			printf("Error : %.20lf\n",error_d);
		}

		if(cuda){
			error_d = verifyVectord(ans_dserial, ans_dcuda, t_size);
			if(error_d > confident_d ){
				printf("======================Cuda vs Serial=================================\n");
				printf("Significant difference between cuda and serial\n");
			}else{
				printf("======================Cuda vs Serial=================================\n");
				printf("No significant difference between cuda and serial\n");
			}
			printf("Error : %.20lf\n",error_d);	
		}
	}
	free(vector3);
	free(vector4);
	if(serial || verify)	
		free(ans_dserial);
	if(parallel)
		free(ans_dparallel);
	if(cuda)	
		free(ans_dcuda);
}

void print_results_float( int size, double time_spent){
	printf("Single Precision Time Spent : %lf\n\n",time_spent);		
}

void print_results_double( int size, double time_spent){
	printf("Double Precision Time Spent : %lf\n\n",time_spent);		
}

/*
 * Verify the answer matrix
 */
float verifyVectorf(float *vector1, float *vector2, int size){
	float error = 0;
	int i;	
	for(i = 0; i<size; i++){
		error += abs(vector1[i] - vector2[i]);
	}
	error /= size;
	return error;
}

double verifyVectord(double *vector1, double *vector2, int size){
	double error = 0;
	int i;	
	for(i = 0; i<size; i++){
		error += abs(vector1[i] - vector2[i]);
	}
	error /= size;
	return error;
}

/*
 * Sequential Matrix Multiplication
 */
float *matrixMul_float_serial(float* vector1, float* vector2, int size){
	float sum = 0.0;
	int i,j,k;
	float* ans = (float*) malloc(size * size * sizeof(float)); 
	for(i=0; i < size; i++){
		for(j=0; j < size; j++){
			sum = 0;
			for(k=0; k< size; k++){
				sum += (*(vector1+(i*size+k))) * (*(vector2+(k*size+j)));
			}
			ans[i*size+j] = sum;
		}
	}
	return ans;
}

double *matrixMul_double_serial(double* vector1, double* vector2, int size){
	double sum = 0.0;
	int i,j,k;
	double* ans = (double*) malloc(size * size * sizeof(double));
	for(i=0; i < size; i++){
		for(j=0; j < size; j++){
			sum = 0;
			for(k=0; k< size; k++){
				sum += (*(vector1+(i*size+k))) * (*(vector2+(k*size+j)));
			}
			ans[i*size+j] = sum;
		}
	}
	return ans;
}

/*
 * OMP Thread parallel matrix multiplication
 */
float *matrixMul_float_parallel(float* vector1, float* vector2, int size, int thread_count){
	float sum = 0.0;
	int i,j,k;
	float* ans = (float*) malloc(size * size * sizeof(float));
	#pragma omp parallel num_threads(thread_count) private(i,j,k) shared(vector1, vector2, ans)
	{
		#pragma omp for schedule(static) reduction(+:sum)
		for(i=0; i < size; i++){
			for(j=0; j < size; j++){
				sum = 0;
				for(k=0; k< size; k++){
					sum = sum + (*(vector1+(i*size+k))) * (*(vector2+(k*size+j)));
				}
				ans[i*size+j] = sum;
			}
		}		
	}
	return ans;
}

double *matrixMul_double_parallel(double* vector1, double* vector2, int size, int thread_count){
	double sum = 0.0;
	int i,j,k;
	double* ans = (double*) malloc(size * size * sizeof(double));
	#pragma omp parallel num_threads(thread_count) private(i,j,k) shared(vector1, vector2, ans)
	{
		#pragma omp for schedule(static) reduction(+:sum)
		for(i=0; i < size; i++){
			for(j=0; j < size; j++){
				sum = 0;
				for(k=0; k< size; k++){
					sum = sum + (*(vector1+(i*size+k))) * (*(vector2+(k*size+j)));
				}
				ans[i*size+j] = sum;
			}
		}
	}

	return ans;
}
 
/*
 * CUDA GPU Matrix Multiplication
 */
float *matrixMul_float_cuda(float* vector1, float* vector2, int num){	
	int num_block = (num*num + NUM_THREAD - 1)/(NUM_THREAD); //print(358);
	size_t size = num*num*sizeof(float);  //Array memory size.
	float *sumHost, *sumDev;  // Pointer to host & device arrays
	float *vector1_device;
	float *vector2_device;
	sumHost = (float *)malloc(size); //  Allocate array on host

	cudaMalloc((void **) &sumDev, size);  // Allocate array on device
	cudaMalloc((void **) &vector1_device, size);  // Allocate array on device
	cudaMalloc((void **) &vector2_device, size); // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);//print(370);
	cudaMemcpy(vector1_device, vector1, size, cudaMemcpyHostToDevice); //print(371);
	cudaMemcpy(vector2_device, vector2, size, cudaMemcpyHostToDevice); //print(372);

	// Do calculation on device
	matMul_CUDA_float <<<num_block, NUM_THREAD>>> (sumDev, num, vector1_device, vector2_device); // call CUDA kernel
	// Retrieve result from device and store it in host array 
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);//print(378);
	// Cleanup
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);
	return sumHost;
}

double *matrixMul_double_cuda(double* vector1, double* vector2, int num){
	int num_block = (num*num + NUM_THREAD - 1)/(NUM_THREAD);
	size_t size = num*num*sizeof(double);  //Array memory size.
	double *sumHost, *sumDev;  // Pointer to host & device arrays
	double *vector1_device;
	double *vector2_device;
	sumHost = (double *)malloc(size); //  Allocate array on host
	
	cudaMalloc((void **) &sumDev, size); // Allocate array on device
	cudaMalloc((void **) &vector1_device, size);  // Allocate array on device
	cudaMalloc((void **) &vector2_device, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	cudaMemcpy(vector1_device, vector1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(vector2_device, vector2, size, cudaMemcpyHostToDevice);

	// Do calculation on device
	matMul_CUDA_double <<<num_block, NUM_THREAD>>> (sumDev, num, vector1_device, vector2_device); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	// Cleanup
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);
	return sumHost;
}

/*
 * Random Number generator
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
