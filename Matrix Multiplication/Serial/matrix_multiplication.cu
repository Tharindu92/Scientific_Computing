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
void print_results_float(FILE *f, int size, double time_spent);
void print_results_double(FILE *f, int size, double time_spent);
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
	/*
		0 1 2  0 1 2   3x0+4x3+5x6	idx = 3   
		3 4 5  3 4 5   3x1+4x4+5x7	idx = 4
		6 7 8  6 7 8	
	*/
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
	FILE *f1 = fopen("serial_results.txt", "ab+");
	FILE *f2 = fopen("parallel_results.txt", "ab+");
	FILE *f3 = fopen("cuda_results.txt", "ab+");
	srand(time(NULL));
	t_size = size*size;
	float *vector1;
	vector1 = (float*) malloc(t_size * sizeof(float));
	float *vector2;
	vector2 = (float*) malloc(t_size * sizeof(float));
	float *ans_fserial;
	//ans_fserial = (float*) malloc(t_size * sizeof(float));
	float *ans_fparallel;
	//ans_fparallel = (float*) malloc(t_size * sizeof(float));
	float *ans_fcuda;
	//ans_fcuda = (float*) malloc(t_size * sizeof(float));
	for(j=0; j < t_size; j++){
		*(vector1+j) = floatGen();
		*(vector2+j) = floatGen();
	}
	printf("===================================================================\n");
	fprintf(f1, "===================================================================\n");
	printf("\tVector Initialization is completed\n");
	if(serial || verify){
		begin = clock();
		ans_fserial = matrixMul_float_serial(vector1,vector2,size);
		end = clock();
		time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float(f1, size, time_spent_serial);
	}

	if(parallel){
		begin = clock();
		ans_fparallel = matrixMul_float_parallel(vector1,vector2,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float(f2, size, time_spent_parallel);
	}
	//print(148);
	if(cuda){
		printf("Run CUDA\n");
		begin = clock(); //print(151);
		ans_fcuda = matrixMul_float_cuda(vector1,vector2,size);
		end = clock();
		time_spent_cuda = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float(f3, size, time_spent_cuda);
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
	vector3 = (double*) malloc(t_size * sizeof(double));	//print(161);
	double *vector4;
	vector4 = (double*) malloc(t_size * sizeof(double));	//print(163);
	double *ans_dserial;
	//ans_fserial = (float*) malloc(t_size * sizeof(float));
	double *ans_dparallel;
	//ans_fparallel = (float*) malloc(t_size * sizeof(float));
	double *ans_dcuda;
	//ans_dcuda = (double*) malloc(t_size * sizeof(double));
	for(j=0; j < t_size; j++){
		*(vector3+j) = doubleGen();
		*(vector4+j) = doubleGen();
	}
	if(serial || verify){
		begin = clock();		
		ans_dserial = matrixMul_double_serial(vector3,vector4,size);
		end = clock();
		time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double(f1, size, time_spent_serial);		
	}

	if(parallel){
		begin = clock();		
		ans_dparallel = matrixMul_double_parallel(vector3,vector4,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double(f2, size, time_spent_parallel);
	}

	if(cuda){
		printf("Run CUDA\n");
		begin = clock();
		ans_dcuda = matrixMul_double_cuda(vector3,vector4,size);
		end = clock();
		time_spent_cuda = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_double(f3, size, time_spent_cuda);
	}
	//print(236);
	if(verify){
		//printf(238);
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
	fclose(f1);
	fclose(f2);
	fclose(f3);
}

void print_results_float(FILE *f, int size, double time_spent){
	printf("Single Precision Time Spent : %lf\n\n",time_spent);		
	fprintf(f,"Vector size : %d\n",size);
	fprintf(f,"Single Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f, "===================================================================\n");
}

void print_results_double(FILE *f, int size, double time_spent){
	printf("Double Precision Time Spent : %lf\n\n",time_spent);		
	fprintf(f,"Vector size : %d\n",size);
	fprintf(f,"Double Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f, "===================================================================\n");
}

float verifyVectorf(float *vector1, float *vector2, int size){
	//print(293);
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
	//print(377);
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);//print(378);
	// Cleanup
	//free(sumHost); 
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);
	return sumHost;
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
	//printf("First value %lf", *vector1);
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
	
	//printf("First value %lf", *vector1);
	return ans;
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
	//free(sumHost); 
	cudaFree(sumDev);
	cudaFree(vector1_device);
	cudaFree(vector2_device);
	return sumHost;
}

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
