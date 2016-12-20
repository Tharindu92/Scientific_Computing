#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>
#include <unistd.h>

float dotProduct_float_serial(float vector1[], float vector2[], long size);
float dotProduct_float_parallel(float vector1[], float vector2[], long size, int thread_count);
double dotProduct_double_serial(double vector1[], double vector2[], long size);
double dotProduct_double_parallel(double vector1[], double vector2[], long size, int thread_count);
double doubleGen();
float floatGen();
int operations(long sizes[3], int parallel, int serial, int cuda, int verify, int thread_count);
int print_results_float(FILE *f, long size, float sum1, double time_spent);
int print_results_double(FILE *f, long size, double sum1, double time_spent);

int main(int argc, char **argv){
	int parallel = 0;
	int serial = 0;
	int cuda = 0;
	int verify = 0;
	int thread_count = 2;
	int c;
	while((c = getopt(argc, argv, "scp:v")) != -1){
		switch(c){
			case 'p':
				parallel = 1;
				thread_count = atoi(optarg);
				if(thread_count > 8 || thread_count < 2){
					printf("Invalid Number of threads\nThread number is set to 2\n");
					thread_count = 2;
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
	long sizes[3] = {100000000, 500000000, 1000000000};

	operations(sizes, parallel, serial, cuda, verify, thread_count);
}

int operations(long sizes[3], int parallel, int serial, int cuda, int verify, int thread_count){
	int i,j;
	long size;
	clock_t begin, end;
	double time_spent_serial = -1, time_spent_parallel, time_spent_cuda;
	float sum1;
	double sum2;
	FILE *f1 = fopen("serial_results.txt", "ab+");
	FILE *f2 = fopen("parallel_results.txt", "ab+");
	FILE *f3 = fopen("cuda_results.txt", "ab+");
	srand(time(NULL));
	for(i=0; i < 3; i++){
		size = sizes[i];
		float *vector1;
		vector1 = (float*) malloc(size * sizeof(float));
		float *vector2;
		vector2 = (float*) malloc(size * sizeof(float));
		for(j=0; j < size; j++){
			*(vector1+j) = floatGen();
			*(vector2+j) = floatGen();
		}
		printf("===================================================================\n");
		fprintf(f1, "===================================================================\n");
		printf("\tVector Initialization is completed\n");
		if(serial || verify){
			begin = clock();
			sum1 = dotProduct_float_serial(vector1,vector2,size);
			end = clock();
			time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
			print_results_float(f1, size, sum1, time_spent_serial);
		}

		if(parallel){
			begin = clock();
			sum1 = dotProduct_float_parallel(vector1,vector2,size,thread_count);
			end = clock();
			time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
			print_results_float(f2, size, sum1, time_spent_parallel);
		}

		if(cuda){
			printf("Cuda need to implement\n");
		}

		if(verify){
			printf("===============================Single Precision====================================\n");
			if(parallel){
				if(abs(time_spent_serial - time_spent_parallel) > 0.1 ){
					printf("======================Paralle vs Serial=================================\n");
					printf("Significant difference between parallal with %d threads and serial\n", thread_count);
					printf("Time spent for parallel OMP threads with thread count of %d : %lf\n", thread_count, time_spent_parallel);
				}else{
					printf("======================Paralle vs Serial=================================\n");
					printf("No significant difference between parallal with %d threads and serial\n", thread_count);
					printf("Time spent for parallel OMP threads with thread count of %d : %lf\n", thread_count, time_spent_parallel);
				}	
			}

			if(cuda){
				if(abs(time_spent_serial - time_spent_cuda) > 0.1 ){
					printf("======================Cuda vs Serial=================================\n");
					printf("Significant difference between cuda and serial\n");
					printf("Time spent for cuda : %lf\n", time_spent_cuda);
				}else{
					printf("======================Cuda vs Serial=================================\n");
					printf("No significant difference between cuda and serial\n");
					printf("Time spent for cuda : %lf\n", time_spent_cuda);
				}	
			}

			printf("Time spent for serial : %lf\n",time_spent_serial);
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
			begin = clock();		
			sum2 = dotProduct_double_serial(vector3,vector4,size);
			end = clock();
			time_spent_serial = (double)(end - begin)/ CLOCKS_PER_SEC;
			print_results_double(f1, size, sum2, time_spent_serial);
		}

		if(parallel){
			begin = clock();		
			sum2 = dotProduct_double_parallel(vector3,vector4,size,thread_count);
			end = clock();
			time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
			print_results_double(f1, size, sum2, time_spent_parallel);
		}

		if(cuda){
			printf("Cuda need to implement\n");
		}

		if(verify){
			printf("===============================Double Precision====================================\n");
			if(parallel){
				if(abs(time_spent_serial - time_spent_parallel) > 0.1 ){
					printf("======================Paralle vs Serial=================================\n");
					printf("Significant difference between parallal with %d threads and serial\n", thread_count);
					printf("Time spent for parallel OMP threads with thread count of %d : %lf\n", thread_count, time_spent_parallel);
				}else{
					printf("======================Paralle vs Serial=================================\n");
					printf("No significant difference between parallal with %d threads and serial\n", thread_count);
					printf("Time spent for parallel OMP threads with thread count of %d : %lf\n", thread_count, time_spent_parallel);
				}	
			}

			if(cuda){
				if(abs(time_spent_serial - time_spent_cuda) > 0.1 ){
					printf("======================Cuda vs Serial=================================\n");
					printf("Significant difference between cuda and serial\n");
					printf("Time spent for cuda : %lf\n", time_spent_cuda);
				}else{
					printf("======================Cuda vs Serial=================================\n");
					printf("No significant difference between cuda and serial\n");
					printf("Time spent for cuda : %lf\n", time_spent_cuda);
				}	
			}
			printf("Time spent for serial : %lf\n",time_spent_serial);
		}
		free(vector3);
		free(vector4);
	}
	fclose(f1);
	fclose(f2);
	fclose(f3);
}

int print_results_float(FILE *f, long size, float sum1, double time_spent){
	//printf("Vector size : %ld\n",size);
	//printf("Dot product : %f\n", sum1);
	//printf("Single Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f,"Vector size : %ld\n",size);
	fprintf(f,"Dot product : %f", sum1);
	fprintf(f,"Single Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f, "===================================================================\n");
}

int print_results_double(FILE *f, long size, double sum1, double time_spent){
	//printf("Vector size : %ld\n",size);
	//printf("Dot product : %lf\n", sum1);
	//printf("Double Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f,"Vector size : %ld\n",size);
	fprintf(f,"Dot product : %lf", sum1);
	fprintf(f,"Double Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f, "===================================================================\n");
}

float dotProduct_float_serial(float* vector1, float* vector2, long size){
	float sum = 0.0;
	int i;
	for(i=0; i < size; i++){
		sum += (*(vector1+i)) * (*(vector2+i));
	}
	return sum;
}

float dotProduct_float_parallel(float* vector1, float* vector2, long size, int thread_count){
	float sum = 0.0;
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


double dotProduct_double_serial(double* vector1, double* vector2, long size){
	double sum = 0.0;
	int i;
	for(i=0; i < size; i++){
		sum += (*(vector1+i)) * (*(vector2+i));
	}
	return sum;
}

double dotProduct_double_parallel(double* vector1, double* vector2, long size, int thread_count){
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