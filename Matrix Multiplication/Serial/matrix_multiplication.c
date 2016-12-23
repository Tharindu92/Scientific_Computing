#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>
#include <unistd.h>

#define print(x) printf("  %d  ",x)

float *matrixMul_float_serial(float vector1[], float vector2[], int size);
float *matrixMul_float_parallel(float vector1[], float vector2[], int size, int thread_count);
double *matrixMul_double_serial(double vector1[], double vector2[], int size);
double *matrixMul_double_parallel(double vector1[], double vector2[], int size, int thread_count);
double doubleGen();
float floatGen();
int operations(int size, int parallel, int serial, int cuda, int verify, int thread_count);
int print_results_float(FILE *f, int size, double time_spent);
int print_results_double(FILE *f, int size, double time_spent);
double verifyVectord(double *vector1, double *vector2, int size);
float verifyVectorf(float *vector1, float *vector2, int size);

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

int operations(int size, int parallel, int serial, int cuda, int verify, int thread_count){
	int i,j;
	int t_size;
	clock_t begin, end;
	double time_spent_serial, time_spent_parallel, time_spent_cuda;
	float sum1;
	double sum2;
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
		printf("first element : %f\n",ans_fserial[0]);
		printf("last element : %f\n",ans_fserial[t_size -1]);
	}

	if(parallel){
		begin = clock();
		ans_fparallel = matrixMul_float_parallel(vector1,vector2,size,thread_count);
		end = clock();
		time_spent_parallel = (double)(end - begin)/ CLOCKS_PER_SEC;
		print_results_float(f2, size, time_spent_parallel);
		printf("first element : %f\n",ans_fparallel[0]);
		printf("last element : %f\n",ans_fparallel[t_size -1]);
	}

	if(cuda){
		printf("Cuda need to implement\n");
	}

	if(verify){
		float error_p = verifyVectorf(ans_fserial, ans_fparallel, t_size);
		double confident = 4*size / 10000;
		printf("===============================Single Precision====================================\n");
		if(parallel){
			
			if(error_p > confident ){
				printf("======================Paralle vs Serial=================================\n");
				printf("Significant difference between parallal with %d threads and serial\n", thread_count);
			}else{
				printf("======================Paralle vs Serial=================================\n");
				printf("No significant difference between parallal with %d threads and serial\n", thread_count);
			}	
			printf("Error : %f\n",error_p);
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
	print(156);
	free(vector1); print(157);
	free(vector2); print(158);
	free(ans_fserial); print(159);
	free(ans_fparallel); print(160);
	//free(ans_fcuda);
	double *vector3;
	vector3 = (double*) malloc(t_size * sizeof(double));	print(161);
	double *vector4;
	vector4 = (double*) malloc(t_size * sizeof(double));	print(163);
	double *ans_dserial;
	//ans_fserial = (float*) malloc(t_size * sizeof(float));
	double *ans_dparallel;
	//ans_fparallel = (float*) malloc(t_size * sizeof(float));
	double *ans_dcuda;
	//ans_fcuda = (float*) malloc(t_size * sizeof(float));
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
		printf("Cuda need to implement\n");
	}

	if(verify){
		double error_pd = verifyVectord(ans_dserial, ans_dparallel, t_size);
		double confident_d = 4*size / 10000;
		printf("===============================Double Precision====================================\n");
		if(parallel){
			if(error_pd > confident_d ){
				printf("======================Paralle vs Serial=================================\n");
				printf("Significant difference between parallal with %d threads and serial\n", thread_count);
			}else{
				printf("======================Paralle vs Serial=================================\n");
				printf("No significant difference between parallal with %d threads and serial\n", thread_count);
			}	
			printf("Error : %lf\n",error_pd);
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
	free(ans_dserial);
	free(ans_dparallel);
	//free(ans_dcuda);
	fclose(f1);
	fclose(f2);
	fclose(f3);
}

int print_results_float(FILE *f, int size, double time_spent){
	//printf("Vector size : %d\n",size);
	//printf("Dot product : %f\n", sum1);
	//printf("Single Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f,"Vector size : %d\n",size);
	//fprintf(f,"Dot product : %f", sum1);
	fprintf(f,"Single Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f, "===================================================================\n");
}

int print_results_double(FILE *f, int size, double time_spent){
	//printf("Vector size : %d\n",size);
	//printf("Dot product : %lf\n", sum1);
	//printf("Double Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f,"Vector size : %d\n",size);
	//fprintf(f,"Dot product : %lf", sum1);
	fprintf(f,"Double Precision Time Spent : %lf\n\n",time_spent);
	fprintf(f, "===================================================================\n");
}

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
	/*
	if(size < 5){
		printf("%s\n", "Vector 1");
		for(i=0; i<size; i++){
			for(j=0; j<size; j++){
				printf("%f  ",*(vector1+(i*size+j)));
			}
			printf("\n");
		}
		printf("%s\n", "Vector 2");
		for(i=0; i<size; i++){
			for(j=0; j<size; j++){
				printf("%f  ",*(vector1+(i*size+j)));
			}
			printf("\n");
		}
		
	}
	printf("First value %f\n", *vector1);
	*/
	return ans;
}

//i*size + j
double *matrixMul_double_serial(double* vector1, double* vector2, int size){
	double sum = 0.0;
	double val1, val2, product;
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
	double val1, val2, product;
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
