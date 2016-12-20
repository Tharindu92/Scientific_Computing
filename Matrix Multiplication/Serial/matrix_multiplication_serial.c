#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float matrixMul_float(float vector1[], float vector2[], long size);
double matrixMul_double(double vector1[], double vector2[], long size);
double doubleGen();
float floatGen();
int main(){
	long sizes[4] = {600, 1200, 1800,3};
	int i,j;
	long size;
	clock_t begin, end;
	double time_spent;
	float sum1;
	double sum2;
	FILE *f = fopen("serial_results.txt", "ab+");
	srand(time(NULL));
	for(i=0; i < 4; i++){
		size = sizes[i]*sizes[i];
		float *vector1;
		vector1 = (float*) malloc(size * sizeof(float));
		float *vector2;
		vector2 = (float*) malloc(size * sizeof(float));
		for(j=0; j < size; j++){
			*(vector1+j) = floatGen();
			*(vector2+j) = floatGen();
		}
		printf("===================================================================\n");
		fprintf(f, "===================================================================\n");
		printf("\tVector Initialization is completed\n");
		begin = clock();
		sum1 = matrixMul_float(vector1,vector2,sizes[i]);
		end = clock();
		time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;
		printf("Vector size : %ld\n",size);
		printf("Dot product : %f\n", sum1);
		printf("Single Precision Time Spent : %lf\n\n",time_spent);
		fprintf(f,"Vector size : %ld\n",size);
		fprintf(f,"Dot product : %f", sum1);
		fprintf(f,"Single Precision Time Spent : %lf\n\n",time_spent);
		fprintf(f, "===================================================================\n");
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
		begin = clock();
		sum2 = matrixMul_double(vector3,vector4,sizes[i]);
		end = clock();
		time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;
		printf("Vector size : %ld\n",size);
		printf("Dot product : %lf\n", sum2);
		printf("Double Precision Time Spent : %lf\n\n",time_spent);
		fprintf(f,"Vector size : %ld\n",size);
		fprintf(f,"Dot product : %lf\n", sum2);
		fprintf(f,"Double Precision Time Spent : %lf\n\n",time_spent);
		free(vector3);
		free(vector4);
	}
	fclose(f);
}

float matrixMul_float(float* vector1, float* vector2, long size){
	float sum = 0.0;
	float val1, val2, product;
	int i,j,k;
	for(i=0; i < size; i++){
		for(j=0; j < size; j++){
			sum = 0;
			for(k=0; k< size; k++){
				val1 = *(vector1+(i*size+k));
				val2 = *(vector2+(k*size+j));
				product = val1 * val2;
				sum += product;
			}
			//*(vector3+(i*size+j)) = sum;
			if(size < 5){
				printf("%f  ",sum);
			}
		}
		if(size < 5){
			printf("\n");
		}
	}
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
	return sum;
}
//i*size + j
double matrixMul_double(double* vector1, double* vector2, long size){
	double sum = 0.0;
	double val1, val2, product;
	int i,j,k;
	for(i=0; i < size; i++){
		for(j=0; j < size; j++){
			sum = 0;
			for(k=0; k< size; k++){
				val1 = *(vector1+(i*size+k));
				val2 = *(vector2+(k*size+j));
				product = val1 * val2;
				sum += product;
			}
			//*(vector3+(i*size+j)) = sum;
			if(size < 5){
				printf("%lf  ",sum);
			}
		}
		if(size < 5){
			printf("\n");
		}
	}
	printf("First value %lf", *vector1);
	return sum;
}

float floatGen(){
	float num;
	num = rand() % 2;
	num /= (rand() % 10 + 1);
	num++;
	return num;
}

double doubleGen(){
	double num;
	num = rand() % 2;
	num /= (rand() % 10 + 1);
	num++;
	return num;	
}