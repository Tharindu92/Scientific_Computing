#include <stdio.h>
#include <time.h>

float dotProduct_float(float vector1[], float vector2[], long size);
double dotProduct_double(double vector1[], double vector2[], long size);

int main(){
	long sizes[3] = {100000000, 500000000, 1000000000};
	int i,j;
	long size;
	clock_t begin, end;
	double time_spent;
	float sum1;
	double sum2;
	FILE *f = fopen("serial_results.txt", ab+);
	srand(time(NULL));
	for(i=0; i < 3; i++){
		size = sizes[i];
		float vector1[size];
		float vector2[size];
		double vector3[size];
		double vector4[size];
		for(j=0; j < size; j++){
			vector1[j] = rand() % 2;
			vector2[j] = rand() % 2;
			vector3[j] = rand() % 2;
			vector4[j] = rand() % 2;
		}
		printf("===================================================================\n");
		fprintf(f, "===================================================================\n");
		printf("\tVector Initialization is completed\n");
		begin = clock();
		sum1 = dotProduct_float(vector1,vector2,size);
		end = clock();
		time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;
		printf("Vector size : %ld\n",size);
		printf("Dot product : %f\n", sum1);
		printf("Single Precision Time Spent : %lf\n\n",time_spent);
		fprintf(f,"Vector size : %ld\n",size);
		fprintf(f,"Dot product : %f", sum1);
		fprintf(f,"Single Precision Time Spent : %lf\n\n",time_spent);
		fprintf(f, "===================================================================\n");
		begin = clock();
		sum2 = dotProduct_double(vector3,vector4,size);
		end = clock();
		time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;
		printf("Vector size : %ld\n",size);
		printf("Dot product : %lf\n", sum2);
		printf("Single Precision Time Spent : %lf\n\n",time_spent);
		fprintf(f,"Vector size : %ld\n",size);
		fprintf(f,"Dot product : %lf\n", sum2);
		fprintf(f,"Single Precision Time Spent : %lf\n\n",time_spent);
	}
	fclose(f);
}

float dotProduct_float(float vector1[], float vector2[], long size){
	float sum = 0.0;
	int i;
	for(i=0; i < size; i++){
		sum += vector1[i]*vector2[i];
	}
	return sum;
}

double dotProduct_double(double vector1[], double vector2[], long size){
	double sum = 0.0;
	int i;
	for(i=0; i < size; i++){
		sum += vector1[i]*vector2[i];
	}
	return sum;
}