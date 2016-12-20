#include <stdio.h>
#include <stdlib.h>

int main(){
	long size = 100;
	float *pointer,num;
	pointer = (float*) malloc(size * sizeof(float));
	srand(time(NULL));
	int i;
	for(i=0; i<size; i++){
		num = rand() % 2;
		num /= rand() % 10;
		num++;
		*(pointer + i) = num;
	}
	for(i=0; i<size; i++){
		printf("Index : %d\t Value : %f",i,*(pointer+i));
	}
	free(pointer);
}