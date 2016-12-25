README
120095L P.M.T de Silva

Follow the given instructions to compile and execute the source code.

1. Matrix Multilplication

File Name 		: matrix_multiplication.cu

Compile Command : nvcc -Xcompiler -fopenmp -o matrix_multiplication matrix_multiplication.cu -O3

Run Commands 
	1. Run serial only (vector size 1000) : ./matrix_multiplication -s -n 1000
	2. Run parallel only with 4 threads (vector size 1000) : ./matrix_multiplication -p 4 -n 1000
	3. Run cuda only (vector size 1000) : ./matrix_multiplication -c -n 1000
	4. Verify cuda and parallel (vector size 1000) : ./matrix_multiplication -p 4 -c -v -n 1000
	5. Run all (vector size 1000) : ./matrix_multiplication -p 4 -c -s -n 1000  

Special Note	: When running both single and double precision instances run

Close Command	: ctrl + c
