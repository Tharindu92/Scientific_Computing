README
120095L P.M.T de Silva

Follow the given instructions to compile and execute the source code.

Q1)

b. Compare pi-mystery and pi-curand 

File Name 		: pi-mystery-vs-crand.cu

Compile Command : nvcc -o pi-mystery-vs-crand pi-mystery-vs-crand.cu -O3

Run Commands 
	1. Run single precision : ./pi-mystery-vs-crand 
	2. Run double precision : ./pi-mystery-vs-crand -d

c. Parallel Computation of pi-curand

File Name 		: para_pi_curand.cu

Compile Command : nvcc -Xcompiler -fopenmp -o para_pi_curand para_pi_curand.cu -O3

Run Commands 
	1. Run single precision with 4 threads: ./para_pi_curand -n 4
	2. Run double precision with 4 threads: ./para_pi_curand -n 4 -d

Special Note : No arguments means run single precision with 2 threads

d. Compare pi-myrand and pi-curand

File Name 		: pi-myrand-vs-crand.cu

Compile Command : nvcc -o pi-myrand-vs-crand pi-myrand-vs-crand.cu -O3

Run Commands 
	1. Run single precision : ./pi-myrand-vs-crand
	2. Run double precision : ./pi-myrand-vs-crand -d

e. Compare pi-curand-thrust and pi-curand

File Name 		: pi-thrust-vs-curand.cu

Compile Command : nvcc -o pi-thrust-vs-curand pi-thrust-vs-curand.cu -O3

Run Commands 
	1. Run single precision with M = 1000: ./pi-thrust-vs-curand -n 1000
	2. Run double precision with M = 1000: ./pi-thrust-vs-curand -d -n 1000

Special Note : No arguments means run single precision with M = 32768

f. Modified as to select single precision or double precision

	i)		File Name 		: pi-mystery_modified.cu

			Compile Command : nvcc -o pi-mystery_modified pi-mystery_modified.cu -O3

			Run Commands 
				1. Run single precision : ./pi-mystery_modified
				2. Run double precision : ./pi-mystery_modified -d

	ii) 	File Name 		: pi-curand_modified.cu

			Compile Command : nvcc -o pi-curand_modified pi-curand_modified.cu -O3

			Run Commands 
				1. Run single precision : ./pi-curand_modified
				2. Run double precision : ./pi-curand_modified -d

	iii)	File Name 		: pi-myrand_modified.cu

			Compile Command : nvcc -o pi-myrand_modified pi-myrand_modified.cu -O3

			Run Commands 
				1. Run single precision : ./pi-myrand_modified
				2. Run double precision : ./pi-myrand_modified -d

	iv)		File Name 		: pi-curand-thrust_modified.cu

			Compile Command : nvcc -o pi-curand-thrust_modified pi-curand-thrust_modified.cu -O3

			Run Commands 
				1. Run single precision : ./pi-curand-thrust_modified
				2. Run double precision : ./pi-curand-thrust_modified -d
