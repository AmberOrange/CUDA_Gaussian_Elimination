#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <ctime>

#define MATRIX_SIZE 900
#define MATRIX_MAX_NUMBER 15
#define BLOCK_SIZE 100

typedef double matrix[MATRIX_SIZE+1][MATRIX_SIZE+1];

matrix A;
matrix A_GPUresult;
double b[MATRIX_SIZE];
//double b[MATRIX_SIZE];
//double y[MATRIX_SIZE];

__host__
void generateMatrix();

//__host__
//void generateVectors();

__host__
void printMatrix(matrix mat);

__host__
void solveOnCPU();

__host__
bool solveOnGPU();

__host__
void generateMatrix()
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE+1; j++)
		{
			if(i == j)
				A[i][j] = (double)(rand() % MATRIX_MAX_NUMBER) + 5.0;
			else
				A[i][j] = (double)(rand() % MATRIX_MAX_NUMBER) + 1.0;
		}
	}
}

//__host__
//void generateVectors()
//{
//	for (int i = 0; i < MATRIX_SIZE; i++)
//	{
//		b[i] = 2.0;
//		y[i] = 1.0;
//	}
//}

__host__
void printMatrix(matrix mat)
{
	for (int i = 0; i < MATRIX_SIZE + 1; i++)
	{
		std::cout << "[";
		for (int j = 0; j < MATRIX_SIZE; j++)
			std::cout << " " << mat[i][j] << ",";
		std::cout << " " << mat[i][MATRIX_SIZE] << " ]\n";
	}
}

__host__
void solveOnCPU()
{
	// k = Will need to be seperate kernel calls
	// The integer k would fit PERFECTLY as a symbol/constant
	// j = To be replaced with threads in parallell
	// i = Amount of work each thread needs to to per row

	// *********************
	// FIRST KERNEL : Solve the bottom triangle
	// *********************
	
	// k = The row being processed at the moment
	// Let's call it the "selected row"
	for (int k = 0; k < MATRIX_SIZE; k++)
	{
		// *********************
		// FIRST STEP : Dividing the selected row
		// *********************

		// temp = The leftmost element that we will be dividing the entire row with
		// If an entire MATRIX_SIZE can fit in a block, tempDiv does not have to be
		// allocated in Shared Memory. Otherwise, we got to.
		/*SHARED*/ double temp = A[k][k];
		// selectedRow = pointer to the selected row. This will be stored in Shared Memory
		// once implemented as parallell code.
		/*SHARED*/ double *selectedRow = A[k];
		// j = selecting each element on the row
		// j = k : No point processing elements before k; they're 0
		for (int j = k; j < MATRIX_SIZE + 1; j++)
		{
			selectedRow[j] /= temp;
		}
		// There is a big risk that A[k][k] doesn't become 1, which would be very troublesome

		// *********************
		// SECOND STEP : Subtract all other rows!
		// *********************

		// i = Row we want to do subtraction on at the moment
		// i = k + 1 : do all rows underneath the selected row
		for (int i = k + 1; i < MATRIX_SIZE; i++)
		{
			// temp = the leftmost element (that isn't a 0)
			temp = A[i][k];
			// j = selecting each element on the row
			for (int j = k; j < MATRIX_SIZE + 1; j++)
			{
				A[i][j] -= selectedRow[j] * temp;
			}
		}
	}
	// Now the bottom half the matrix is solved

	//printMatrix();

	// This is where sequential and parallell split up implementation wise
	// In parallell I'm intending on transposing the matrix in the first part
	// to make use of memory bursting.
	// In sequential, this does probably not speed up the implementation,
	// it would rather slow it down because of uncessesary memory writing.

	// *********************
	// SECOND KERNEL : Solve the top triangle
	// *********************

	// j = What column we're on
	// j = MATRIX_SIZE - 1 : to start from the column rightmost (not the vector column)
	// j > 0 : Don't do the one leftmost, it's just 1/1
	for (int j = MATRIX_SIZE - 1; j > 0; j--)
	{
		// i = What row we're on
		// i = MATRIX_SIZE - 2 : (MATRIX_SIZE - 1) to start from the row lowest down,
		// then an extra -1 because we already have the solution for the bottom row.
		for (int i = 0; i < j; i++)
		{
			/*std::cout << "i: " << i << ", j: " << j << std::endl;
			for (int ii = 0; ii < MATRIX_SIZE; ii++)
			{
				std::cout << "[ ";
				for (int jj = 0; jj < MATRIX_SIZE+1; jj++)
				{
					if (ii == i && jj == MATRIX_SIZE)
						std::cout << "T ";
					else if (ii == j && jj == MATRIX_SIZE)
						std::cout << "A ";
					else if (ii == i && jj == j)
						std::cout << "M ";
					else
						std::cout << "- ";
				}
				std::cout << "]\n";
			}
			std::cout << std::endl;*/
			A[i][MATRIX_SIZE] -= A[j][MATRIX_SIZE] * A[i][j];
		}
	}


	//std::cout << std::endl;
	//for (int i = 0; i < MATRIX_SIZE; i++)
	//{
	//	b[i] = A[i][MATRIX_SIZE];
	//	//std::cout << b[i] << ", ";
	//}
	//std::cout << std::endl;
}

int main()
{
	srand((unsigned int)time(NULL));
	int totalFail = 0;
	for (int j = 0; j < 1000; j++)
	{
		generateMatrix();
		//generateVectors();
		//printMatrix(A);
		//std::cout << std::endl << std::endl;
		solveOnGPU();
		//printMatrix(A_GPUresult);
		//std::cout << std::endl << std::endl;
		solveOnCPU();
		//printMatrix(A);


		int fail = 0;
		for (int i = 0; i < MATRIX_SIZE; i++)
		{
			if (abs(A_GPUresult[MATRIX_SIZE][i] - A[i][MATRIX_SIZE]) > 0.01)
			{
				//std::cout << "FAIL\n" << A_GPUresult[MATRIX_SIZE][i] << " : " << A[i][MATRIX_SIZE] << std::endl;
				fail++;
			}
		}
		//std::cout << "\nTotal Fail: " << fail << std::endl;
		if (fail != 0)
			std::cout << "@";
		else
			std::cout << ".";
		totalFail += fail;
	}
	std::cout << "\nAll together: " << totalFail;
	//std::cout << std::endl << std::endl
	//	<< "GPU: ";
	//for (int i = 0; i < MATRIX_SIZE; i++)
	//{
	//	std::cout << A_GPUresult[MATRIX_SIZE][i] << ", ";
	//}
	//std::cout << std::endl
	//	<< "CPU: ";
	//for (int i = 0; i < MATRIX_SIZE; i++)
	//{
	//	std::cout << A[i][MATRIX_SIZE] << ", ";
	//}



	//std::cout << std::endl << std::endl;
	//printMatrix();

	return 0;
}

__constant__ int k;

__global__
void gpuSolveBottom(matrix d_A)
{
	int j = (blockIdx.x * blockDim.x + threadIdx.x) + k;

	__shared__ double temp;
	temp = d_A[k][k];
	double selectedRow = d_A[k][j] / temp;

	__syncthreads();

	for (int i = k + 1; i < MATRIX_SIZE; i++)
	{
		temp = d_A[i][k];	// Load the entire thing directly?
		//d_A[i][j] = k;
		d_A[i][j] -= selectedRow * temp;
		__syncthreads();
	}

	d_A[j][k] = selectedRow;
	//d_A[k][j] = selectedRow;
}

__global__
void gpuSolveTop(matrix d_A)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x);

	for (int j = MATRIX_SIZE - 1; j > 0; j--)
	{
		if (i < j)
		{
			d_A[MATRIX_SIZE][i] -= d_A[MATRIX_SIZE][j] * d_A[j][i];
			__syncthreads();
		}
	}
}


__host__
bool solveOnGPU()
{
	cudaError_t cudaStatus;
	matrix* d_A;
	//int *d_k;
	int sizeOfMatrix = (MATRIX_SIZE + 1) * (MATRIX_SIZE + 1) * sizeof(double);


	cudaStatus = cudaMalloc((void**)&d_A, sizeOfMatrix);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed on d_A!\n";
		goto Error;
	}

	//cudaStatus = cudaMalloc((void**)&d_k, sizeof(int));
	//if (cudaStatus != cudaSuccess)
	//{
	//	std::cerr << "cudaMalloc failed on d_k!\n";
	//	goto Error;
	//}

	cudaStatus = cudaMemcpy(d_A, A, sizeOfMatrix, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!\n" << cudaGetErrorString(cudaStatus) << std::endl;
		goto Error;
	}

	//cudaStatus = cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess)
	//{
	//	std::cerr << "cudaMemcpy failed!\n";
	//	goto Error;
	//}

	// *******************
	// KERNEL CALLS GOES HERE!
	// *******************

	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		cudaStatus = cudaMemcpyToSymbol(k, &i, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaMemcpyToSymbol failed at iteration "<< k <<"!\n";
			goto Error;
		}

		gpuSolveBottom<<<1, MATRIX_SIZE+1-i>>>(*d_A);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "gpuSortEven kernel call failed at iteration " << k << "!\n"
				<< cudaGetErrorString(cudaStatus) << std::endl;
			goto Error;
		}
		
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching gpuSolveBottom!\n";
			goto Error;
		}
	}

	gpuSolveTop<<<1, MATRIX_SIZE>> > (*d_A);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "gpuSortEven kernel call failed at iteration " << k << "!\n"
			<< cudaGetErrorString(cudaStatus) << std::endl;
		goto Error;
	}


	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!\n";
		goto Error;
	}

	//cudaStatus = cudaMemcpy(b, *d_A + MATRIX_SIZE, (MATRIX_SIZE) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(A_GPUresult, d_A, sizeOfMatrix, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy (cudaMemcpyDeviceToHost) failed!\n";
		goto Error;
	}

	Error:
	cudaFree(d_A);
	return false;
}