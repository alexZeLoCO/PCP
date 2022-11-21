/* ********************************************************************** */
/*                     ESTE FICHERO NO DEBE SER MODIFICADO                */
/* ********************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#include <string.h>
#ifdef OMP
	#include <omp.h>
#endif

#define CHECKNULL(x) do { if ((x) == NULL) { \
	printf("Error reservando memoria en %s linea %d\n", __FILE__, __LINE__);\
	return EXIT_FAILURE;}} while(0)

#define CUDAERR(x) do { if ((x) != cudaSuccess) {\
	printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(x), __FILE__, __LINE__);\
	return EXIT_FAILURE;}} while(0)

#define CUBLASERR(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { \
	printf("CUBLAS error: %s, line %d\n", __FILE__, __LINE__);\
	return EXIT_FAILURE;}} while (0)

#define CHECKLASTERR() __getLastCudaError(__FILE__, __LINE__);
inline void __getLastCudaError (const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i): getLastCudaError() CUDA error: (%d) %s.\n",
				file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void Genera (double*, int, int);
double Ctimer (void);

void mandel_gpu(double, double, double, double, int, int, int, double *, int, int);
void mandel(double, double, double, double, int, int, int, double *);

double media(int, int, double *);

void binariza(int, int, double *, double);

__global__ void pixel (double, double, double, double, int, double*);

