#include "Prototipos.h"

double Ctimer(void)
{
  struct timeval tm;

  gettimeofday(&tm, NULL);

  return tm.tv_sec + tm.tv_usec/1.0E6;
}

// a --> m*k
// b --> k*n
// c --> m*n
// c[i][j] = j*ldc*i (n_col*tot_fil+n_fil)

// --- PRODUCTO MATRICIAL --- ORIGINAL ---
double* pm (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	for (i = 0 ; i < m ; i++) {	// fil
		for (j = 0 ; j < n ; j++) { // col
			// C[i][j] = beta * C[i][j];
			*(C+j*ldc+i) = beta * *(C+j*ldc+i);
			tmp = 0.0;
			for (p = 0 ; p < k ; p++) { // col A y fil B
				// tmp += A[i][p] * B[p][i]
				tmp+=*(A+p*lda+i) * *(B+j*ldb+p);
			}
			// C[i][j] += alpha * tmp;
			*(C+j*ldc+i) += alpha * tmp;
		}	
	}
	return C;
}

// --- T MATRIZ --- ORIGINAL ---
double* tsp (int m, int k, double* A)
{
	int i, j;
	double* at = (double*) malloc (m*k*sizeof(double));
	for (i = 0 ; i < m ; i++) { // filas
		for (j = 0 ; j < k ; j++) { // cols
			*(at+j+i*k) = *(A+i+j*m);
		}
	}
	return at;
}

// --- PRODUCTO MATRICIAL T --- ORIGINAL ---
double* pmT (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	// A = tsp (m, k, A); // Eliminado en PRAC03.c para reutilizar la funcion en MyDGEMMB
	for (i = 0 ; i < m ; i++) {	// fil
		for (j = 0 ; j < n ; j++) { // col
			// C[i][j] = beta * C[i][j];
			*(C+j*ldc+i) = beta * *(C+j*ldc+i);
			tmp = 0.0;
			for (p = 0 ; p < k ; p++) { // col A y fil B
				// tmp += A[i][p] * B[p][i]
				tmp+=*(A+i*ldb+p) * *(B+j*ldb+p);
			}
			// C[i][j] += alpha * tmp;
			*(C+j*ldc+i) += alpha * tmp;
		}	
	}
	return C;
}

// --- MATRIZ T --- PARALLEL FOR ---
double* tspFor (int m, int k, double* A)
{
	int i, j;
	double* at = (double*) malloc (m*k*sizeof(double));
	#pragma omp parallel for private(j)
	for (i = 0 ; i < m ; i++) { // filas
		for (j = 0 ; j < k ; j++) { // cols
			*(at+j+i*k) = *(A+i+j*m);
		}
	}
	return at;
}

// --- PRODUCTO MATRICIAL T --- PARALLEL FOR ---
double MyDGEMM (int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	if (tipo == 2) A = tspFor (m, k, A);
	#pragma omp parallel for private(j, tmp, p)
	for (i = 0 ; i < m ; i++) {	// fil
		for (j = 0 ; j < n ; j++) { // col
			// C[i][j] = beta * C[i][j];
			*(C+j*ldc+i) = beta * *(C+j*ldc+i);
			tmp = 0.0;
			for (p = 0 ; p < k ; p++) { // col A y fil B
				// tmp += A[i][p] * B[p][i]
				tmp+=*(A+i*ldb+p) * *(B+j*ldb+p);
			}
			// C[i][j] += alpha * tmp;
			*(C+j*ldc+i) += alpha * tmp;
		}	
	}
	return 0;
}

double* tspTask (int m, int k, double* A)
{
	int i, j;
	double* at = (double*) malloc (m*k*sizeof(double));
	#pragma omp parallel private (i, j)
	#pragma omp single
	for (i = 0 ; i < m ; i++) { // filas
		#pragma omp task
		for (j = 0 ; j < k ; j++) { // cols
			*(at+j+i*k) = *(A+i+j*m);
		}
	}
	return at;
}

// --- PRODUCTO MATRICIAL T --- PARALLEL TASK ---
double MyDGEMMT (int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	if (tipo == 2) A = tspTask (m, k, A);
	#pragma omp parallel private (i, j, tmp, p) // ==> El siguiente bloque (single) se ejectua con varios hilos
	#pragma omp single // ==> Solo un hilo ejecuta el siguiente bloque (for)
	for (i = 0 ; i < m ; i++) {	// fil
		#pragma omp task  // ==> El siguiente bloque (for) es una tarea, un hilo la hara
		for (j = 0 ; j < n ; j++) { // col
			// C[i][j] = beta * C[i][j];
			*(C+j*ldc+i) = beta * *(C+j*ldc+i);
			tmp = 0.0;
			for (p = 0 ; p < k ; p++) { // col A y fil B
				// tmp += A[i][p] * B[p][i]
				tmp+=*(A+i*ldb+p) * *(B+j*ldb+p);
			}
			// C[i][j] += alpha * tmp;
			*(C+j*ldc+i) += alpha * tmp;
		}	
	}
	return 0;
}

double MyDGEMMB (int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, int blk)
{
	int i, j, p;
	
	// Escalar
	for (i = 0 ; i < m*m ; i++)
		*(C+i) = beta * *(C+i);

	// Transponer
	A = tsp (m, k, A);

	for (i = 0 ; i < m ; i+=blk)
	{
		for (j = 0 ; j < n ; j+=blk)
		{
			// #pragma omp parallel for
            for (p = 0 ; p < k ; p+=blk) 
            {
                MyDGEMM(3, blk, blk, blk, alpha, A+p+i*ldb, lda, B+p+j*ldb, ldb, 1, C+i+j*ldc, ldc);
            }
		}
	}
	
	return 0;
}
