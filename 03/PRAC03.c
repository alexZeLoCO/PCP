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
	A = tsp (m, k, A);
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

// --- PRODUCTO MATRICIAL --- PARALLEL FOR ---
double* pmFor (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	#pragma omp parallel for private(j, tmp, p)
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
double* pmForT (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	A = tspFor (m, k, A);
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
	return C;
}

// --- PRODUCTO MATRICIAL --- PARALLEL TASK ---
double* pmTask (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	#pragma omp parallel private (i, j, tmp, p)
	#pragma omp single
	for (i = 0 ; i < m ; i++) {	// fil
		#pragma omp task
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

// --- T MATRIZ --- PARALLEL TASK ---
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
double* pmTaskT (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	A = tspFor (m, k, A);
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
	return C;
}

// --- PRODUCTO MATRICIAL --- PARALLEL BLOCK ---
// Matrices cuadradas ==> m = n = k
double* pmBlock (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, int blk)
{
	double tmp;
	int i, j, p;
	int size;
	size = n/blk;
	
	// Escalar
	for (i = 0 ; i < n*n ; i++)
		*(C+i) = beta * *(C+i);
/*
 * A B C D E
 * A B C D E
 * A B C D E
 * A B C D E
 */
	for (j = 0 ; j < size ; j++) // Col
	{
		for (i = 0 ; i < m ; i+=blk) // Block
		{
			for (p = 0 ; p < m; p+=blk)
				pm(blk, blk, blk, alpha, A+(p*lda+i), lda, B+(i*ldb+p), ldb, beta, C+(j*ldc+i), ldc);
		}
	}
	return C;
}

void blockDGEMM (int n, int m, int k, double alpha, double* A, int lda, double* B, int ldb, double* C, int ldc, int blk)
{
	int i, j, p;
	double tmp;
	for (i = 0 ; i < blk ; i+=blk)
	{
		for (j = 0 ; j < blk ; j+=blk)
		{
			tmp = 0.0;
			for (p = 0 ; p < lda ; p++)
				tmp+= *(A+i*lda+p) * *(B+p*ldb+i);
		}
	}
}

double MyDGEMMB (int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, int blk)
{
	int i, j, p, n_blocks_row = n/blk;
	
	// Escalar
	for (i = 0 ; i < m*m ; i++)
		// C[i] = beta * C[i];
		*(C+i) = beta * *(C+i);

/*
	// T
	for (i = 0 ; i < n ; i++)
	{
		for (j = i+1 ; j<n ; j++)
			// A[i*lda+j] = A[j*lda+i]
			*(A+i*lda+j) = *(A+j*lda+i);
	}
*/
	A = tsp (m, k, A);
/*
	for (i = 0 ; i < n_blocks_row ; i++)
	{
		for (j = 0 ; j < n_blocks_row ; j++)
		{
			blockDGEMM(n, m, k, alpha, A, lda, B, ldb, C+i*blk+blk*lda+j, ldc, blk);
		}
	}
*/
	for (i = 0 ; i < m ; i+=blk)
	{
		for (j = 0 ; j < n ; j+=blk)
		{
			for (d = 0 ; d < k ; d+=blk)
				blockDGEMM(n, m, k, alpha, A+p*lda+i, lda, B+j*ldb+d, ldb, C+j*ldc+i, ldc, blk);
		}
	}
	return 0;
}

double MyDGEMM(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;

  // Lo que el alumno necesite hacer 
  
  switch (tipo)
  {
    case Normal:
      timeini=Ctimer();  
      // llamada a la funcion del alumno normal. Se simula con un timer (sleep)
      pm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); // Original
      sleep(0.5);
      timefin=Ctimer()-timeini;  
      break;
    case TransA:
      timeini=Ctimer();  
      // llamada a la funcion del alumno que trabaja con la transpuesta. Se simula con un timer (sleep)
      // pmTransp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      sleep(0.5);
      timefin=Ctimer()-timeini;
      break;
    default:
      timefin=-10;
  }
  return timefin;
}
