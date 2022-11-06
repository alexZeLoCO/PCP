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
	A = tspTask (m, k, A);
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

void blockDGEMM (int n, int blk, int ld, double alpha, double *A, double *B, double *C, int blk_column, int blk_row)
{
	int i, j, k, total_size = n*n, blk_column_start = blk_column * blk * ld, blk_row_start = blk_row * blk;
	double tmp;
	for (i = 0 ; i < blk ; i++) // i = idx col 
	{
		for (j = 0 ; j < blk ; j++) // j = idx fil 
		{
			tmp = 0.0;
			for (k = 0 ; k < total_size ; k++)
				tmp+= *(A+k+(i+blk_column_start)*ld) * *(B+j+blk_row_start+k*ld);

			*(C+j+i*ld) += alpha * tmp;
		}	
	}
}

double MyDGEMMB (int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc, int blk)
{
	int i, j, n_blocks_row = n/blk;
	
	// Escalar
	for (i = 0 ; i < m*m ; i++)
		// C[i] = beta * C[i];
		*(C+i) = beta * *(C+i);

/*
	// T
	for (i = 0 ; i < n ; i++)
	{
		for (j = i+1 ; j<n ; j++)
			// A[i*lda+j] = A[j*lda+i];
			*(A+i*lda+j) = *(A+j*lda+i);
	}
*/
	A = tsp (m, k, A);
	for (i = 0 ; i < n_blocks_row ; i++)
	{
		for (j = 0 ; j < n_blocks_row ; j++)
		{
			blockDGEMM (n, blk, ldc, alpha, A, B, C, i, j);
		}
	}
	
	return 0;
}

/*
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
      pmTransp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      sleep(0.5);
      timefin=Ctimer()-timeini;
      break;
    default:
      timefin=-10;
  }
  return timefin;
}
*/
