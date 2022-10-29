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

double* transp (int m, int k, double* A)
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

double* pmTransp (int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
	double tmp;
	int i, j, p;
	A = transp(m, k, A);
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

double MyDGEMM(int tipo, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
  double timeini, timefin;

  // Lo que el alumno necesite hacer 
  
  switch (tipo)
  {
    case Normal:
      timeini=Ctimer();  
      // llamada a la funcion del alumno normal. Se simula con un timer (sleep)
      // pm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      pm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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
