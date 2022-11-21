#include "Prototipos.h"

/* Calcula v[i] = x[i] + y[i] usando CUDA
   Entrada:
   	xmin
	xmax
	ymin
	yres
	maxiter
	n_threads
	threads_per_block
*/

int main(int argc, char *argv[])
{
  int maxiter, threads_per_block, n_blocks, seed, cpu = 1, mem = 0;
  
  double *A = NULL, *Devi_A = NULL, xmin, xmax, ymin, ymax, xres, yres, time;

  /* CUDA and CUBLAS variables */
  int ndev;
    
  if (argc < 8) {
     printf("Uso: %s <xmin> <xmax> <ymin> <yres> <maxiter> <threads_per_block> <seed> [CPU]\n", argv[0]);
     return -1;
  }

  xmin			= atof(argv[1]);
  xmax			= atof(argv[2]);
  ymin			= atof(argv[3]);
  yres			= atof(argv[4]);
  maxiter 		= atoi(argv[5]);
  threads_per_block	= atoi(argv[6]);
  seed			= atoi(argv[7]);
  if (argc > 8)	cpu 	= atoi(argv[8]);
  if (argc > 9) mem	= atoi(argv[9]);
  xres = yres;
  ymax = ymin+xmax-xmin;

  if (cpu)
  {
  	CHECKNULL(A=(double*)malloc(xres*yres*sizeof(double)));
  	Genera(A, xres*yres, seed);

  	/* Resuelve el problema en la CPU */
  	time=Ctimer();
  		mandel(xmin, ymin, xmax, ymax, maxiter, xres, yres, A);
  	time=Ctimer()-time;
  	printf("El tiempo en la CPU  es %2.7E segundos.\n", time);
  }

  cudaError_t ret=cudaGetDeviceCount(&ndev);
  if (ndev == 0||ret!=0)
  {
     printf("Error 1: No hay GPU con capacidades CUDA\n");
     return -1;
  }
  else printf("INFO: Hay %d GPUs con capacidades CUDA, seguimos\n", ndev);  
  
  if (mem == 1) CUDAERR(cudaMallocManaged((void **)&Devi_A, xres*yres*sizeof(double), cudaMemAttachGlobal));
  else if (mem == 2)
  {
	CUDAERR(cudaHostAlloc((void **)&A, xres*yres*sizeof(double), cudaHostAllocMapped));
	CUDAERR(cudaHostGetDevicePointer((void **)&Devi_A, (void*)A, 0));
  }
  else CUDAERR(cudaMalloc((void **)&Devi_A, xres*yres*sizeof(double)));

  if (cpu) CUDAERR(cudaMemcpy(Devi_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));
  else Genera(Devi_A, xres*yres, seed);


  /* Resuelve el problema en la GPU */
  n_blocks = ceil((xres*yres + threads_per_block - 1) / threads_per_block);
  time=Ctimer();
	pixel <<<n_blocks, threads_per_block>>> ((xmax-xmin)/xres, (ymax-ymin)/yres, xmin, ymin, maxiter, Devi_A);
     	cudaDeviceSynchronize();
  time=Ctimer()-time;

  /* Paso 5º */
  CHECKLASTERR();
  printf("El tiempo del kernel CUDA es %2.7E segundos.\n", time);

  /* Paso 6º */

  if (cpu) free(A);
  if (mem == 2) CUDAERR(cudaFreeHost(A));
  else cudaFree(Devi_A);

  return 0;
}
