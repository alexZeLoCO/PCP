
#include "PrototiposGPU.h"

__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x, j;
	if (i < xres)
	{
		for (j = 0 ; j < yres ; j++)
		{
			double 	dx = (xmax-xmin)/xres,
				dy = (ymax-ymin)/yres,
				u = 0, v = 0, u_old = 0,
				paso_x = i*dx+xmin,
				paso_y = j*dy+ymin;
			int 	k = 1;

			while (k < maxiter && (u*u+v*v) < 4)
			{
				u_old = u;
				u = u_old*u_old - v*v + paso_x;
				v = 2*u_old*v + paso_y;
				k = k + 1;
			}
			if (k >= maxiter)	*(A+i+j*xres) = 0;
			else			*(A+i+j*xres) = k;
		}
	}
	return;
}



__global__ void kernelBinariza(int xres, int yres, double* A, double med)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;	
	if (i < xres*yres)
	{
		if (*(A+i) > med) 	*(A+i) = 255;	
		else			*(A+i) = 0;	
	}
}

extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double *Dev_a = NULL;
  	CUDAERR(cudaMalloc((void **)&Dev_a, xres*yres*sizeof(double)));
  	// CUDAERR(cudaMallocManaged((void**)&Dev_a, xres*yres*sizeof(double), cudaMemAttachGlobal));
	int n_blks = (int) (yres/ThpBlk)+1;
	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);
	cudaDeviceSynchronize();
	CHECKLASTERR();
	CUDAERR(cudaMemcpy(A, Dev_a, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void managed_mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double *Dev_a = NULL;
  	CUDAERR(cudaMallocManaged((void**)&Dev_a, xres*yres*sizeof(double), cudaMemAttachGlobal));
	int n_blks = (int) (yres/ThpBlk)+1;
	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);
	cudaDeviceSynchronize();
	CHECKLASTERR();
	CUDAERR(cudaMemcpy(A, Dev_a, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void pinned_mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double *Dev_a = NULL, *ptr_Dev_a = NULL;
	CUDAERR(cudaHostAlloc((void**)&Dev_a, xres*yres*sizeof(double), cudaHostAllocMapped));
	CUDAERR(cudaHostGetDevicePointer((void**) &ptr_Dev_a, (void*)Dev_a, 0));
	int n_blks = (int) (yres/ThpBlk)+1;
	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, ptr_Dev_a);
	cudaDeviceSynchronize();
	CHECKLASTERR();
	CUDAERR(cudaMemcpy(A, ptr_Dev_a, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void better_pinned_mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double *Dev_a = NULL;
	CUDAERR(cudaHostGetDevicePointer((void**) &Dev_a, (void*)A, 0)); // Invalid arg
	int n_blks = (int) (yres/ThpBlk)+1;
	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);
	cudaDeviceSynchronize();
	CHECKLASTERR();
}

extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk)
{
	double avg = 0;
	cublasHandle_t handle;
	int size = xres*yres;
	double *Dev_a;
	CUDAERR(cudaMalloc((void**) &Dev_a, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Dev_a, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));
	cublasCreate(&handle);
	cublasDasum(handle, size, Dev_a, 1, &avg);
	cublasDestroy(handle);
	CUDAERR(cudaMemcpy(A, Dev_a, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
	return avg/size;
}

extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk)
{
	double *Dev_a = NULL;
  	CUDAERR(cudaMalloc((void **)&Dev_a, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Dev_a, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));
	int n_blks = (int) (yres/ThpBlk)+1;
	kernelBinariza<<< n_blks, ThpBlk >>> (xres, yres, Dev_a, med);
	cudaDeviceSynchronize();
	CHECKLASTERR(); // Invalid arg on pinned. better_pinned is yet untested
	CUDAERR(cudaMemcpy(A, Dev_a, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}
