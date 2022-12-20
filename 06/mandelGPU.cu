
#include "PrototiposGPU.h"

__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A)
{
	int 	i = threadIdx.x+blockIdx.x*blockDim.x,
	    	j;

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

__global__ void kernelMandel2D(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A)
{
	int 	i = threadIdx.x+blockIdx.x*blockDim.x,
		j = threadIdx.y+blockIdx.y*blockDim.y;

	if (i < xres && j < yres)
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
	return;
}


__global__ void kernelBinariza(int xres, int yres, double* A, double med)
{
	int 	i = threadIdx.x + blockIdx.x * blockDim.x;	

	if (i < xres*yres)
	{
		if (*(A+i) > med) 	*(A+i) = 255;	
		else			*(A+i) = 0;	
	}
	return;
}

__global__ void kernelBinariza2D(int xres, int yres, double* A, double med)
{
	int 	i = threadIdx.x + blockIdx.x * blockDim.x,
	    	j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < xres && j < yres)
	{
		if (*(A+i+j*xres) > med) 	*(A+i+j*xres) = 255;	
		else				*(A+i+j*xres) = 0;	
	}
	return;
}

__global__ void sum_blks (int size, double* A, double* dst)
{
	extern __shared__ double shared_data [];
	int	i = threadIdx.x + blockIdx.x * blockDim.x,
		j = blockDim.x/2;

	double tmp = 0.0;

	// if (i < size)
	// {
		while (i < size)
		{
			tmp += *(A+i);
			i += blockDim.x * gridDim.x;
		}

		*(shared_data+threadIdx.x) = tmp;
		__syncthreads();	

		while(j != 0)
		{
			if (threadIdx.x < j)
				*(shared_data+threadIdx.x) += *(shared_data+j+threadIdx.x);
			__syncthreads();
			j/=2;
		}

		if (threadIdx.x == 0)
			*(dst+blockIdx.x) = *(shared_data);
	// }
	return;
}

__global__ void sum (double* data, double* dst, int n_blks)
{
	extern __shared__ double shared_data [];
	int	i = threadIdx.x + blockIdx.x * blockDim.x,
		j = blockDim.x/2,
		bpt = n_blks / blockDim.x,
		k;

	double tmp = 0.0;

	// if (i < size)
	// {
		for (k = 0 ; k < bpt ; k++)
			tmp += *(data+bpt*i+k);

		*(shared_data+threadIdx.x) = tmp;
		__syncthreads();	

		while(j != 0)
		{
			if (threadIdx.x < j)
				*(shared_data+threadIdx.x) += *(shared_data+j+threadIdx.x);
			__syncthreads();
			j/=2;
		}

		if (threadIdx.x == 0)
			*(dst) = *(shared_data);
	// }
	return;
}

__global__ void sum_atomic (int size, double* data, double* dst)
{
	int	i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < size)
		// atomicAdd(dst, *(data+i));
	
	return;
}

extern "C" void mandel_omp (double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, double perc)
{
	double dx, dy, u = 0, v = 0, u_old = 0, paso_x, paso_y;
	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;
	int i = 0, j = 0, k = 0;
	#pragma omp parallel for private (i, j, u, v, k, u_old, paso_x, paso_y) schedule(dynamic)
	for (i = xres*2*(1-perc) ; i < xres ; i++)
		for (j = 0 ; j < yres ; j++)
		{
			paso_x = i*dx+xmin;
			paso_y = j*dy+ymin;
			u = 0;
			v = 0;
			k = 1;
			while (k < maxiter && (u*u+v*v) < 4)
			{
				u_old = u;
				u = u_old*u_old - v*v + paso_x;
				v = 2*u_old*v + paso_y;
				k = k+1;
			}
			if (k >= maxiter)	*(A+j*xres+i) = 0;
			else			*(A+j*xres+i) = k;
		}
	return;	
}

extern "C" void mandelHetero(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double),
		n_blks = (int) (yres*0.9+ThpBlk-1)/ThpBlk;

  	CUDAERR(cudaMallocManaged((void **)&Dev_a, size, cudaMemAttachGlobal));
	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax, ymax, maxiter, xres, yres, Dev_a);
	mandel_omp(xmin, ymin, xmax, ymax, maxiter, xres, yres, Dev_a, 0.9);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double),
		n_blks = (int) (yres+ThpBlk-1)/ThpBlk;

  	CUDAERR(cudaMalloc((void **)&Dev_a, size));

	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void managed_mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double),
		n_blks = (int) (yres+ThpBlk-1)/ThpBlk;

  	CUDAERR(cudaMallocManaged((void**)&Dev_a, size, cudaMemAttachGlobal));

	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void pinned_mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL, *ptr_Dev_a = NULL;
	int 	size = xres*yres*sizeof(double),
	    	n_blks = (int) (yres/ThpBlk)+1;

	CUDAERR(cudaHostAlloc((void**)&Dev_a, size, cudaHostAllocMapped));
	CUDAERR(cudaHostGetDevicePointer((void**) &ptr_Dev_a, (void*)Dev_a, 0));

	kernelMandel <<<n_blks, ThpBlk>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, ptr_Dev_a);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFreeHost(Dev_a);
}

extern "C" void mandelGPU2D(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double);
	dim3	dim_block (ThpBlk, ThpBlk),
		dim_grid  ((xres+dim_block.x-1)/dim_block.x, (yres+dim_block.y-1)/dim_block.y);
	
  	CUDAERR(cudaMalloc((void **)&Dev_a, size));

	kernelMandel2D <<<dim_grid, dim_block>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void managed_mandelGPU2D(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double);
	dim3	dim_block (ThpBlk, ThpBlk),
		dim_grid  ((xres+dim_block.x-1)/dim_block.x, (yres+dim_block.y-1)/dim_block.y);

  	CUDAERR(cudaMallocManaged((void**)&Dev_a, size, cudaMemAttachGlobal));

	kernelMandel2D <<<dim_grid, dim_block>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFree(Dev_a);
}

extern "C" void pinned_mandelGPU2D(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL, *ptr_Dev_a = NULL;
	int 	size = xres*yres*sizeof(double);
	dim3	dim_block (ThpBlk, ThpBlk),
		dim_grid  ((xres+dim_block.x-1)/dim_block.x, (yres+dim_block.y-1)/dim_block.y);

	CUDAERR(cudaHostAlloc((void**)&Dev_a, size, cudaHostAllocMapped));
	CUDAERR(cudaHostGetDevicePointer((void**) &ptr_Dev_a, (void*)Dev_a, 0));

	kernelMandel2D <<<dim_grid, dim_block>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, ptr_Dev_a);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));
	cudaFreeHost(Dev_a);
}

// BetterPinned summary: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk)
{
	double 	avg = 0, *Dev_a = NULL;
	int 	size = xres*yres*sizeof(double);
	cublasHandle_t handle;

	CUDAERR(cudaMalloc((void**) &Dev_a, size));
	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	cublasCreate(&handle);
	cublasDasum(handle, size/sizeof(double), Dev_a, 1, &avg);
	cublasDestroy(handle);

	cudaFree(Dev_a);
	return avg/size*sizeof(double);
}

extern "C" double promedioGPUSum(int xres, int yres, double* A, int ThpBlk)
{
	double 	*avg = NULL,
		*Dev_blks = NULL,
		*Dev_avg = NULL,
		*Dev_a = NULL;

	int 	size = xres*yres*sizeof(double),
		n_blks = (xres*yres+ThpBlk-1)/ThpBlk;

	avg = (double*) malloc (sizeof(double));

	CUDAERR(cudaMalloc((void**) &Dev_blks, n_blks*sizeof(double)));	// midway
	CUDAERR(cudaMalloc((void**) &Dev_avg, sizeof(double)));	// dst
	CUDAERR(cudaMalloc((void**) &Dev_a, size));	// src data

	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	sum_blks <<< n_blks, ThpBlk, ThpBlk*sizeof(double) >>> (xres*yres, Dev_a, Dev_blks);
	sum <<< 1, 1024, 1024*sizeof(double) >>> (Dev_blks, Dev_avg, n_blks);

	CHECKLASTERR();
	CUDAERR(cudaMemcpy(avg, Dev_avg, sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(Dev_blks);
	cudaFree(Dev_a);
	cudaFree(Dev_avg);
		
	return *avg/size*sizeof(double);
}

extern "C" double promedioGPUAtomic(int xres, int yres, double* A, int ThpBlk)
{
	double 	*avg = NULL,
		*Dev_a = NULL,
		*Dev_avg = NULL;

	int 	size = xres*yres*sizeof(double),
		n_blks = (xres*yres+ThpBlk-1)/ThpBlk;

	avg = (double*) malloc (sizeof(double));

	CUDAERR(cudaMalloc((void**) &Dev_avg, sizeof(double)));	// dst
	CUDAERR(cudaMalloc((void**) &Dev_a, size));	// src data

	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	sum_atomic <<< n_blks, ThpBlk >>> (xres*yres, Dev_a, Dev_avg);

	CHECKLASTERR();
	CUDAERR(cudaMemcpy(avg, Dev_avg, sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(Dev_a);
	cudaFree(Dev_avg);
		
	return *avg/size*sizeof(double);
}

extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk)
{
	double *Dev_a = NULL;
	int 	n_blks = (int) (xres*yres+ThpBlk-1)/ThpBlk,
		size = xres*yres*sizeof(double);

  	CUDAERR(cudaMalloc((void **)&Dev_a, size));
	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	kernelBinariza<<< n_blks, ThpBlk >>> (xres, yres, Dev_a, med);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));

	cudaFree(Dev_a);
}

extern "C" void binarizaGPU2D(int xres, int yres, double* A, double med, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double);
	dim3	dim_block (ThpBlk, ThpBlk),
		dim_grid  ((xres+dim_block.x-1)/dim_block.x, (yres+dim_block.y-1)/dim_block.y);

  	CUDAERR(cudaMalloc((void **)&Dev_a, size));
	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	kernelBinariza2D<<< dim_grid, dim_block>>> (xres, yres, Dev_a, med);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));

	cudaFree(Dev_a);
}

