
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

__global__ void sum (int size, double* A, double* dst)
{
	extern __shared__ float shared_data [];
	int	i = threadIdx.x + blockIdx.x * blockDim.x,
		j = blockDim.x/2;
	if (i < size)
	{
		*(shared_data+threadIdx.x) = *(A+i);
		__syncthreads();	
		while(j != 0)
		{
			if (threadIdx.x < j)
				*(shared_data+threadIdx.x) += *(shared_data+j+threadIdx.x);
			__syncthreads();
			j/=2;
		}
		if (i == 0)
			*(dst+blockIdx.x) = *(shared_data);
	}
	return;
}

extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk)
{
	double 	*Dev_a = NULL;
	int 	size = xres*yres*sizeof(double),
		n_blks = (int) (yres/ThpBlk)+1;

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
		n_blks = (int) (yres/ThpBlk)+1;

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

	kernelMandel <<<dim_grid, dim_block>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);

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

	kernelMandel <<<dim_grid, dim_block>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, Dev_a);

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

	kernelMandel <<<dim_grid, dim_block>>> (xmin, ymin, xmax,  ymax, maxiter, xres, yres, ptr_Dev_a);

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
	double 	*avg_array,
		avg = 0.0,
		*Dev_avg = NULL,
		*Dev_a = NULL;

	int 	size = xres*yres*sizeof(double),
		n_blks = (yres+ThpBlk-1)/ThpBlk,
		size_avg = n_blks*sizeof(double),
		i;

	avg_array = (double*) malloc (size_avg);

	CUDAERR(cudaMalloc((void**) &Dev_a, size));
	CUDAERR(cudaMemcpy(Dev_a, A, size, cudaMemcpyHostToDevice));

	CUDAERR(cudaMalloc((void**) &Dev_avg, size_avg));

	sum <<< n_blks, ThpBlk, n_blks*sizeof(float) >>> (xres*yres, Dev_a, Dev_avg);

	CHECKLASTERR();
	CUDAERR(cudaMemcpy(avg_array, Dev_avg, size_avg, cudaMemcpyDeviceToHost));

	cudaFree(Dev_a);
	cudaFree(Dev_avg);

	for (i = 0 ; i < n_blks ; i++)
		avg += (double) *(avg_array+i);
		
	return avg/size*sizeof(double);
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

	kernelBinariza<<< dim_grid, dim_block>>> (xres, yres, Dev_a, med);

	cudaDeviceSynchronize();
	CHECKLASTERR();

	CUDAERR(cudaMemcpy(A, Dev_a, size, cudaMemcpyDeviceToHost));

	cudaFree(Dev_a);
}

