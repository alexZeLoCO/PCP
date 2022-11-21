#include "Prototipos.h"
#include <omp.h>

__global__ void pixel (double dx, double dy, double xmin, double ymin, int maxiter, double* A)
{
	double u = 0, v = 0, u_old, paso_x = xmin+dx*blockIdx.x, paso_y = ymin+dy*threadIdx.x;
	int k = 1;
	while (k < maxiter && (u*u+v*v) < 4)
	{
		u_old = u;
		u = u*u - v*v + paso_x;
		v = 2*u_old*v + paso_y;
		k = k + 1;
	}
	if (k >= maxiter) 	*(A+ (blockIdx.x * blockDim.x + threadIdx.x)) = 0;
	else 			*(A+ (blockIdx.x * blockDim.x + threadIdx.x)) = k;
	return;
}

void mandel_gpu(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int threads_per_blk, int n_blks){
	double dx, dy;
	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;
	pixel <<<n_blks, threads_per_blk>>> (dx, dy, xmin, ymin, maxiter, A);
}

void mandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A){
	double dx, dy, u = 0, v = 0, u_old = 0, paso_x, paso_y;
	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;
	int i = 0, j = 0, k = 0;
	#pragma omp parallel private (i, j, k, u, v, paso_x, paso_y)
	#pragma omp single
	for (i = 0 ; i < xres ; i++)
	{
		for (j = 0 ; j < yres ; j++)
		{
			#pragma omp task
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
					k = k + 1;
				}
				if (k >= maxiter)
				{
					*(A+j*xres+i) = 0;
				}
				else
				{
					*(A+j*xres+i) = k;
				}
			}
		}
	}
}

double promedio(int xres, int yres, double* A){
	int i;
	double s;
	s = 0;
	#pragma omp parallel for reduction(+:s)
	for (i = 0 ; i < xres*yres ; i++)
		s+=*(A+i);
   return s/(xres*yres);
}

void binariza(int xres, int yres, double* A, double med){
	int i;
	#pragma omp parallel for
	for (i = 0 ; i < xres*yres ; i++)
	{
		if (*(A+i) >= med)	*(A+i) = 255;
		else	*(A+i) = 0;
	}
	return;
}

void Genera (double* A, int n, int seed)
{
	int i = 0;
	srand(seed);
	for (i = 0 ; i < n ; i++)
		*(A+i) = ((double)(rand()%1000+1))/1.0E3;
	return;
}

double Ctimer (void)
{
	struct timeval tm;
	gettimeofday(&tm, NULL);
	return tm.tv_sec + tm.tv_usec/1.0E6;
}
