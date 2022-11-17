#include "Prototipos.h"
#include <omp.h>

void mandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A){
	double dx, dy, u = 0, v = 0, u_old = 0, paso_x, paso_y;
	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;
	int i = 0, j = 0, k = 0;
	// #pragma omp parallel for private(i, j, u, v, k, u_old)
	#pragma omp parallel private(i, j, u, v, k, u_old, paso_x, paso_y)
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
					k++;
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
}

