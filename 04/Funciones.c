#include "Prototipos.h"

void mandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A){
	double dx, dy, u, v, u_old;
	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;
	int i, j, k;
	for (i = 0 ; i < xres ; i++)
	{
		for (j = 0 ; j < yres ; j++)
		{
			u = 0;
			v = 0;
			k = 1;
			while (k < maxiter && (u*u+v*v) < 4)
			{
				u_old = u;
				u = u_old*u_old - v*v + i*dx+xmin;
				v = 2*u_old*v + j*dy+ymin;
				k++;
			}
			if (k >= maxiter)	*(A+j*xres+i) = 0;
			else	*(A+j*xres+i) = k;
		}
	}
}

double promedio(int xres, int yres, double* A){
	int i;
	double s;
	s = 0;
	for (i = 0 ; i < xres*yres ; i++)
		s+=*(A+i);
   return s/(xres*yres);
}

void binariza(int xres, int yres, double* A, double med){
	int i;
	for (i = 0 ; i < xres*yres ; i++)
	{
		if (*(A+i) >= med)	*(A+i) = 255;
		else	*(A+i) = 0;
	}
}

