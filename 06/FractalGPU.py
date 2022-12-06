import sys

from PIL import Image
import ctypes
import numpy as np 
from numpy import linalg as LA
from numpy.ctypeslib import ndpointer
from time import time



#########################################################################
# Preparar gestión librería externa de Profesor	 			#
#########################################################################
libProf = ctypes.cdll.LoadLibrary('./mandelProfGPU.so')
# Preparando para el uso de "void mandel(double, double, double, double, int, int, int, double *)
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None"
# .argtypes se refiere al tipo de los argumentos de la funcion
# Hay dos funciones en la librería compartida: mandel(...) y mandelPar(...)
mandelProf = libProf.mandelGPU

mandelProf.restype  = None
mandelProf.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "double promedio(int, int, double *)
mediaProf = libProf.promedioGPU

mediaProf.restype  = ctypes.c_double
mediaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "void binariza(int, int, double *)
binarizaProf = libProf.binarizaGPU

binarizaProf.restype  = None
binarizaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double,ctypes.c_int]

#########################################################################
# Preparar gestión librería externa de Alumnx mandelGPU.so		#
#########################################################################
libAlumn = ctypes.cdll.LoadLibrary('./mandelGPU.so')
# Preparando para el uso de "void mandel(double, double, double, double, int, int, int, double *)
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None"
# .argtypes se refiere al tipo de los argumentos de la funcion
# Hay dos funciones en la librería compartida: mandel(...) y mandelPar(...)
mandelAlumn = libAlumn.mandelGPU

mandelAlumn.restype  = None
mandelAlumn.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "double promedio(int, int, double *)
mediaAlumn= libAlumn.promedioGPU

mediaAlumn.restype  = ctypes.c_double
mediaAlumn.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "void binariza(int, int, double *)
binarizaAlumn= libAlumn.binarizaGPU

binarizaAlumn.restype  = None
binarizaAlumn.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double,ctypes.c_int]

######################################################################### # 	Función para guardar la imagen a archivo			#
#########################################################################
def grabar(vect, xres, yres, nom):
    A2D=vect.astype(np.ubyte).reshape(yres,xres) #(filas,columnas)
    im=Image.fromarray(A2D)
    im.save(nom)

def fractal(foo, xmin, ymin, xmax, ymax, maxiter, xres, yres, fract, th):
    sC = time()
    foo(xmin, ymin, xmax, ymax, maxiter, xres, yres, fract, th)
    sC = time()- sC
    print(f";{sC:1.5E}")
    return fract

def err(my, ref, is_comparable):
    if (is_comparable):
        print(';',str(LA.norm(my-ref)), sep="", end="")
    else:
        print(';0.0')

def avg(foo, xres, yres, frac, th):
    sP = time()
    media=foo(xres, yres, frac, th)
    sP = time()- sP
    print(f";{media};{sP:1.5E}")
    return media

def bina(foo, xres, yres, fract, avg, th):
    sB = time()
    foo(xres, yres, fract, avg, th)
    sB = time()- sB
    print(f";{sB:1.5E}")
    return fract

def run (foo_mandel, xmin, xmax, ymin, ymax, xres, yres, maxiter, fract, foo_avg, foo_bin, fract_bin, ref_mandel, ref_bin, ref_med, is_comprable, th):
    fract = fractal(foo_mandel, xmin, ymin, xmax, ymax, maxiter, xres, yres, fract, th)
    err(fract, ref_mandel, is_comparable)
    media = avg(foo_avg, xres,  yres, fract, th)
    err(media, ref_med, is_comparable)
    fract_bin = fract
    my_bin = bina(foo_bin, xres, yres, fract_bin, media, th)
    err(my_bin, ref_bin, is_comparable)
    return [fract, media, fract_bin]

#########################################################################
# 			MAIN						#
#########################################################################   
if __name__ == "__main__":
    #  Proceado de los agumentos			#
    if len(sys.argv) != 8:
        print('FractalGPU.py <xmin> <xmax> <ymin> <yres> <maxiter> <ThpBlk> <outputfile>')
        print("Ejemplo: -0.7489 -0.74925 0.1007 1024 1000 32 out.bmp")
        sys.exit(2)
    outputfile = sys.argv[7]
    outPy="py"+outputfile
    xmin=float(sys.argv[1])
    xmax=float(sys.argv[2])
    ymin=float(sys.argv[3])
    yres=int(sys.argv[4])
    maxiter=int(sys.argv[5])
    ThpBlk=int(sys.argv[6])	# ThpBlk es nº de hilos en cada dimensión.
    
    #  Cálculo de otras variables necesarias					#
    xres = yres
    ymax = ymin+(xmax-xmin)
    
    #  Reserva de memoria de las imágenes en 1D					#
    fractalAlumn = np.zeros(yres*xres).astype(np.double)
    fractalC = np.zeros(yres*xres).astype(np.double)

    fractalAlumn_bin = np.zeros(yres*xres).astype(np.double)
    fractalC_bin = np.zeros(yres*xres).astype(np.double)

    print(f'xmin;xmax;ymin;ymax;xres;yres;maxiter;ThpBlk;outfile;t_mandel;e_mandel;media;t_media;e_media;t_binarizado;e_binarizado\n')
    print(f'{xmin};{xmax};{ymin};{ymax};{xres};{yres};{maxiter};{ThpBlk};{outputfile}')
# def run (foo_mandel, xmin, xmax, ymin, ymax, xres, yres, maxiter, fract, foo_avg, foo_bin, fract_bin, ref_mandel, ref_bin, ref_med, is_comprable, th):
    prof_data = run(mandelProf, xmin, xmax, ymin, ymax, xres, yres, maxiter, fractalC, mediaProf, binarizaProf, fractalC_bin, None, None, None, False, ThpBlk)

    printf()

    alumn_data = run(mandelAlumn, xmin, xmax, ymin, ymax, xres, yres, maxiter, fractalAlumn, mediaAlumn, binarizaAlumn, fractalAlumn_bin, prof_data[0], prof_data[1], prof_data[2], True, ThpBlk)

    """
    # print(f'\nEjecutando {yres}x{xres}')
    
    #  Llamadas a las funciones de cálculo del fractal Prof (NO MODIFICAR)	#
    sC = time()
    mandelProf(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalC, ThpBlk)
    sC = time()- sC
    print(f"mandelProfGPU		ha tardado {sC:1.5E} segundos")
    
    #  Llamadas a las funciones de cálculo del fractal	Alum, guardar en fractalAl#
    sC = time()
    mandelAlumn(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalAlumn, ThpBlk)
    sC = time()- sC
    print(f"mandelAlumn ha tardado {sC:1.5E} segundos")

    
    #  Comprobación de los errores			(NO MODIFICAR)		#
    print('El error es '+ str(LA.norm(fractalAlumn-fractalC)))


    #  Llamadas a las funciones de cálculo de la promedio Prof (NO MODIFICAR)	#
    sP = time()
    media=mediaProf(xres, yres, fractalC, ThpBlk)
    sP = time()- sP
    print(f"mediaProfGPU={media} y	ha tardado {sP:1.5E} segundos")

    
    #  Llamadas a las funciones de cálculo del promedio Alum			#
    sP = time()
    mediaAlumn=mediaAlumn(xres, yres, fractalAlumn, ThpBlk)
    sP = time()- sP
    print(f"mediaAlumn={mediaAlumn} y	ha tardado {sP:1.5E} segundos")
    
    
    #  Comprobación de los errores						#
    print('El error es '+ str(LA.norm(mediaAlumn-media)))
    
    
   #  Llamadas a las funciones de cálculo de la binarizado Prof (NO MODIFICAR)	#
    sB = time()
    binarizaProf(xres, yres, fractalC, media, ThpBlk)
    sB = time()- sB
    print(f"binarizaProfGPU	ha tardado {sB:1.5E} segundos")
    


    #  Llamadas a las funciones de cálculo del binarizado Alum			#
    sB = time()
    binarizaAlumn(xres, yres, fractalAlumn, mediaAlumn, ThpBlk)
    sB = time()- sB
    print(f"binarizaAlumn	ha tardado {sB:1.5E} segundos")

    #  Comprobación de los errores						#
    print('El error es '+ str(LA.norm(fractalAlumn-fractalC)))
    
    #  Grabar a archivos (nunca usar si yres>2048)				#
    grabar(fractalC,xres,yres,"Prof"+outputfile)
    grabar(fractalAlumn,xres,yres,"Alumn"+outputfile)
    """
