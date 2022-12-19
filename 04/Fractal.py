import sys
from PIL import Image
import ctypes
import numpy as np 
from numpy import linalg as LA
from numpy.ctypeslib import ndpointer
from time import time



#########################################################################
# Prepara gestión librería externa de Profesor 	(NO MODIFICAR)		#
#########################################################################
libProf = ctypes.cdll.LoadLibrary('./mandelProf.so')
# Preparando para el uso de "void mandel(double, double, double, double, int, int, int, double *)"
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None".
# .argtypes se refiere al tipo de los argumentos de la funcion.
# Hay tres funciones en la librería compartida: mandel(...), promedio(...) y binariza(...). 
# Todas ellas funcionan en paralelo si hay más de un hilo disponible.
mandelProf = libProf.mandel

mandelProf.restype  = None
mandelProf.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "double promedio(int, int, double *)"
mediaProf = libProf.promedio

mediaProf.restype  = ctypes.c_double
mediaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "void binariza(int, int, double *, double)"
binarizaProf = libProf.binariza

binarizaProf.restype  = None
binarizaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double]

#########################################################################
# Preparar gestión librería externa de Alumnx llamada mandelAlumnx.so	#
#########################################################################

libAlumnx = ctypes.cdll.LoadLibrary("./mandelAlumnx.so")

mandel = libAlumnx.mandel
mandel.restype = None
mandel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "double media(int, int, double *)"
media = libAlumnx.promedio
media.restype  = ctypes.c_double
media.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "void binariza(int, int, double *, double)"
binariza = libAlumnx.binariza
binariza.restype  = None
binariza.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double]

# Preparando para el uso de "void mandel_schedule_static"
mandel_schedule_static = libAlumnx.mandel_schedule_static
mandel_schedule_static.restype = None
mandel_schedule_static.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "void mandel_schedule_dynamic"
mandel_schedule_dynamic = libAlumnx.mandel_schedule_dynamic
mandel_schedule_dynamic.restype = None
mandel_schedule_dynamic.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "double media_atomic (int, int, double *)"
media_atomic = libAlumnx.promedio_atomic
media_atomic.restype  = ctypes.c_double
media_atomic.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Preparando para el uso de "void mandel_collapse"
mandel_collapse = libAlumnx.mandel_collapse
mandel_collapse.restype  = None 
mandel_collapse.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


#########################################################################
# Función en Python para resolver el cálculo del fractal		#
# Codificar el algoritmo que, para los parámetros dados, calcula	#
# el fractal siguiendo el pseudocódigo del guion. Resultado en vector A	#
#########################################################################
def mandelPy(xmin, ymin, xmax, ymax, maxiter, xres, yres, A):
    dx = (xmax - xmin)/xres
    dy = (ymax - ymin)/yres
    for j in range (0, xres, 1):
        for i in range (0, yres, 1):
            paso_x = j*dx+xmin
            paso_y = i*dy+ymin
            u = 0
            v = 0
            k = 1
            while (k < maxiter and (u**2+v**2)<4):
                u_old = u
                u = u_old**2-v**2 + paso_x
                v = 2*u_old*v  + paso_y
                k = k + 1
            if (k >= maxiter):
                A[i*xres + j] = 0
            else:
                A[i*xres + j] = k
    return

def diffs (xres, yres, A, B, C):
    for i in range (0, xres*yres , 1):
        if (A[i] != B[i]):
            C[i] = 255
        else:
            C[i] = 0

def errv (xres, yres, A, B):
    s = 0
    for i in range (0, xres*yres, 1):
        s+=abs(A[i]-B[i])
    return s/(xres*yres)

#########################################################################
# 	Función para guardar la imagen a archivo (NO MODIFICAR)		#
#########################################################################
def grabar(vect, xres, yres, output):
    A2D=vect.astype(np.ubyte).reshape(yres,xres) #row-major por defecto
    im=Image.fromarray(A2D)
    im.save("imgs/"+output)
    # print(f"Grabada imagen como {output}")

def run (foo, xmin, ymin, xmax, ymax, maxiter, xres, yres, nom_dst, dst, dst_b, m_foo, b_foo, f_ref, m_ref, b_ref, compare, n_threads):
    fract(foo, xmin, ymin, xmax, ymax, maxiter, xres,  yres, nom_dst, dst)
    err(dst, f_ref, compare)

    grabar(dst, xres, yres, nom_dst+".bmp")
    dst_b=dst

    m=med(m_foo, xres, yres, dst)
    err(m, m_ref, compare)

    bina(b_foo, xres, yres, dst_b, m)
    err(dst_b, b_ref, compare)

    grabar(dst_b, xres, yres, nom_dst+"_b.bmp")
    print(";", n_threads, sep="")
    return [dst, m, dst_b]

def err (src, ref, compare):
    if (compare):
    	print(';', str(LA.norm(src-ref)), sep="", end="")
    else:
    	print(';', 0.0, sep="", end="")

def fract (f_foo, xmin, ymin, xmax, ymax, maxiter, xres, yres, nom_dst, dst):
    s = time()
    dst=f_foo(xmin, ymin, xmax, ymax, maxiter, xres, yres, dst)
    s = time()- s
    # print(f"mandelPy (alumno)	ha tardado {sPy:1.5E} segundos")
    print(f"{nom_dst}.bmp;{xmin};{ymin};{xmax};{ymax};{maxiter};{xres};{yres};{s:1.5E}", end="")
    return dst

def med(m_foo, xres, yres, src):
    sM = time()
    promedio=m_foo(xres, yres, src)
    sM = time()- sM
    # print(f"Promedio (prof)={promedioProf:1.3E}	ha tardado {sM:1.5E} segundos")
    print(f";{promedio:1.3E};{sM:1.5E}", sep="", end="")
    return promedio
    
def bina (b_foo, xres, yres, src, m_src):
    sB = time()
    b_foo(xres, yres, src, m_src)
    sB = time()- sB
    # print(f"Binariza (prof)	ha tardado {sB:1.5E} segundos")
    print(f";{sB:1.5E}", end="")
        

#########################################################################
# 			MAIN						#
#########################################################################   
if __name__ == "__main__":
    #  Procesado de los agumentos					(NO MODIFICAR)	#
    if len(sys.argv) != 7:
        print('\033[91m'+'USO: main.py <xmin> <xmax> <ymin> <yres> <maxiter> <outputfile>')
        print("Ejemplo: -0.7489 -0.74925 0.1 1024 1000 out.bmp"+'\033[0m')
        sys.exit(2)

    print("out_file;xmin;ymin;xmax;ymax;maxiter;xres;yres;run_time;err;mean;mean_time;mean_err;bin_time;bin_err;n_threads")
    
    xmin=float(sys.argv[1])
    xmax=float(sys.argv[2])
    ymin=float(sys.argv[3])
    yres=int(sys.argv[4])
    maxiter=int(sys.argv[5])
    n_threads = int(sys.argv[6])
    
    #  Cálculo de otras variables necesarias						#
    xres = yres
    ymax = ymin+xmax-xmin

    #  Reserva de memoria de las imágenes en 1D	(AÑADIR TANTAS COMO SEAN NECESARIAS)	#
    prof_data = [None, None, None]
    py_data = [None, None, None]
    c_data = [None, None, None]

# --------------------------- CALCULOS -----------------------------

# plantilla: def run (foo, xmin, ymin, xmax, ymax, maxiter, xres, yres, nom_dst, dst, m_foo, b_foo, f_ref, m_ref, b_ref, compare=False):

    reses = [ 256, 512, 1024, 2048, 4096, 8192 ]
    # reses = [ 256, 512 ]

    for resolucion in reses:
        xres = resolucion
        yres = resolucion

        fractalPy = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión python
        fractalProf = np.zeros(yres*xres).astype(np.double) #Esta es para el profesor
        fractalC = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión C

        fractalPy_b = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión python
        fractalProf_b = np.zeros(yres*xres).astype(np.double) #Esta es para el profesor
        fractalC_b = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión C

        #  Llamada a la función de cálculo del fractal en C (versión profesor) (NO MODIFICAR) #
        prof_data=run(mandelProf, xmin, ymin, xmax, ymax, maxiter, xres, yres, "prof", fractalProf, fractalProf_b, mediaProf, binarizaProf, prof_data[0], prof_data[1], prof_data[2], False, n_threads)

        if (xres < 4096):
            #  Llamada a la función de cálculo del fractal en python (versión alumnx)	(NO MODIFICAR) #
            py_data=run(mandelPy, xmin, ymin, xmax, ymax, maxiter, xres, yres, "python", fractalPy, fractalPy_b, media, binariza, prof_data[0], prof_data[1], prof_data[2], True, n_threads)
    
        #  Llamada a la función de cálculo del fractal en C (versión alumnx). 		#
        c_data=run(mandel, xmin, ymin, xmax, ymax, maxiter, xres, yres, "c_tasks", fractalC, fractalC_b, media, binariza, prof_data[0], prof_data[1], prof_data[2], True, n_threads)

        c_data=run(mandel_schedule_static, xmin, ymin, xmax, ymax, maxiter, xres, yres, "c_schedule_static", fractalC, fractalC_b, media, binariza, prof_data[0], prof_data[1], prof_data[2], True, n_threads)
     
        c_data=run(mandel_schedule_dynamic, xmin, ymin, xmax, ymax, maxiter, xres, yres, "c_schedule_dynamic", fractalC, fractalC_b, media, binariza, prof_data[0], prof_data[1], prof_data[2], True, n_threads)

        c_data=run(mandel_schedule_dynamic, xmin, ymin, xmax, ymax, maxiter, xres, yres, "c_media_atomic", fractalC, fractalC_b, media_atomic, binariza, prof_data[0], prof_data[1], prof_data[2], True, n_threads)

        c_data=run(mandel_collapse, xmin, ymin, xmax, ymax, maxiter, xres, yres, "c_collapse", fractalC, fractalC_b, media_atomic, binariza, prof_data[0], prof_data[1], prof_data[2], True, n_threads)
     
     
     
