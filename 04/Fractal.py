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

#########################################################################
# Función en Python para resolver el cálculo del fractal		#
# Codificar el algoritmo que, para los parámetros dados, calcula	#
# el fractal siguiendo el pseudocódigo del guion. Resultado en vector A	#
#########################################################################
def mandelPy(xmin, ymin, xmax, ymax, maxiter, xres, yres, A):
    dx = (xmax - xmin)/xres
    dy = (ymax - ymin)/yres
    for i in range (0, xres, 1):
        for j in range (0, yres, 1):
            u = 0
            v = 0
            k = 1
            while (k <= maxiter and u**2+v**2<4):
                u_old = u
                u = u_old**2-v**2 + i*dx + xmin
                v = 2*u_old*v  + j*dy + ymin
                k = k + 1
            if (k >= maxiter):
                A[j*xres + i] = 0
            else:
                A[j*xres + i] = k
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
    im.save(output)
    print(f"Grabada imagen como {output}")


#########################################################################
# 			MAIN						#
#########################################################################   
if __name__ == "__main__":
    #  Procesado de los agumentos					(NO MODIFICAR)	#
    if len(sys.argv) != 7:
        print('\033[91m'+'USO: main.py <xmin> <xmax> <ymin> <yres> <maxiter> <outputfile>')
        print("Ejemplo: -0.7489 -0.74925 0.1 1024 1000 out.bmp"+'\033[0m')
        sys.exit(2)
    
    xmin=float(sys.argv[1])
    xmax=float(sys.argv[2])
    ymin=float(sys.argv[3])
    yres=int(sys.argv[4])
    maxiter=int(sys.argv[5])
    outputfile = sys.argv[6]
    
    #  Cálculo de otras variables necesarias						#
    xres = yres
    ymax = ymin+xmax-xmin

    #  Reserva de memoria de las imágenes en 1D	(AÑADIR TANTAS COMO SEAN NECESARIAS)	#
    fractalPy = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión python
    fractalProf = np.zeros(yres*xres).astype(np.double) #Esta es para el profesor
    fractalC = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión C

    fractalDiff = np.zeros(yres*xres).astype(np.double) #Esta es para el alumnado, versión Diferencias 
    
    #  Comienzan las ejecuciones							#
    print(f'\nCalculando fractal de {yres}x{xres} maxiter:{maxiter}:')
    

# --------------------------- CALCULOS -----------------------------
    print("--------------------------- CALCULOS -----------------------------")
    #  Llamada a la función de cálculo del fractal en python (versión alumnx)	(NO MODIFICAR) #
    sPy = time()
    mandelPy(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalPy)
    sPy = time()- sPy
    print(f"mandelPy (alumno)	ha tardado {sPy:1.5E} segundos")
    
    #  Llamada a la función de cálculo del fractal en C (versión profesor) (NO MODIFICAR) #
    sC = time()
    mandelProf(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalProf)
    sC = time()- sC
    print(f"mandelC (prof)		ha tardado {sC:1.5E} segundos")
    
    #  Llamada a la función de cálculo del fractal en C (versión alumnx). 		#
    sCa = time()
    mandel(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalC)
    sCa = time()- sCa
    print(f"mandelC (alumno)	ha tardado {sCa:1.5E} segundos")
 
    #  DIFFS
    diffs(xres, yres, fractalProf, fractalPy, fractalDiff)

# --------------------------- ERRORES CALCULOS -----------------------------
    print("--------------------------- ERRORES CALCULOS -----------------------------")
    #  Comprobación del error de cálculo del fractal en python (versión alumnx frente a prof) (No MODIFICAR)#
    print('(Prof v. Py)	El error es '+ str(LA.norm(fractalPy-fractalProf)), " (", errv(xres, yres, fractalPy, fractalProf), ")", sep="")
   
    #  Comprobación del error de cálculo del fractal en C (versión alumnx frente a prof)#
    print('(Prof v. C)	El error es '+ str(LA.norm(fractalC-fractalProf)), " (", errv(xres, yres, fractalC, fractalProf), ")", sep="")

    #  Comprobación del error de cálculo del fractal en C (versión alumnx frente a python)#
    print('(Py v. C)	El error es '+ str(LA.norm(fractalC-fractalPy)), " (", errv(xres, yres, fractalC, fractalPy), ")", sep="")

# --------------------------- MEDIAS -----------------------------
    print("--------------------------- MEDIAS -----------------------------")
    #  Llamada a la función de cálculo de la media (versión profesor) 	(NO MODIFICAR) 	#
    sM = time()
    promedioProf=mediaProf(xres, yres, fractalProf)
    sM = time()- sM
    print(f"Promedio (prof)={promedioProf:1.3E}	ha tardado {sM:1.5E} segundos")
    
    #  Llamada a la función de cálculo de la media en C (versión alumnx)		#
    sCM = time()
    promedioC = media(xres, yres, fractalC)
    sCM = time() - sCM
    print(f"Promedio (C)={promedioC:1.3E}	ha tardado {sCM:1.5E} segundos")
    
# --------------------------- ERRORES MEDIAS -----------------------------

    #  Comprobación del error en el promedio en C (versión alumnx frente a prof)	#
    print('El error es '+ str(LA.norm(promedioC-promedioProf)))

# --------------------------- BINARIZACION -----------------------------
    print("--------------------------- BINARIZACION -----------------------------")
    #  Llamada a la función de cálculo del binarizado (versión profesor) (NO MODIFICAR)	#
    sB = time()
    binarizaProf(xres, yres, fractalProf, promedioProf)
    sB = time()- sB
    print(f"Binariza (prof)	ha tardado {sB:1.5E} segundos")
    
    #  Llamada a la función de cálculo del binarizado en C (versión alumnx)		#
    sCB = time()
    binariza(xres, yres, fractalC, promedioC)
    sCB = time()- sCB
    print(f"Binariza (C)	ha tardado {sCB:1.5E} segundos")   
    
 # --------------------------- ERRORES BINARIZACION -----------------------------
    #  Comprobación del error en el binarizado en C (versión alumnx) 			#
    print('El error es '+ str(LA.norm(promedioC-promedioProf)))
    
 # --------------------------- DISCO -----------------------------
    print("--------------------------- DISCO -----------------------------")

    #  Grabar a archivo	la imagen que se desee (SOLO PARA DEPURAR)			#
    grabar(fractalPy,xres,yres,"out_py.bmp")
    grabar(fractalProf,xres,yres,"out_prof.bmp")
    grabar(fractalC,xres,yres,"out_c.bmp")
    grabar(fractalDiff,xres,yres,"out_diffs.bmp")
    

  
