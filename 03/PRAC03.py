import ctypes
import numpy  as np 

from numpy.ctypeslib import ndpointer as ND
from numpy           import linalg    as LA
from random          import random     
from time            import time

# LibGccO0 = ctypes.cdll.LoadLibrary('LIBS/PRACGccO0.so')
# LibGccO3 = ctypes.cdll.LoadLibrary('LIBS/PRACGccO3.so')
LibIccO0 = ctypes.cdll.LoadLibrary('LIBS/PRACIccO0.so')
LibIccO3 = ctypes.cdll.LoadLibrary('LIBS/PRACIccO3.so')

# GccO0 = LibGccO0.MyDGEMM
# GccO3 = LibGccO3.MyDGEMM
IccO0 = LibIccO0.MyDGEMMB
IccO3 = LibIccO3.MyDGEMMB

# GccO0.restype = ctypes.c_double
# GccO3.restype = ctypes.c_double
IccO0.restype = ctypes.c_double
IccO3.restype = ctypes.c_double

# GccO0.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
# GccO3.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
IccO0.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
IccO3.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]

foo = ["MyDGEMM", "MyDGEMMT", "MyDGEMMB"]
lib = [[LibIccO0.MyDGEMM, LibIccO3.MyDGEMM], [LibIccO0.MyDGEMMT, LibIccO3.MyDGEMMT], [LibIccO0.MyDGEMMB, LibIccO3.MyDGEMMB]]

talla = [ 1000, 2000, 3000]
rept  = [10, 8, 6, 4, 2]

alpha = 1.3
beta  = 1.7
tipo  = 2  # 1 normal, 2 transpuesta de A
blk   = 10

def run (lib_idx, rept, foo, tipo, m, n, k, alpha, A, lda, B, ldb, beta, F, ldc, blk):
    secs = time()
    for j in range(rept[i]):
      if (lib_idx == 2):
        TiempC=foo(tipo, m, n, k, alpha, A, lda, B, ldb, beta, F, ldc, blk) # Llamada a foo
      else:
        TiempC=foo(tipo, m, n, k, alpha, A, lda, B, ldb, beta, F, ldc) # Llamada a foo

    TIEMPO = (time()- secs)/rept[i]
    # print(f"IccO0  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
    # print(f"Error entre Python y IccO0 {LA.norm(D-F, 'fro'):1.5E}")
    print(f"{current_foo};IccO0;{m}x{n}x{k};{TIEMPO:1.5E};{LA.norm(D-F, 'fro'):1.5E}")


    
print("foo;Lang/Compiler;Size;Time;Err")
for lib_idx in range (1, 2, 1):

  current_foo = foo[lib_idx]

  IccO0 = lib[lib_idx][0]
  IccO3 = lib[lib_idx][1]

  for i in range(0,len(talla)):
    # m      = talla[i]
    # n      = m + int(m/2)
    # k      = m - int(m/2)

    m = talla[i]
    n = m + 1
    k = m - 1

    if (lib_idx == 2):
      n = k = m
      IccO0.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_int]
      IccO3.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_int]
    else:
      IccO0.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]
      IccO3.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ND(ctypes.c_double, flags="F"), ctypes.c_int, ND(ctypes.c_double, flags="F"), ctypes.c_int, ctypes.c_double,  ND(ctypes.c_double, flags="F"), ctypes.c_int]



    A = np.random.rand(m, k).astype(np.float64)
    B = np.random.rand(k, n).astype(np.float64)
    C = np.random.rand(m, n).astype(np.float64)

    D = np.copy(C)
    secs = time()
    for j in range(rept[i]):
      D = beta*D + alpha*(A @ B)

    TIEMPO = (time()- secs)/rept[i]
    # print(f"Python {m}x{n}x{k} Segundos={TIEMPO:1.5E}")
    print(f"{current_foo};Python;{m}x{n}x{k};{TIEMPO:1.5E};0")

    A = np.asarray(A, order='F')
    B = np.asarray(B, order='F')

    F = np.asarray(C, order='F')
    """
    secs = time()
    for j in range(rept[i]):
      if (lib_idx == 2):
        TiempC=IccO0(tipo, m, n, k, alpha, A, m, B, k, beta, F, m, blk) # Llamada a foo
      else:
        TiempC=IccO0(tipo, m, n, k, alpha, A, m, B, k, beta, F, m) # Llamada a foo

    TIEMPO = (time()- secs)/rept[i]
    # print(f"IccO0  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
    # print(f"Error entre Python y IccO0 {LA.norm(D-F, 'fro'):1.5E}")
    print(f"{current_foo};IccO0;{m}x{n}x{k};{TIEMPO:1.5E};{LA.norm(D-F, 'fro'):1.5E}")
    """

    run (lib_idx, rept, IccO0, tipo, m, n, k, alpha, A, m, B, k, beta, F, m, blk)

    F = np.asarray(C, order='F')
    """
    secs = time()
    for j in range(rept[i]):
      if (lib_idx == 2):
        TiempC=IccO3(tipo, m, n, k, alpha, A, m, B, k, beta, F, m, blk) # Llamada a foo
      else:
        TiempC=IccO3(tipo, m, n, k, alpha, A, m, B, k, beta, F, m) # Llamada a foo

    TIEMPO = (time()- secs)/rept[i]
    # print(f"IccO3  {m}x{n}x{k} Segundos={TIEMPO:1.5E} (Segundos medidos en C={TiempC:1.5E})")
    # print(f"Error entre Python y IccO3 {LA.norm(D-F, 'fro'):1.5E}\n")
    print(f"{current_foo};IccO3;{m}x{n}x{k};{TIEMPO:1.5E};{LA.norm(D-F, 'fro'):1.5E}")
    """

    run (lib_idx, rept, IccO3, tipo, m, n, k, alpha, A, m, B, k, beta, F, m, blk)
