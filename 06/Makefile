cflags=-Xcompiler -fopenmp,-O3,-Wall,-fPIC,-L/opt/cuda/include
# cflags=-Xcompiler -fopenmp,-O3,-Wall,-fPIC,-L/opt/cuda/include --gpu-architecture=compute_61 --gpu-code=sm_61
libs=-lcublas -lcudart -lgomp

all: cleanall mandelGPU 

mandelGPU.o: mandelGPU.cu
	nvcc $(cflags) -c mandelGPU.cu -o mandelGPU.o 

mandelGPU: mandelGPU.o
	g++ -shared -Wl,-soname,mandelGPU.so -o mandelGPU.so mandelGPU.o -L$(CUDADIR)/lib64 $(libs)
	# g++ -shared -Wl,-soname,mandelGPU.so -o mandelGPU.so mandelGPU.o -L/opt/cuda/lib64 $(libs)
	rm -f mandelGPU.o

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f mandelGPU.so mandelGPU.o

