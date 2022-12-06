cflags=-Xcompiler -fopenmp,-O3,-Wall,-fPIC
libs=-lcublas -lcudart -lgomp

all: cleanall mandelGPU 

mandelGPU.o: mandelGPU.cu
	nvcc $(cflags) -c mandelGPU.cu -o mandelGPU.o 

mandelGPU: mandelGPU.o
	g++ -shared -Wl,-soname,mandelGPU.so -o mandelGPU.so mandelGPU.o -L$(CUDADIR)/lib64 $(libs)
	rm -f mandelGPU.o

managed_mandelGPU.o: managed_mandelGPU.cu
	nvcc $(cflags) -c managed_mandelGPU.cu -o managed_mandelGPU.o 

managedGPU: managed_mandelGPU.o
	g++ -shared -Wl,-soname,managed_mandelGPU.so -o managed_mandelGPU.so managed_mandelGPU.o -L$(CUDADIR)/lib64 $(libs)
	rm -f managed_mandelGPU.o

clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f mandelGPU.so mandelGPU.o

