comp=nvcc
flags=-Xcompiler -O0,-Wall,-fopenmp,-g
LIBS=-lcurand -lcudart

all: cleanall mandelAlumnx

mandelAlumnx: Funciones.cu
	$(comp) -o mandelAlumnx.so Funciones.cu $(flags) $(LIBS) 

main: main.cu
	$(comp) -o main main.cu Funciones.cu $(flags) $(LIBS) 


clean:
	rm -f *~ *.o core

cleanall: clean
	rm -f mandelAlumnx.so

