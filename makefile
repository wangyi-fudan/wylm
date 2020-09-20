all:	train gen
train:	train.cpp wymlp.hpp makefile
	g++ train.cpp -o train -Ofast -march=native -Wall -fopenmp -static -s
gen:	gen.cpp wymlp.hpp makefile
	g++ gen.cpp   -o gen -Ofast -march=native -Wall -fopenmp -static -s

