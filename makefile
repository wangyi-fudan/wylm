all:	training generate
training:	training.cpp wylm1.hpp makefile
	g++-9 training.cpp -o training -Ofast -Wall -march=native -fopenmp -fopt-info-vec
generate:	generate.cpp wylm1.hpp makefile
	g++-9 generate.cpp -o generate -Ofast -Wall -march=native -fopenmp

