all:	training generate
training:	training.cpp wylm.hpp
	g++ training.cpp -o training -Ofast -Wall -march=native -fopenmp
generate:	generate.cpp wylm.hpp
	g++ generate.cpp -o generate -Ofast -Wall -march=native -fopenmp
