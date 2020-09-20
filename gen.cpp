#include	"wymlp.hpp"
#include	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<sstream>
#include	<vector>
using	namespace	std;
const	unsigned	threads=1;
const	unsigned	batch=1;
const	unsigned	context=31;
wymlp<context*8+1,512,6,256,batch,threads>	model;
int	main(int	ac,	char	**av){
	float	eta=atof(av[2]);
	uint64_t	seed=time(NULL);
	model.load(av[1]);
	vector<uint8_t>	s;	for(size_t	i=0;	i<context;	i++)	s.push_back(wyrand(&seed));
	for(int	i=3;	i<ac;	i++){
		for(char	*p=av[i];	*p;	p++)	s.push_back((uint8_t)*p);
		if(i+1<ac)	s.push_back(' ');
	}
	for(;;){
		uint8_t	c=model.predict(s.data()+s.size()-context,eta,&seed);
		fputc(c,stderr);	s.push_back(c);
	}
	model.free_weight();
	return	0;
}
