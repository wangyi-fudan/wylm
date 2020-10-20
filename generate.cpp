#include	"wyhash.h"
#include	"wylm.hpp"
#include	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<sstream>
#include	<vector>
using	namespace	std;
const	unsigned	context=128;
wylm<context,768,6,256,1>	model;

int	main(int	ac,	char	**av){
	if(!model.load(av[1]))	return	0;
	float	alpha=atof(av[2]);
	vector<uint8_t>	s;	for(size_t	i=0;	i<context;	i++)	s.push_back(wyrand(&model.seed));
	for(int	i=3;	i<ac;	i++){
		for(char	*p=av[i];	*p;	p++)	s.push_back((uint8_t)*p);
		if(i+1<ac)	s.push_back(' ');
	}
	for(;;){
		float	p[256];
		uint8_t	c=model.sample(s.data()+s.size()-context,p,alpha);
		fputc(c,stderr);	s.push_back(c);
	}
	return	0;
}
