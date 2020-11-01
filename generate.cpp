#include	"wyhash.h"
#include	"wylm.hpp"
#include	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<sstream>
#include	<vector>
using	namespace	std;
const	uint64_t	context=32;
const	uint64_t	hidden=64;
wylm<context,hidden,6,256,1>	model;

int	main(int	ac,	char	**av){
	if(!model.load(av[1]))	return	0;
	float	alpha=atof(av[2]),	status[hidden];	
	for(size_t	i=0;	i<hidden;	i++)	status[i]=2*wy2u01(wyrand(&model.seed))-1;
	status[0]=1;	unsigned	word=0;
	for(int	i=3;	i<ac;	i++){
		for(char	*p=av[i];	*p;	p++){	word=(word<<8)|(uint8_t)*p;	model.push_back(status,word);	fputc(*p,stderr);	}
		if(i+1<ac){	word=(word<<8)|(uint8_t)' ';	model.push_back(status,word);	fputc(' ',stderr); }
	}
	for(;;){
		float	p[256];
		uint8_t	c=model.sample(status,p,alpha);
		word=(word<<8)|c;
		fputc(c,stderr);	model.push_back(status,word);
	}
	return	0;
}
