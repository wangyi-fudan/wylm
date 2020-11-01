#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<sys/time.h>
#include	<algorithm>
#include	<unistd.h>
#include	"wylm.hpp"
#include	<unistd.h>
#include	<iostream>
#include	<fstream>
#include	<cstdlib>
#include	<fstream>
#include	<fcntl.h>
#include	<vector>
#include	<omp.h>
using	namespace	std;
const	uint64_t	batch=32;
const	uint64_t	fullbatch=1u<<20;
const	uint64_t	context=32;
const	uint64_t	hidden=64;
wylm<context,hidden,5,256,batch>	model;

int	fd;
struct	stat	sb;
uint8_t	*data;
uint64_t	data_size;
float	eta;

int	open_mmap(const	char	*F){
	fd=open(F,	O_RDONLY);	if(fd<0)	return	0;
	fstat(fd,	&sb);
	data=(uint8_t*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
	if(data==MAP_FAILED)	return	0;
	data_size=sb.st_size;
	return	data_size;
}

void	close_mmap(void){
	munmap(data,sb.st_size);	close(fd);
}

double	sgd(void){
	double	score=0;
	for(size_t	it=0;	it<fullbatch;	it+=batch){
		uint8_t	*x[batch];
		for(size_t	b=0;	b<batch;	b++)	x[b]=data+wyrand(&model.seed)%(data_size-context-1);
		score+=model.train(x,wyrand(&model.seed),eta/(eta+batch));
	}
	return	score/fullbatch/log(2);
}

void	document(void){
	cerr<<"usage:	training [options] text\n";
	cerr<<"\t-i:	input model=NULL\n";
	cerr<<"\t-o:	output model=model\n";
	cerr<<"\t-d:	dropout=0\n";
	cerr<<"\t-e:	learning rate=1\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	eta=1;	string	out="model",	in;	model.dropout=0;
	int	opt;
	while((opt=getopt(ac,	av,	"i:o:d:e:"))>=0){
		switch(opt){
		case	'i':	in=optarg;	break;
		case	'o':	out=optarg;	break;
		case	'd':	model.dropout=atof(optarg);	break;
		case	'e':	eta=atof(optarg);	break;
		default:	document();
		}
	}
	if(ac<optind+1)	return	0;
	if(!open_mmap(av[optind]))	return	0;
	if(in.size())	model.load(in.c_str());
	cerr.precision(4);	cerr.setf(ios::fixed);
	double	l0=FLT_MAX;	uint64_t	gr=0;
	for(size_t	it=0;	gr<4;	it++){
		timeval	beg,	end;
		gettimeofday(&beg,NULL);
		double	l=sgd();
		gettimeofday(&end,NULL);
		double	dt=(end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec));
		model.save(out.c_str());
		cerr<<it<<'\t'<<l<<'\t'<<dt<<"s\t"<<eta<<'\n';
		if(l>l0){	gr++;	eta/=10;	}	else	eta*=0.99;
		l0=l;
	}
	close_mmap();
	return	0;
}

