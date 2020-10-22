#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<sys/time.h>
#include	<algorithm>
#include	<unistd.h>
#include	"wylm1.hpp"
#include	<unistd.h>
#include	<iostream>
#include	<fstream>
#include	<cstdlib>
#include	<fstream>
#include	<fcntl.h>
#include	<vector>
#include	<omp.h>
using	namespace	std;
const	uint64_t	threads=48;
const	uint64_t	batch=32;
const	uint64_t	fullbatch=((1u<<24)/threads/batch)*(threads*batch);
const	uint64_t	context=64;
wylm<context,128,8,256,batch>	model;

int	fd;
struct	stat	sb;
uint8_t	*data;
uint64_t	data_size,	seed[threads];
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
	#pragma omp parallel for
	for(size_t	it=0;	it<fullbatch;	it+=batch){
		uint8_t	*x[batch];	uint64_t	&s=seed[omp_get_thread_num()];
		for(size_t	b=0;	b<batch;	b++)	x[b]=data+wyrand(&s)%(data_size-context-1);
		float	l=model.train(x,wyrand(&s),eta);
		#pragma omp atomic
		score+=l;
	}
	return	score/fullbatch;
}

void	document(void){
	cerr<<"usage:	training [options] text\n";
	cerr<<"\t-o:	output model=model\n";
	cerr<<"\t-d:	hdropout=1/16\n";
	cerr<<"\t-D:	idropout=1/16\n";
	cerr<<"\t-e:	learning rate=16\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	eta=16;	string	out="model";	model.idrop=1.0/16;	model.hdrop=1.0/16;
	int	opt;
	while((opt=getopt(ac,	av,	"o:d:D:e:"))>=0){
		switch(opt){
		case	'o':	out=optarg;	break;
		case	'd':	model.hdrop=atof(optarg);	break;
		case	'D':	model.idrop=atof(optarg);	break;
		case	'e':	eta=atof(optarg);	break;
		default:	document();
		}
	}
	if(ac<optind+1)	return	0;
	omp_set_num_threads(threads);	eta=eta/(eta+threads*batch);
	for(size_t	i=0;	i<threads;	i++)	seed[i]=wyhash64(time(NULL),i);
	
	if(!open_mmap(av[optind]))	return	0;
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

