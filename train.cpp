#define	SAVE_MODEL
#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<unistd.h>
#include	<fcntl.h>
#include	<sys/time.h>
#include	"wymlp.hpp"
#include	<algorithm>
#include	<unistd.h>
#include	<iostream>
#include	<fstream>
#include	<cstdlib>
#include	<fstream>
#include	<vector>
#include	<omp.h>
using	namespace	std;
const	unsigned	threads=48;
const	unsigned	batch=128;
const	unsigned	fullbatch=(1u<<24)/batch;
const	unsigned	context=31;
wymlp<context*8+1,128,6,256,batch,threads>	model;
int	fd;
struct	stat	sb;
uint8_t	*data;
uint64_t	data_size;
float	eta;
uint64_t	seed[threads];

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
	for(size_t	it=0;	it<fullbatch;	it++){
		size_t	t=omp_get_thread_num();
		uint8_t	*x[batch];
		for(size_t	b=0;	b<batch;	b++)	x[b]=data+wyrand(seed+t)%(data_size-context-1);
		double	s=model.train(x,eta);
		#pragma omp atomic
		score+=s;
	}
	return	score/batch/fullbatch;
}

int	main(int	ac,	char	**av){
	if(ac!=4){	cerr<<"train data model eta(~0.001)\n";	return	0;	}
	for(size_t	i=0;	i<threads;	i++)	seed[i]=wyhash64(time(NULL),i);
	open_mmap(av[1]);	eta=atof(av[3]);
	model.alloc_weight();	model.init_weight(seed[0]);
	#ifdef	SAVE_MODEL
	model.load(av[2]);
	string	fn=av[2];	fn+=".log";
	ofstream	fo(fn.c_str(),ios::app);
	fo.precision(7);	fo.setf(ios::fixed);
	#endif
	cerr.precision(5);	cerr.setf(ios::fixed);
	double	l0=FLT_MAX;	unsigned	gr=0;
	for(size_t	it=0;	gr<4;	it++){
		timeval	beg,	end;
		gettimeofday(&beg,NULL);
		double	l=sgd();
		gettimeofday(&end,NULL);
		double	dt=(end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec));
		#ifdef	SAVE_MODEL
		model.save(av[2]);
		fo<<it<<'\t'<<l<<'\t'<<dt<<"s\t"<<eta<<'\n';	fo.flush();
		#endif
		cerr<<it<<'\t'<<l<<'\t'<<dt<<"s\t"<<eta<<'\n';
		if(l>l0){	gr++;	eta/=10;	}
		l0=l;
	}
	#ifdef	SAVE_MODEL
	fo.close();
	#endif
	model.free_weight();
	close_mmap();
	time_t	t0=time(NULL);
	cerr<<ctime(&t0);

	return	0;
}
