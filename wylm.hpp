#include	"sgemm512.hpp"
#include	<string.h>
#include	"wyhash.h"
#include	<float.h>
#include	<stdio.h>
#include	<math.h>
#include	<time.h>
#include	<omp.h>
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	batch>
class	wylm{
private:
	#define	wylm_stride	(hidden<<3)
	#define	roff(b,l)	(r+(l)*batch*hidden+(b)*hidden)
	#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
	#define	doff(b,l)	(a+(depth+(l))*batch*hidden+(b)*hidden)
	#define	ooff(b)	(a+2*depth*batch*hidden+(b)*output)
	#define	wylm_size	(output*hidden+hidden*hidden+hidden*hidden+output*hidden)
	#define eoff(i,l)	(output*hidden+(i)*hidden)
	#define woff(i,l)	(output*hidden+hidden*hidden+(l==depth)*hidden*hidden+(i)*hidden)
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	bool	drop(uint64_t	key,	uint64_t	l,	uint64_t	b,	uint64_t	i){	return	dropout==0||wy2u01(wyhash64(key^l,(b<<32)|i))>dropout;	}
public:
	float	weight[wylm_size],	dropout;
	uint64_t	seed;
	omp_lock_t	lock[wylm_size/wylm_stride];

	wylm(){
		seed=wyhash64(time(NULL),0);
		for(unsigned	i=0;	i<wylm_size;	i++)	weight[i]=wy2gau(wyrand(&seed));
		fprintf(stderr,	"model weights:\t%u\n",	wylm_size);
		for(size_t	i=0;	i<sizeof(lock)/sizeof(omp_lock_t);	i++)	omp_init_lock(lock+i);
	}

	~wylm(){	
		for(size_t	i=0;	i<sizeof(lock)/sizeof(omp_lock_t);	i++)	omp_destroy_lock(lock+i);
	}

	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		unsigned	n;
		n=input;	fwrite(&n,4,1,f);
		n=hidden;	fwrite(&n,4,1,f);
		n=depth;	fwrite(&n,4,1,f);
		n=output;	fwrite(&n,4,1,f);
		fwrite(&dropout,4,1,f);
		fwrite(weight,sizeof(weight),1,f);
		fclose(f);
		return	true;
	}

	bool	load(const	char	*F){
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		unsigned	n;
		if(fread(&n,4,1,f)!=1||n!=input)	return	false;
		if(fread(&n,4,1,f)!=1||n!=hidden)	return	false;
		if(fread(&n,4,1,f)!=1||n!=depth)	return	false;
		if(fread(&n,4,1,f)!=1||n!=output)	return	false;
		if(fread(&dropout,4,1,f)!=1)	return	false;
		if(fread(weight,sizeof(weight),1,f)!=1)	return	false;
		fclose(f);
		return	true;
	}

	float	train(uint8_t	*x[batch],	uint64_t	key,	float	eta){
		float	r[input*batch*hidden+2*batch*hidden]={},	*d0=r+input*batch*hidden,	*d1=d0+batch*hidden,	init[batch*hidden]={};
		float	a[2*depth*batch*hidden+batch*output]={},wh=1/sqrtf(hidden),wi=1/sqrtf(hidden+1),	grad[wylm_size]={};
		for(unsigned	b=0;	b<batch;	b++)	init[b*hidden]=1;
		for(unsigned	l=0;	l<input;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+eoff(0,l),l?roff(0,l-1):init,d0);
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=roff(b,l),	*q=d0+b*hidden,	*e=weight+x[b][l]*hidden;
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(wi*(q[i]+e[i]))*drop(key,l,b,i);
				p[0]=drop(key,l,b,0);
			}
		}
		for(unsigned	l=0;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),l?aoff(0,l-1):roff(0,input-1),aoff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				float	*p=aoff(b,l);
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(p[i])*drop(key,l+input,b,i);
				p[0]=drop(key,l+input,b,0);
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,0>(wh,weight+woff(0,depth),aoff(0,depth-1),ooff(0));
		float	ret=0;
		for(unsigned	b=0;	b<batch;	b++){
			float	ma=-FLT_MAX,	sum=0,	*o=ooff(b);
			for(unsigned	i=0;	i<output;	i++)	if(o[i]>ma)	ma=o[i];
			for(unsigned	i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
			for(unsigned	i=0;	i<output;	i++)	o[i]/=sum;
			ret+=-logf(fmaxf(o[x[b][input]],FLT_MIN));
			for(unsigned	i=0;	i<output;	i++)	o[i]=(o[i]-(i==x[b][input]))*wh*eta;
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),ooff(0),doff(0,depth-1));
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,1>(1,aoff(0,depth-1),ooff(0),grad+woff(0,depth));
		for(unsigned	l=depth-1;	l<depth;	l--) {
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=aoff(b,l),	*q=doff(b,l);
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	q[i]*=gradient(p[i])*wh*drop(key,l+input,b,i);
				q[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),doff(0,l),l?doff(0,l-1):d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,l?aoff(0,l-1):roff(0,input-1),doff(0,l),grad+woff(0,l));
		}
		for(unsigned	l=input-1;	l<input;	l--){
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=roff(b,l),	*q=d0+b*hidden,	*o=d1+b*hidden;
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	o[i]=q[i]*gradient(p[i])*wi*drop(key,l,b,i);
				o[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+eoff(0,l),d1,d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,l?roff(0,l-1):init,d1,grad+eoff(0,l));
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=d1+b*hidden,	*e=grad+x[b][l]*hidden;
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	e[i]+=p[i];
			}
		}
		for(unsigned	i=0;	i<wylm_size;	i+=wylm_stride){
			float	*p=weight+i,	*q=grad+i;
			omp_set_lock(lock+i/wylm_stride);
			#pragma GCC ivdep
			for(unsigned	j=0;	j<wylm_stride;	j++)	p[j]-=q[j];
			omp_unset_lock(lock+i/wylm_stride);
		}
		return	ret;
	}
	unsigned	sample(uint8_t	*x,	float	*o,	float	alpha){
		float	a[2*hidden]={},*d=a+hidden,wh=1/sqrtf(hidden),wi=1/sqrt(hidden+1),s,*w;
		unsigned	i,	j,	l;
		a[0]=1;
		for(l=0;	l<input;	l++){
			float	*e=weight+x[l]*hidden;
			for(i=0;	i<hidden;	i++){
				s=0;	w=weight+eoff(i,l);
				for(j=0;	j<hidden;	j++)	s+=a[j]*w[j];
				d[i]=s;
			}
			for(i=0;	i<hidden;	i++)	a[i]=activate(wi*(d[i]+e[i]))*(1-dropout);
			a[0]=1-dropout;
		}
		for(l=0;	l<depth;	l++){
			for(i=0;	i<hidden;	i++){
				s=0;	w=weight+woff(i,l);
				for(j=0;	j<hidden;	j++)	s+=a[j]*w[j];
				d[i]=s;
			}
			for(i=0;	i<hidden;	i++)	a[i]=activate(wh*d[i])*(1-dropout);
			a[0]=1-dropout;
		}
		float	ma=-FLT_MAX,	sum=0;
		for(i=0;	i<output;	i++){
			s=0;	w=weight+woff(i,depth);
			for(j=0;	j<hidden;	j++)	s+=w[j]*a[j];
			o[i]=s*wh*alpha;	if(o[i]>ma)	ma=o[i];
		}	
		for(i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
		for(i=0;	i<output;	i++)	o[i]/=sum;
		float	ran=wy2u01(wyrand(&seed));	sum=0;
		for(i=0;	i<output;	i++){	sum+=o[i];	if(sum>=ran)	return	i;	}
		return	output-1;
	}
};

