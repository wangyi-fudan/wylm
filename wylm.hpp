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
	#define	wylm_size	(output*hidden+hidden*hidden+hidden*hidden+output*hidden)
	#define	wylm_stride	(hidden<<3)
	#define	roff(b,l)	(r+(l)*batch*hidden+(b)*hidden)
	#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
	#define	doff(b,l)	(a+(depth+(l))*batch*hidden+(b)*hidden)
	#define	ooff(b)	(a+2*depth*batch*hidden+(b)*output)
	#define woff(i,l)	(output*hidden+hidden*hidden+((l)==depth)*hidden*hidden+(i)*hidden)
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	bool	drop(uint64_t	key,	uint64_t	l,	uint64_t	b,	uint64_t	i){	return	wyhash64(key,(l<<48)|(b<<32)|i)>_dropout;	}
public:
	float	weight[wylm_size],	dropout;
	uint64_t	seed,	_dropout;
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
		fwrite(weight,sizeof(weight),1,f);
		fwrite(&dropout,4,1,f);
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
		if(fread(weight,sizeof(weight),1,f)!=1)	return	false;
		if(fread(&dropout,4,1,f)!=1)	return	false;
		fclose(f);
		_dropout=dropout*0xffffffffffffffffull;
		return	true;
	}

	float	train(uint8_t	*x[batch],	uint64_t	key,	float	eta){
		float	r[input*batch*hidden+2*batch*hidden]={},	*d0=r+input*batch*hidden,	*d1=d0+batch*hidden;
		float	a[2*depth*batch*hidden+batch*output]={},wh=1/sqrtf(hidden),wi=1/sqrtf(hidden+1),	grad[wylm_size]={};
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=roff(b,0);	uint64_t	rng=wyhash64(key,b);
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=(2*wy2u01(wyrand(&rng))-1)*drop(key,0,b,i);
			p[0]=drop(key,0,b,0);
		}		
		for(unsigned	l=1;	l<input;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+output*hidden,roff(0,l-1),d0);
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=roff(b,l),	*q=d0+b*hidden,	*e=weight+x[b][l-1]*hidden;
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(wi*(q[i]+e[i]))*drop(key,l,b,i);
				p[0]=drop(key,l,b,0);
			}
		}
		sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+output*hidden,roff(0,input-1),aoff(0,0));
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=aoff(b,0),	*e=weight+x[b][input-1]*hidden;
			#pragma GCC ivdep
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(wi*(p[i]+e[i]))*drop(key,input,b,i);
			p[0]=drop(key,input,b,0);
		}
		for(unsigned	l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),aoff(0,l-1),aoff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				float	*p=aoff(b,l);
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(p[i])*drop(key,input+l,b,i);	
				p[0]=drop(key,input+l,b,0);
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
		for(unsigned	l=depth-1;	l;	l--) {
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=aoff(b,l),	*q=doff(b,l);
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	q[i]*=gradient(p[i])*wh*drop(key,input+l,b,i);
				q[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),doff(0,l),doff(0,l-1));
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,aoff(0,l-1),doff(0,l),grad+woff(0,l));
		}
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=aoff(b,0),	*q=doff(b,0);
			#pragma GCC ivdep
			for(unsigned	i=0;	i<hidden;	i++)	q[i]*=gradient(p[i])*wi*drop(key,input,b,i);
			q[0]=0;
		}
		sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+output*hidden,doff(0,0),d0);
		sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,roff(0,input-1),doff(0,0),grad+output*hidden);
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=doff(b,0),	*e=grad+x[b][input-1]*hidden;
			#pragma GCC ivdep
			for(unsigned	i=0;	i<hidden;	i++)	e[i]+=p[i];
		}
		for(unsigned	l=input-1;	l;	l--){
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=roff(b,l),	*q=d0+b*hidden,	*o=d1+b*hidden;
				#pragma GCC ivdep
				for(unsigned	i=0;	i<hidden;	i++)	o[i]=q[i]*gradient(p[i])*wi*drop(key,l,b,i);
				o[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+output*hidden,d1,d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,roff(0,l-1),d1,grad+output*hidden);
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=d1+b*hidden,	*e=grad+x[b][l-1]*hidden;
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
	void	push_back(float	*status,	unsigned	c){
		c&=0xff;
		float	temp[hidden],	wi=1/sqrtf(hidden+1);
		memcpy(temp,status,hidden*sizeof(float));
		float	*e=weight+c*hidden;
		for(unsigned	i=0;	i<hidden;	i++){
			float	s=0,	*w=weight+output*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*temp[j];
			status[i]=activate(wi*((1-dropout)*s+e[i]));
		}
		status[0]=1;
	}
	unsigned	sample(float	*status,	float	*o,	float	alpha){
		float	a[depth*hidden]={},wh=1/sqrtf(hidden)*(1-dropout),s,*w,*p,*q;
		unsigned	i,	j,	l;
		memcpy(a,status,hidden*sizeof(float));
		for(l=1;	l<depth;	l++){
			p=a+(l-1)*hidden;	q=p+hidden;
			for(i=0;	i<hidden;	i++){
				s=0;	w=weight+woff(i,l);
				for(j=0;	j<hidden;	j++)	s+=p[j]*w[j];
				q[i]=s;
			}
			for(i=0;	i<hidden;	i++){	q[i]=activate(q[i]*wh);	}	q[0]=1;
		}
		p=a+(depth-1)*hidden;
		float	ma=-FLT_MAX,	sum=0;
		for(i=0;	i<output;	i++){
			s=0;	w=weight+woff(i,depth);
			for(j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			o[i]=s*wh*alpha;	if(o[i]>ma)	ma=o[i];
		}	
		for(i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
		for(i=0;	i<output;	i++)	o[i]/=sum;
		float	ran=wy2u01(wyrand(&seed));	sum=0;
		for(i=0;	i<output;	i++){	sum+=o[i];	if(sum>=ran)	return	i;	}
		return	output-1;
	}
};

