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
	#define	ooff(b)	(a+depth*batch*hidden+(b)*output)
	#define	wylm_size	(output*hidden+hidden*hidden+depth*hidden*hidden+output*hidden)
	#define eoff(i,l)	(output*hidden+(i)*hidden)
	#define woff(i,l)	(output*hidden+hidden*hidden+(l)*hidden*hidden+(i)*hidden)
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	bool	drop(uint64_t	key,	uint64_t	l,	uint64_t	b,	uint64_t	i){	return	dropout==0||wy2u01(wyhash64(key^l,(b<<32)|i))>dropout;	}
public:
	float	weight[wylm_size],	dropout;
	uint64_t	seed;

	wylm(){
		seed=wyhash64(time(NULL),0);	const	float	norm=sqrtf(2);
		for(unsigned	i=0;	i<wylm_size;	i++)	weight[i]=wy2gau(wyrand(&seed))*norm;
		fprintf(stderr,	"model weights:\t%u\n",	wylm_size);
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
		float	a[depth*batch*hidden+batch*output]={},wh=1/sqrtf(hidden),wi=1/sqrtf(hidden+1),	grad[hidden*hidden]={};
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=init+b*hidden;
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=(wy2u01(wyrand(&seed))*2-1)*(1-dropout);
			p[0]=1-dropout;
		}
		for(unsigned	l=0;	l<input;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,1>(1,weight+eoff(0,l),l?roff(0,l-1):init,roff(0,l));
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=roff(b,l),	*e=weight+x[b][l]*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(wi*(p[i]+e[i]))*drop(key,l,b,i);
				p[0]=drop(key,l,b,0);
			}
		}
		for(unsigned	l=0;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,1>(wh,weight+woff(0,l),l?aoff(0,l-1):roff(0,input-1),aoff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				float	*p=aoff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(p[i])*drop(key,l+input,b,i);
				p[0]=drop(key,l+input,b,0);
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,1>(wh,weight+woff(0,depth),aoff(0,depth-1),ooff(0));
		float	ret=0;
		for(unsigned	b=0;	b<batch;	b++){
			float	ma=-FLT_MAX,	sum=0,	*o=ooff(b);
			for(unsigned	i=0;	i<output;	i++)	if(o[i]>ma)	ma=o[i];
			for(unsigned	i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
			for(unsigned	i=0;	i<output;	i++)	o[i]/=sum;
			ret+=-logf(fmaxf(o[x[b][input]],FLT_MIN));
			for(unsigned	i=0;	i<output;	i++)	o[i]=(o[i]-(i==x[b][input]))*wh*eta;
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),ooff(0),d0);
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,1>(-1,aoff(0,depth-1),ooff(0),weight+woff(0,depth));
		for(unsigned	l=depth-1;	l<depth;	l--) {
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=aoff(b,l),	*q=d0+b*hidden,	*o=d1+b*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	o[i]=q[i]*gradient(p[i])*wh*drop(key,l+input,b,i);
				q[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),d1,d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(-1,l?aoff(0,l-1):roff(0,input-1),d1,weight+woff(0,l));
		}

		for(unsigned	l=input-1;	l<input;	l--){
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=roff(b,l),	*q=d0+b*hidden,	*o=d1+b*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	o[i]=q[i]*gradient(p[i])*wi*drop(key,l,b,i);
				o[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+eoff(0,l),d1,d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(-1,l?roff(0,l-1):init,d1,grad);
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=d1+b*hidden,	*e=weight+x[b][l]*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	e[i]-=p[i];
			}
		}
		float	*p=weight+eoff(0,0);
		for(unsigned	i=0;	i<hidden*hidden;	i++)	p[i]+=grad[i];
		return	ret;
	}
	void	push_back(float	*status,	uint8_t	x){
		float	temp[hidden]={},	*e=weight+x*hidden,	wi=1/sqrt(hidden+1);	unsigned	i,j;
		for(i=0;	i<hidden;	i++){
			float	s=0,	*w=weight+eoff(i,0);
			for(j=0;	j<hidden;	j++)	s+=status[j]*w[j];
			temp[i]=s;
		}
		for(i=0;	i<hidden;	i++)	status[i]=activate(wi*(temp[i]+e[i]))*(1-dropout);
		status[0]=1-dropout;
	}

	unsigned	sample(float	*status,	float	*o,	float	alpha){
		float	a[2*hidden]={},*d=a+hidden,wh=1/sqrtf(hidden),s,*w;
		unsigned	i,	j,	l;
		memcpy(a,	status,	hidden*4);
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

