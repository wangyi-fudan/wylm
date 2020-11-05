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
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	output*hidden+input*hidden+(l-1)*hidden*hidden+i*hidden;	}
	unsigned	eoff(unsigned	i){	return	output*hidden+i*hidden;	}
	unsigned	ooff(unsigned	b){	return	depth*batch*hidden+b*output;	}
	unsigned	ioff(unsigned	b,	unsigned	l){	return	l*batch*hidden+b*hidden;	}
	bool	drop(uint64_t	key,	uint64_t	l,	uint64_t	b,	uint64_t	i){	return	dropout==0||wy2u01(wyhash64(key^l,(b<<32)|i))>dropout;	}
public:
	float	weight[output*hidden+input*hidden+(depth-1)*hidden*hidden+output*hidden],	dropout;
	uint64_t	seed;

	wylm(){
		seed=wyhash64(time(NULL),0);	dropout=0;	const	float	norm0=sqrtf(0.5f),	norm1=sqrtf(2);
		for(unsigned	i=0;	i<sizeof(weight)/sizeof(float);	i++)	weight[i]=(i<output*hidden+input*hidden?norm0:norm1)*wy2gau(wyrand(&seed));
		fprintf(stderr,	"model weights:\t%u\n",	(unsigned)(sizeof(weight)/sizeof(float)));
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
		float	a[depth*batch*hidden+batch*output]={},	d0[2*batch*hidden]={},	*d1=d0+batch*hidden;
		float	wh=1/sqrtf(hidden),wi=1/sqrtf(input),	grad[output*hidden+input*hidden]={};
		for(unsigned	l=0;	l<input;	l++){
			float	*q=weight+eoff(l);
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=a+ioff(b,0),	*e=weight+x[b][l]*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	p[i]+=q[i]*e[i];
			}
		}
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=a+ioff(b,0);
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(wi*p[i])*drop(key,0,b,i);
			p[0]=drop(key,0,b,0);
		}
		for(unsigned	l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,1>(wh,weight+woff(0,l),a+ioff(0,l-1),a+ioff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				float	*p=a+ioff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(p[i])*drop(key,l,b,i);
				p[0]=drop(key,l,b,0);
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,1>(wh,weight+woff(0,depth),a+ioff(0,depth-1),a+ooff(0));
		float	ret=0;
		for(unsigned	b=0;	b<batch;	b++){
			float	ma=-FLT_MAX,	sum=0,	*o=a+ooff(b);
			for(unsigned	i=0;	i<output;	i++)	if(o[i]>ma)	ma=o[i];
			for(unsigned	i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
			for(unsigned	i=0;	i<output;	i++)	o[i]/=sum;
			ret+=-logf(fmaxf(o[x[b][input]],FLT_MIN));
			for(unsigned	i=0;	i<output;	i++)	o[i]=(o[i]-(i==x[b][input]))*wh*eta;
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),a+ooff(0),d0);
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,1>(-1,a+ioff(0,depth-1),a+ooff(0),weight+woff(0,depth));
		for(unsigned	l=depth-1;	l;	l--) {
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=a+ioff(b,l),	*q=d0+b*hidden,	*o=d1+b*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	o[i]=q[i]*gradient(p[i])*wh*drop(key,l,b,i);
				o[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),d1,d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(-1,a+ioff(0,l-1),d1,weight+woff(0,l));
		}
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=a+ioff(b,0),	*q=d0+b*hidden;
			for(unsigned	i=0;	i<hidden;	i++)	q[i]*=gradient(p[i])*wi*drop(key,0,b,i);
			q[0]=0;
		}
		for(unsigned	l=0;	l<input;	l++){
			float	*q=weight+eoff(l),	*gq=grad+eoff(l);
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=d0+b*hidden,	*e=weight+x[b][l]*hidden,	*ge=grad+x[b][l]*hidden;
				for(unsigned	i=0;	i<hidden;	i++){	gq[i]-=p[i]*e[i];	ge[i]-=p[i]*q[i];	}
			}
		}
		const	unsigned	n=output*hidden+input*hidden;
		for(unsigned	i=0;	i<n;	i++)	weight[i]+=grad[i];
		return	ret;
	}
	unsigned	sample(uint8_t	*x,	float	*o,	float	alpha){
		float	a[2*hidden]={},*d=a+hidden,wh=1/sqrtf(hidden),wi=1/sqrt(input);
		for(unsigned	l=0;	l<input;	l++){
			float	*q=weight+eoff(l),	*e=weight+x[l]*hidden;
			for(unsigned	i=0;	i<hidden;	i++)	a[i]+=q[i]*e[i];
		}
		for(unsigned	i=0;	i<hidden;	i++)	a[i]=activate(wi*a[i])*(1-dropout);
		a[0]=1-dropout;
		for(unsigned	l=1;	l<depth;	l++){
			for(unsigned	i=0;	i<hidden;	i++){
				float	s=0,	*w=weight+woff(i,l);
				for(unsigned	j=0;	j<hidden;	j++)	s+=a[j]*w[j];
				d[i]=s;
			}
			for(unsigned	i=0;	i<hidden;	i++)	a[i]=activate(wh*d[i])*(1-dropout);
			a[0]=1-dropout;
		}
		float	ma=-FLT_MAX,	sum=0;
		for(unsigned	i=0;	i<output;	i++){
			float	s=0,	*w=weight+woff(i,depth);
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*a[j];
			o[i]=s*wh*alpha;	if(o[i]>ma)	ma=o[i];
		}	
		for(unsigned	i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
		for(unsigned	i=0;	i<output;	i++)	o[i]/=sum;
		float	ran=wy2u01(wyrand(&seed));	sum=0;
		for(unsigned	i=0;	i<output;	i++){	sum+=o[i];	if(sum>=ran)	return	i;	}
		return	output-1;
	}
};

