#include	"sgemm512.hpp"
#include	<string.h>
#include	"wyhash.h"
#include	<float.h>
#include	<stdio.h>
#include	<math.h>
#include	<time.h>
#include	<omp.h>
const	uint64_t	word=2;
template<uint64_t	input,	uint64_t	hidden,	uint64_t	depth,	uint64_t	output,	uint64_t	batch>
class	wylm{
private:
	const	uint64_t	embed=1u<<(word*8);
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	uint64_t	size(void){	return	2*embed*hidden+input*hidden+hidden+(depth-1)*hidden*hidden+output*hidden;	}
	bool	drop(uint64_t	key,	uint64_t	l,	uint64_t	b,	uint64_t	i,	double	p){	return	wy2u01(wyhash64(key,(l<<56)|(b<<40)|i))>p;	}
	uint64_t	readp(const	uint8_t	*p,	uint64_t	k){	
		uint64_t	h=wyhash(&p,sizeof(const uint8_t*),k,_wyp);
		return	(h&255)?*(uint16_t*)p:(h>>48);
	}
	uint64_t	read(const	uint8_t	*p){	return	*(uint16_t*)p;	}
public:
	double	idrop,	hdrop;
	float	*weight;
	uint64_t	seed,	locks;
	omp_lock_t	*lock;

	wylm(){
		seed=wyhash64(time(NULL),0);	
		uint64_t	n=size();
		weight=(float*)aligned_alloc(64,	n*sizeof(float));
		float	v=sqrtf(0.5);
		for(uint64_t	i=0;	i<n;	i++)	weight[i]=(i<2*embed*hidden+input*hidden+hidden?v:1)*wy2gau(wyrand(&seed));
		fprintf(stderr,	"model weights:\t%llu\n",	(unsigned long long)n);
		locks=size()/hidden-2*embed;	lock=new	omp_lock_t[locks];
		for(uint64_t	i=0;	i<locks;	i++)	omp_init_lock(lock+i);
	}

	~wylm(){	
		free(weight);	
		for(uint64_t	i=0;	i<locks;	i++)	omp_destroy_lock(lock+i);
		delete	[]	lock;
	}

	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		uint64_t	n;
		n=input;	fwrite(&n,8,1,f);
		n=hidden;	fwrite(&n,8,1,f);
		n=depth;	fwrite(&n,8,1,f);
		n=output;	fwrite(&n,8,1,f);
		fwrite(&idrop,8,1,f);
		fwrite(&hdrop,8,1,f);
		fwrite(weight,size()*sizeof(float),1,f);
		fclose(f);
		return	true;
	}

	bool	load(const	char	*F){
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		uint64_t	n;
		if(fread(&n,8,1,f)!=1||n!=input)	return	false;
		if(fread(&n,8,1,f)!=1||n!=hidden)	return	false;
		if(fread(&n,8,1,f)!=1||n!=depth)	return	false;
		if(fread(&n,8,1,f)!=1||n!=output)	return	false;
		if(fread(&idrop,8,1,f)!=1)	return	false;
		if(fread(&hdrop,8,1,f)!=1)	return	false;
		if(fread(weight,size()*sizeof(float),1,f)!=1)	return	false;
		fclose(f);
		return	true;
	}

	float	train(uint8_t	*x[batch],	uint64_t	key,	float	eta){
		float	a[2*depth*batch*hidden+batch*output]={},wh=1/sqrtf(hidden),wi=1/sqrtf(input-word+1),*w,*w0,*w1,*p,*q,*g,*o;
		#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
		#define	doff(b,l)	(a+(depth+(l))*batch*hidden+(b)*hidden)
		#define	ooff(b)	(a+2*depth*batch*hidden+(b)*output)
		#define woff(i,l)	(2*embed*hidden+input*hidden+hidden+((l)-1)*hidden*hidden+(i)*hidden)
		float	grad[size()-2*embed*hidden]={};
		uint64_t	b,	i,	j,	l;
		for(b=0;	b<batch;	b++){
			p=aoff(b,0);	
			for(i=0;	i<input-word;	i++)	if(drop(key,255,b,i,idrop)){
				w=weight+2*embed*hidden+i*hidden;	w0=weight+readp(x[b]+i,key^b)*hidden;	w1=weight+embed*hidden+readp(x[b]+input-word,key^b)*hidden;
				for(j=0;	j<hidden;	j++)	p[j]+=w[j]*w0[j]*w1[j];
			}
			w=weight+2*embed*hidden+input*hidden;
			for(i=0;	i<hidden;	i++){	p[i]=activate(wi*(p[i]+w[i]))*drop(key,0,b,i,hdrop);	}	p[0]=1;
		}
		for(l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),aoff(0,l-1),aoff(0,l));
			for(uint64_t    b=0;    b<batch;    b++){
				p=aoff(b,l);
				for(uint64_t	i=0;	i<hidden;	i++){	p[i]=activate(p[i])*drop(key,l,b,i,hdrop);	}	p[0]=1;
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,0>(wh,weight+woff(0,depth),aoff(0,depth-1),ooff(0));
		float	ret=0;
		for(b=0;	b<batch;	b++){
			float	ma=-FLT_MAX,	sum=0;	o=ooff(b);
			for(i=0;	i<output;	i++)	if(o[i]>ma)	ma=o[i];
			for(i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
			for(i=0;	i<output;	i++)	o[i]/=sum;
			ret+=-log2f(fmaxf(o[x[b][input]],FLT_MIN));
			for(i=0;	i<output;	i++)	o[i]=(o[i]-(i==x[b][input]))*wh*eta;
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),ooff(0),doff(0,depth-1));
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,1>(1,aoff(0,depth-1),ooff(0),grad+woff(0,depth)-2*embed*hidden);
		for(l=depth-1;	l;	l--) {
			for(uint64_t	b=0;	b<batch;	b++){
				p=aoff(b,l);	q=doff(b,l);
				for(uint64_t	i=0;	i<hidden;	i++)	q[i]*=gradient(p[i])*wh*drop(key,l,b,i,hdrop);
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),doff(0,l),doff(0,l-1));
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,aoff(0,l-1),doff(0,l),grad+woff(0,l)-2*embed*hidden);
		}
		for(b=0;	b<batch;	b++){
			p=aoff(b,0);	g=doff(b,0);	w=grad+input*hidden;
			for(i=0;	i<hidden;	i++){	g[i]*=gradient(p[i])*wi*drop(key,0,b,i,hdrop);	w[i]+=g[i];	}
			for(i=0;	i<input-word;	i++)	if(drop(key,255,b,i,idrop)){
				w=weight+2*embed*hidden+i*hidden;	w0=weight+readp(x[b]+i,key^b)*hidden;	w1=weight+embed*hidden+readp(x[b]+input-word,key^b)*hidden;
				p=grad+i*hidden;
				for(j=0;	j<hidden;	j++){	
					p[j]+=g[j]*w0[j]*w1[j];
					float	s=w0[j];
					w0[j]-=g[j]*w[j]*w1[j];	
					w1[j]-=g[j]*w[j]*s;	
				}
			}
		}
		const	uint64_t	n=size()-2*embed*hidden;
		for(uint64_t	i=0;	i<n;	i+=hidden){
			p=weight+i+2*embed*hidden;	q=grad+i;	l=i/hidden;
			omp_set_lock(lock+l);
			for(uint64_t	j=0;	j<hidden;	j++)	p[j]-=q[j];
			omp_unset_lock(lock+l);
		}
		return	ret;
	}
	uint64_t	sample(uint8_t	*x,	float	*o,	float	alpha){
		float	a[depth*hidden]={},wh=1/sqrtf(hidden),wi=1/sqrtf(input-word+1),s,*w,*w0,*w1,*p,*q;
		uint64_t	i,	j,	l;
		for(i=0;	i<input-word;	i++){
			w=weight+2*embed*hidden+i*hidden;	w0=weight+read(x+i)*hidden;	w1=weight+embed*hidden+read(x+input-word)*hidden;
			for(j=0;	j<hidden;	j++)	a[j]+=w[j]*w0[j]*w1[j];
		}
		w=weight+2*embed*hidden+input*hidden;
		for(i=0;	i<hidden;	i++){	a[i]=activate(wi*((1-idrop)*a[i]+w[i]))*(1-hdrop);	}	a[0]=1;
		for(l=1;	l<depth;	l++){
			p=a+(l-1)*hidden;	q=p+hidden;
			for(i=0;	i<hidden;	i++){
				s=0;	w=weight+2*embed*hidden+input*hidden+hidden+(l-1)*hidden*hidden+i*hidden;
				for(j=0;	j<hidden;	j++)	s+=p[j]*w[j];
				q[i]=s;
			}
			for(i=0;	i<hidden;	i++){	q[i]=activate(q[i]*wh)*(1-hdrop);	}	q[0]=1;
		}
		p=a+(depth-1)*hidden;
		float	ma=-FLT_MAX,	sum=0;
		for(i=0;	i<output;	i++){
			s=0;	w=weight+2*embed*hidden+input*hidden+hidden+(depth-1)*hidden*hidden+i*hidden;
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
