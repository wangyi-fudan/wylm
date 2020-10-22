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
	#define	wylm_size	(2*65536*hidden+input*hidden+(depth-1)*hidden*hidden+output*hidden)
	#define	wylm_embed	(2*65536*hidden)
	#define	wylm_stride	(hidden<<2)
	#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
	#define	doff(b,l)	(a+(depth+(l))*batch*hidden+(b)*hidden)
	#define	ooff(b)	(a+2*depth*batch*hidden+(b)*output)
	#define woff(i,l)	(2*65536*hidden+input*hidden+((l)-1)*hidden*hidden+(i)*hidden)
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	bool	drop(uint64_t	key,	unsigned	l,	unsigned	b,	unsigned	i,	float	p){	return	wy2u01(wyhash64(key^l,(b<<16)|i))>p;	}
	unsigned	readp(const	uint8_t	*p,	uint64_t	k){	
		uint64_t	h=wyhash(&p,sizeof(const uint8_t*),k,_wyp);
		return	(h&255)?*(uint16_t*)p:(h>>48);
	}
	unsigned	read(const	uint8_t	*p){	return	*(uint16_t*)p;	}
public:
	float	idrop, hdrop, weight[wylm_size];
	uint64_t	seed;
	omp_lock_t	lock[(wylm_size-wylm_embed)/wylm_stride];

	wylm(){
		seed=wyhash64(time(NULL),0);	
		float	v=sqrtf(0.5);
		for(unsigned	i=0;	i<wylm_size;	i++)	weight[i]=(i<2*65536*hidden+input*hidden+hidden?v:1)*wy2gau(wyrand(&seed));
		for(unsigned	i=0;	i<sizeof(lock)/sizeof(omp_lock_t);	i++)	omp_init_lock(lock+i);
		fprintf(stderr,	"model weights:\t%u\n",	wylm_size);
	}

	~wylm(){	
		for(unsigned	i=0;	i<sizeof(lock)/sizeof(omp_lock_t);	i++)	omp_destroy_lock(lock+i);
	}

	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		unsigned	n;
		n=input;	fwrite(&n,4,1,f);
		n=hidden;	fwrite(&n,4,1,f);
		n=depth;	fwrite(&n,4,1,f);
		n=output;	fwrite(&n,4,1,f);
		fwrite(&idrop,4,1,f);
		fwrite(&hdrop,4,1,f);
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
		if(fread(&idrop,4,1,f)!=1)	return	false;
		if(fread(&hdrop,4,1,f)!=1)	return	false;
		if(fread(weight,sizeof(weight),1,f)!=1)	return	false;
		fclose(f);
		return	true;
	}

	float	train(uint8_t	*x[batch],	uint64_t	key,	float	eta){
		float	a[2*depth*batch*hidden+batch*output]={},wh=1/sqrtf(hidden),wi=1/sqrtf(input-1),*w,*w0,*w1,*p,*q,*g,*o;
		float	grad[wylm_size-2*65536*hidden]={};
		unsigned	b,	i,	j,	l;
		for(b=0;	b<batch;	b++){
			p=aoff(b,0);	
			for(i=0;	i<input-1;	i++)	if(drop(key,255,b,i,idrop)){
				w=weight+wylm_embed+i*hidden;	w0=weight+readp(x[b]+i,key^b)*hidden;	w1=weight+65536*hidden+readp(x[b]+input-2,key^b)*hidden;
				for(j=0;	j<hidden;	j++)	p[j]+=w[j]*w0[j]*w1[j];
			}
			for(i=0;	i<hidden;	i++)	p[i]=activate(wi*p[i])*drop(key,0,b,i,hdrop);	
			p[0]=drop(key,0,b,0,hdrop);
		}
		for(l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),aoff(0,l-1),aoff(0,l));
			for(b=0;    b<batch;    b++){
				p=aoff(b,l);
				for(i=0;	i<hidden;	i++)	p[i]=activate(p[i])*drop(key,l,b,i,hdrop);
				p[0]=drop(key,l,b,0,hdrop);
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,0>(wh,weight+woff(0,depth),aoff(0,depth-1),ooff(0));
		float	ret=0;
		for(b=0;	b<batch;	b++){
			float	ma=-FLT_MAX,	sum=0;	o=ooff(b);
			for(i=0;	i<output;	i++)	if(o[i]>ma)	ma=o[i];
			for(i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
			for(i=0;	i<output;	i++)	o[i]/=sum;
			ret+=-logf(fmaxf(o[x[b][input]],FLT_MIN));
			for(i=0;	i<output;	i++)	o[i]=(o[i]-(i==x[b][input]))*wh*eta;
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),ooff(0),doff(0,depth-1));
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,1>(1,aoff(0,depth-1),ooff(0),grad+woff(0,depth)-wylm_embed);
		for(l=depth-1;	l;	l--) {
			for(b=0;	b<batch;	b++){
				p=aoff(b,l);	q=doff(b,l);
				for(i=0;	i<hidden;	i++)	q[i]*=gradient(p[i])*wh*drop(key,l,b,i,hdrop);
				q[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),doff(0,l),doff(0,l-1));
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(1,aoff(0,l-1),doff(0,l),grad+woff(0,l)-wylm_embed);
		}
		for(b=0;	b<batch;	b++){
			p=aoff(b,0);	g=doff(b,0);
			for(i=0;	i<hidden;	i++)	g[i]*=gradient(p[i])*wi*drop(key,0,b,i,hdrop);
			g[0]=0;
			for(i=0;	i<input-1;	i++)	if(drop(key,255,b,i,idrop)){
				w=weight+wylm_embed+i*hidden;	w0=weight+readp(x[b]+i,key^b)*hidden;	w1=weight+65536*hidden+readp(x[b]+input-2,key^b)*hidden;
				p=grad+i*hidden;
				for(j=0;	j<hidden;	j++){	
					p[j]+=g[j]*w0[j]*w1[j];
					float	s=w0[j];
					w0[j]-=g[j]*w[j]*w1[j];	
					w1[j]-=g[j]*w[j]*s;	
				}
			}
		}
		const	unsigned	n=wylm_size-wylm_embed;
		for(i=0;	i<n;	i+=wylm_stride){
			p=weight+i+wylm_embed;	q=grad+i;
			omp_set_lock(lock+i/wylm_stride);
			for(j=0;	j<wylm_stride;	j++)	p[j]-=q[j];
			omp_unset_lock(lock+i/wylm_stride);
		}
		return	ret;
	}
	unsigned	sample(uint8_t	*x,	float	*o,	float	alpha){
		float	a[depth*hidden]={},wh=1/sqrtf(hidden),wi=1/sqrtf(input-1),s,*w,*w0,*w1,*p,*q;
		unsigned	i,	j,	l;
		for(i=0;	i<input-1;	i++){
			w=weight+wylm_embed+i*hidden;	w0=weight+read(x+i)*hidden;	w1=weight+65536*hidden+read(x+input-2)*hidden;
			for(j=0;	j<hidden;	j++)	a[j]+=w[j]*w0[j]*w1[j];
		}
		for(i=0;	i<hidden;	i++){	a[i]=activate(wi*(1-idrop)*a[i])*(1-hdrop);	}	a[0]=1-hdrop;
		for(l=1;	l<depth;	l++){
			p=a+(l-1)*hidden;	q=p+hidden;
			for(i=0;	i<hidden;	i++){
				s=0;	w=weight+woff(i,l);
				for(j=0;	j<hidden;	j++)	s+=p[j]*w[j];
				q[i]=s;
			}
			for(i=0;	i<hidden;	i++){	q[i]=activate(q[i]*wh)*(1-hdrop);	}	q[0]=1-hdrop;
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

