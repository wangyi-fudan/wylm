#include	"wyhash.h"
#include	<float.h>
#include	<stdio.h>
#include	<math.h>
#include	<omp.h>
#include	"sgemm512.hpp"
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	batch,	unsigned	threads>
class	wymlp{
private:
	float   act(float x){	return	x/sqrtf(1+x*x); }
	float   gra(float x){	x=1-x*x;	return	x*sqrtf(x);	}
	uint64_t	size(void){	return	input*hidden+(depth-1)*hidden*hidden+output*hidden;	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	input*hidden+(l-1)*hidden*hidden+i*hidden;	}
public:
	float	*weight;
	wymlp(){	weight=NULL;	omp_set_num_threads(threads);	}
	void	alloc_weight(void){	free(weight);weight=(float*)aligned_alloc(64,size()*sizeof(float));	}
	void	free_weight(void){	free(weight);	weight=NULL;	}
	void	init_weight(uint64_t	seed){	for(size_t	i=0;	i<size();	i++)	weight[i]=wy2gau(wyrand(&seed));	}
	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		fwrite(weight,sizeof(float),size(),f);
		fclose(f);
		return	true;
	}
	bool	load(const	char	*F){
		if(weight==NULL)	alloc_weight();
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		fread(weight,sizeof(float),size(),f);
		fclose(f);
		return	true;
	}
	float	train(uint8_t	**x,	float	eta) {
		#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
		#define	doff(b,l)	(d+(l)*batch*hidden+(b)*hidden)
		const	float	wh=1/sqrtf(hidden), wi=1/sqrtf(input);
		float   *a=(float*)aligned_alloc(64,(2*depth*batch*hidden+batch*output)*sizeof(float));	
		float	*d=a+depth*batch*hidden,	*o=d+depth*batch*hidden,	*p,	*q,	ret=0;
		float	inp[batch*input],	gr[(hidden>input?(hidden>output?hidden:output):(input>output?input:output))*hidden];
		for(unsigned	b=0;	b<batch;	b++){
			p=inp+b*input;
			for(unsigned	i=0;	i<input-1;	i++)	p[i]=(((int)((x[b][i>>3]>>(i&7))&1)<<1)-1);
			p[input-1]=1;
		}	
		sgemm<1,0,hidden,batch,input,input,input,hidden,0>(wi,weight,inp,a);
		for(unsigned	b=0;	b<batch;	b++){
			p=aoff(b,0);
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=act(p[i]);
			p[0]=1;
		}
		for(unsigned	l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),aoff(0,l-1),aoff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				p=aoff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=act(p[i]);
				p[0]=1;
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,0>(wh,weight+woff(0,depth),aoff(0,depth-1),o);
		float	e=eta/(eta+threads*batch)*wh;
		for(unsigned    b=0;    b<batch;    b++){
			p=o+b*output;
			float	m=p[0],	s=0;
			for(unsigned	i=1;	i<output;	i++)	if(p[i]>m)	m=p[i];
			for(unsigned	i=0;	i<output;	i++)	s+=(p[i]=expf(p[i]-m));
			for(unsigned    i=0;    i<output;   i++)	p[i]/=s;
			for(unsigned    i=0;    i<output;   i++){
				if(i==x[b][input>>3]){
					ret-=log2f(p[i]>FLT_MIN?p[i]:FLT_MIN);
					p[i]=(p[i]-1)*e;
				}	
				else	p[i]=p[i]*e;
			}
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),o,doff(0,depth-1));
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,0>(1,aoff(0,depth-1),o,gr);
		p=weight+woff(0,depth);
		for(unsigned	i=0;	i<output*hidden;	i++){
			#pragma omp atomic
			p[i]-=gr[i];
		}
		for(unsigned	l=depth-1;	l;	l--) {
			for(unsigned	b=0;	b<batch;	b++){
				p=aoff(b,l);	q=doff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	q[i]*=gra(p[i])*wh;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),doff(0,l),doff(0,l-1));
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,0>(1,aoff(0,l-1),doff(0,l),gr);
			p=weight+woff(0,l);
			for(unsigned	i=0;	i<hidden*hidden;	i++){
				#pragma omp atomic
				p[i]-=gr[i];
			}
		}
		for(unsigned    b=0;    b<batch;    b++){
			p=aoff(b,0);	q=doff(b,0);
			for(unsigned	j=0;	j<hidden;	j++)	q[j]*=gra(p[j])*wi;
		}
		sgemm<0,1,input,hidden,batch,input,hidden,input,0>(1,inp,d,gr);
		for(unsigned	i=0;	i<input*hidden;	i++){
			#pragma omp atomic
			weight[i]-=gr[i];
		}
		free(a);
		return	ret;
	}
	uint8_t	predict(uint8_t	*x,	float	alpha,	uint64_t	*seed) {
		float   *a=(float*)aligned_alloc(64,(depth*hidden+output)*sizeof(float));	
		float	*o=a+depth*hidden,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input),	inp[input],	*w,s;
		for(unsigned	i=0;	i<input-1;	i++)	inp[i]=(((int)((x[i>>3]>>(i&7))&1)<<1)-1);
		inp[input-1]=1;
		for(unsigned	i=0;	i<hidden;	i++){
			w=weight+i*input;	s=0;
			for(unsigned	j=0;	j<input;	j++)	s+=inp[j]*w[j];
			a[i]=s*wi;
		}
		for(unsigned	i=0;	i<hidden;	i++)	a[i]=act(a[i]);
		a[0]=1;
		for(unsigned	l=1;	l<depth;	l++) {
			float	*p=a+(l-1)*hidden,	*q=a+l*hidden;
			for(unsigned	i=0;	i<hidden;	i++) {
				w=weight+woff(i,l);	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=s*wh;
			}
			for(unsigned	i=0;	i<hidden;	i++)	q[i]=act(q[i]);
			q[0]=1;
		}
		float	*p=a+(depth-1)*hidden,	m=-FLT_MAX;
		for(unsigned    i=0;    i<output;   i++){
			w=weight+woff(i,depth);	s=0;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			o[i]=s*wh*alpha;	if(o[i]>m)	m=o[i];
		}
		s=0;
		for(unsigned    i=0;    i<output;   i++)	s+=(o[i]=expf(o[i]-m));
		float	r=wy2u01(wyrand(seed))*s;	s=0;
		for(unsigned    i=0;    i<output;   i++){
			s+=o[i];	if(s>=r)	return	i;
		}
		free(a);
		return	0;
	}
};
