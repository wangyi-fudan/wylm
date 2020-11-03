#include	"sgemm512.hpp"
#include	<string.h>
#include	"wyhash.h"
#include	<float.h>
#include	<stdio.h>
#include	<math.h>
#include	<time.h>
#include	<omp.h>
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	batch,	unsigned	kmer=5>
class	wylm{
private:
	float	activate(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gradient(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	input*hidden+kmer*2*output*hidden+l*hidden*hidden+i*hidden;	}
	unsigned	eoff(unsigned	i,	unsigned	l){	return	input*hidden+l*output*hidden+i*hidden;	}
	unsigned	ooff(unsigned	b){	return	depth*batch*hidden+b*output;	}
	unsigned	ioff(unsigned	b,	unsigned	l){	return	l*batch*hidden+b*hidden;	}
public:
	float	weight[input*hidden+kmer*2*output*hidden+depth*hidden*hidden+output*hidden],	dropout;
	uint64_t	seed;

	wylm(){
		seed=wyhash64(time(NULL),0);	dropout=0;	const	float	norm=sqrtf(2);
		for(unsigned	i=0;	i<sizeof(weight)/sizeof(float);	i++)	weight[i]=wy2gau(wyrand(&seed))*norm;
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
		float	r[input*batch*hidden]={},	a[depth*batch*hidden+batch*output]={},	d0[2*batch*hidden]={},	*d1=d0+batch*hidden;
		float	wh=1/sqrtf(hidden),	wi=1/sqrtf(kmer*2+2),	wi1=1/(input+1.0f-kmer);
		for(unsigned	l=kmer-1;	l<input;	l++){
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=r+ioff(b,l);
				for(unsigned	k=0;	k<kmer*2+2;	k++){
					float	*q;
					if(k<kmer)	q=weight+eoff(x[b][l-k],k);	
					else	if(k<2*kmer)	q=weight+eoff(x[b][input-1-(k-kmer)],k);
					else	q=weight+(k==kmer*2?l:0)*hidden;
					for(unsigned	i=0;	i<hidden;	i++)	p[i]+=q[i];
				}
				float	*q=a+ioff(b,0);
				for(unsigned	i=0;	i<hidden;	i++)	q[i]+=(p[i]=activate(wi*p[i]));
			}
		}
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=a+ioff(b,0);
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(p[i]*wi1);
			p[0]=1;
		}
		for(unsigned	l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,1>(wh,weight+woff(0,l),a+ioff(0,l-1),a+ioff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				float	*p=a+ioff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=activate(p[i]);
				p[0]=1;
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
				for(unsigned	i=0;	i<hidden;	i++)	o[i]=q[i]*gradient(p[i])*wh;
				o[0]=0;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),d1,d0);
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(-1,a+ioff(0,l-1),d1,weight+woff(0,l));
		}
		for(unsigned	b=0;	b<batch;	b++){
			float	*p=a+ioff(b,0),	*p0=d0+b*hidden,	*p1=d1+b*hidden;
			for(unsigned	i=0;	i<hidden;	i++)	p1[i]=p0[i]*gradient(p[i])*wi1;
			p1[0]=0;
		}
		for(unsigned	l=input-1;	l>=kmer-1;	l--){
			for(unsigned	b=0;	b<batch;	b++){
				float	*p=r+ioff(b,l),	*p0=d0+b*hidden,	*p1=d1+b*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	p0[i]=p1[i]*gradient(p[i])*wi;
				p0[0]=0;
				for(unsigned	k=0;	k<kmer*2+2;	k++){
					float	*q;
					if(k<kmer)	q=weight+eoff(x[b][l-k],k);	
					else	if(k<2*kmer)	q=weight+eoff(x[b][input-1-(k-kmer)],k);
					else	q=weight+(k==kmer*2?l:0)*hidden;
					for(unsigned	i=0;	i<hidden;	i++)	q[i]-=p0[i];
				}
			}
		}
		return	ret;
	}

	unsigned	sample(uint8_t	*x,	float	*o,	float	alpha){
		float	d0[2*hidden]={},	*d1=d0+hidden;
		float	wh=1/sqrtf(hidden),	wi=1/sqrtf(kmer*2+2),	wi1=1/(input+1.0f-kmer);
		for(unsigned	l=kmer-1;	l<input;	l++){
			memset(d0,	0,	hidden*sizeof(float));
			for(unsigned	k=0;	k<kmer*2+2;	k++){
				float	*q;
				if(k<kmer)	q=weight+eoff(x[l-k],k);	
				else	if(k<2*kmer)	q=weight+eoff(x[input-1-(k-kmer)],k);
				else	q=weight+(k==kmer*2?l:0)*hidden;
				for(unsigned	i=0;	i<hidden;	i++)	d0[i]+=q[i];
			}
			for(unsigned	i=0;	i<hidden;	i++)	d1[i]+=activate(wi*d0[i]);
		}
		for(unsigned	i=0;	i<hidden;	i++)	d0[i]=activate(d1[i]*wi1);
		d0[0]=1;
		for(unsigned	l=1;	l<depth;	l++){
			for(unsigned	i=0;	i<hidden;	i++){
				float	s=0,	*w=weight+woff(i,l);
				for(unsigned	j=0;	j<hidden;	j++)	s+=d0[j]*w[j];
				d1[i]=s;
			}
			for(unsigned	i=0;	i<hidden;	i++)	d0[i]=activate(wh*d1[i]);
			d0[0]=1;
		}
		float	ma=-FLT_MAX,	sum=0;
		for(unsigned	i=0;	i<output;	i++){
			float	s=0,	*w=weight+woff(i,depth);
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*d0[j];
			o[i]=s*wh*alpha;	if(o[i]>ma)	ma=o[i];
		}	
		for(unsigned	i=0;	i<output;	i++)	sum+=(o[i]=expf(o[i]-ma));
		for(unsigned	i=0;	i<output;	i++)	o[i]/=sum;
		float	ran=wy2u01(wyrand(&seed));	sum=0;
		for(unsigned	i=0;	i<output;	i++){	sum+=o[i];	if(sum>=ran)	return	i;	}
		return	output-1;
	}
};

