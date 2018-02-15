#!/home/mrstats/maamen/epd/bin python
# MY Functions definitions are here
# THE SIMULATION IS AT THE END !!! alb_functions.call_simulation(nn)
#contains
#GS = alb_rndn(100,0,1);
#gms = alb_rnd_gamma(3,5,10);
#invgam(x,a,b):
#fit_Gauss_InvGamma
#fit_Gauss_InvGamma2 #It allows init values for a1 and a2
#call_simulation: simulation fitting gaussian and 2 inv gammas ate each end of gauss.
import scipy.special as sp
import alb_MM_functions as alb
import math
import scipy as sc
from operator import itemgetter
from pylab import * #for the find command	
import numpy as np;


import os
from nibabel.testing import data_path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
import time
import warnings
import scipy as sc
warnings.filterwarnings("ignore")

from pylab import *
import pylab
from optparse import OptionParser
#create a folder to save results and name it ERASEME
import os	
import os.path


#RANDOM GAUSS GENERATOR: you can call alb_rndn function as > GS = alb_rndn(100,0,1);
def rndn(n,mu,sigma):
	Gauss_samples = np.random.normal(mu, sigma, n);
	GS=Gauss_samples;
	return GS;

	
#RANDOM GAMMA GENERATOR: you can call alb_rnd_gamma function as gms = alb_rnd_gamma(3,5,10);
def rnd_gamma(alpha,beta,n):
	Gamma_samples = np.random.gamma(alpha, beta,n);
	gms=Gamma_samples;
	return gms;

#iNVERSE GAMMA PDF,InvG=invgam(GS,2,3): IN MATLAB invgam = @(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
#def normpdf(x,m,v):
#	out=np.multiply(  pow( 2*math.pi*(pow(v,2)),-0.5 ) ,  exp(np.divide(pow(x-(m*np.ones(size(x))),2),(-2*v)*np.ones(size(x)))); 
#	return out;


#iNVERSE GAMMA PDF,InvG=invgam(GS,2,3): IN MATLAB invgam = @(x,a,b) b^a/gamma(a).*(1./x).^(a+1).*exp(-b./x);
def invgam(x,aa,bb):
	out=np.multiply(np.ones(x.shape[0]) *np.divide(np.power(bb,aa),math.gamma(aa)), np.multiply(np.power(np.divide(np.ones(size(x)),x),aa+1) ,np.exp(np.divide(np.ones(size(x))*-bb,x))));
	return out;

def gam_old(x,aa,bb):
	out=np.multiply(np.multiply(np.true_divide(1,np.multiply(np.power(bb,aa),math.gamma(aa))), np.power(x,aa-1)),np.exp(np.divide(np.ones(size(x))*-x,bb)));
	return out;

def gam(x,aa,bb):
	import scipy.stats
	out=scipy.stats.gamma.pdf(x,aa,0,bb)
	return out; 
 
#define functions to translate parameters of InvGamma dist from alpha,beta  <-----> mu, var
def muIG(alpha,beta):
	return np.true_divide(beta,(alpha-1)); #alpha>1
def varIG(alpha,beta):
	return np.true_divide( np.power(beta,2) , ((alpha-2) * (alpha-1)^2 ) );
def alphaIG(mu,var):
	#a=(math.pow(mu,2)/var)+2;
	aa=np.true_divide(np.power(mu, 2),var)+2;
	return aa
def betaIG(mu,var):
	#a= (mu*((math.pow(mu,2)/var)+1));
	aa=np.multiply(mu,np.true_divide(np.power(mu, 2),var)+1);
	return aa
def alphaGm(mu,var):
	aa=np.true_divide(np.power(mu, 2),var);
	return aa
def betaGm(mu,var):
	aa=np.true_divide(var,mu);
	return aa

#Fits inv gamma in positive or negative tail of Gaussian with mean m y var v; 
#The gaussian is fixed, with mean m and variance v (variance or std?????)  they must be given!!!
#side=1, fits pos inv gamma , side =-1 negative inv gamma (where pos and negative are defined wrt mean m!)
#repetitions (number of EM initializations) and iterations (per EM run) must be given, example 20, 30

	

def mmfit3(x, maxiters,tol,MM):
	import copy;import numpy as np;
	all_params=[0,1,3,1,-3,1]
	init_mu1=all_params[0]
	init_v1=all_params[1]
	init_mu2=all_params[2]
	init_v2=all_params[3]
	init_mu3=all_params[4]
	init_v3=all_params[5]

	init_PI=np.zeros(3)
	init_PI[0]=np.true_divide(1,3)
	init_PI[1]=np.true_divide(1,3)
	init_PI[2]=np.true_divide(1,3)
	#First estimation initial parameters for inv gammas: alphas y betas

	#if MM==1:
	#	1#fix for gmm
	#el
	if MM==2:	
		init_a1 =alb.alphaGm(init_mu2,init_v2)
		init_b1 =alb.betaGm(init_mu2,init_v2)
		init_a2 =alb.alphaGm(-1*init_mu3,init_v3)
		init_b2 =alb.betaGm(-1*init_mu3,init_v3)
	elif MM==3:	
		init_a1 =alb.alphaIG(init_mu2,init_v2)
		init_b1 =alb.betaIG(init_mu2,init_v2)
		init_a2 =alb.alphaIG(-1*init_mu3,init_v3)
		init_b2 =alb.betaIG(-1*init_mu3,init_v3)


	#rename parameters for iteration
	tmp_mu1=copy.deepcopy(init_mu1);
	tmp_v1=copy.deepcopy(init_v1);
	tmp_mu2=copy.deepcopy(init_mu2);
	tmp_v2=copy.deepcopy(init_v2);
	tmp_mu3=copy.deepcopy(init_mu3);
	tmp_v3=copy.deepcopy(init_v3);

	tmp_a1=copy.deepcopy(init_a1);
	tmp_b1=copy.deepcopy(init_b1);
	tmp_a2=copy.deepcopy(init_a2);
	tmp_b2=copy.deepcopy(init_b2);
	tmp_PI=copy.deepcopy(init_PI);
	#make structures to save the parameters estimates at each iteration
	mu1=np.zeros(maxiters+2)
	v1=np.zeros(maxiters+2)
	mu2=np.zeros(maxiters+2)
	v2=np.zeros(maxiters+2)
	mu3=np.zeros(maxiters+2)
	v3=np.zeros(maxiters+2)

	a1=np.zeros(maxiters+2)
	b1=np.zeros(maxiters+2)
	a2=np.zeros(maxiters+2)
	b2=np.zeros(maxiters+2)
	tmp_lik=np.zeros(maxiters+2)
	real_lik=np.zeros(maxiters+2)
	PI=np.zeros(3*(maxiters+2));#3 because we fit 3 components
	PI=np.reshape(PI,[maxiters+2,3]); 
	#save first values of this structures as the initialized values
	mu1[0]=tmp_mu1;
	v1[0]=tmp_v1;
	mu2[0]=tmp_mu2;
	v2[0]=tmp_v2;
	mu3[0]=tmp_mu2;
	v3[0]=tmp_v2;
	a1[0]=tmp_a1;
	b1[0]=tmp_b1;
	a2[0]=tmp_a2;
	b2[0]=tmp_b2;
 


	#indexes of samples to assign 0 prob wrt each inv gammas
	xneg=find(x<pow(10,-14));
	xpos=find(x>-pow(10,-14)); eps=np.finfo(float).eps
 
 
	#First Expectation step to evaluate initilization it=0
	it=0;
	Nobj=sc.stats.norm(tmp_mu1,np.power(tmp_v1,0.5));
	pGa=Nobj.pdf(x);
	pGa[pGa==0]=np.power(10,-14);
	if MM==1:
		1
	elif MM==2:
		dum2=alb.gam(x,tmp_a1, tmp_b1);
		dum3=alb.gam(-1*x,tmp_a2, tmp_b2);
	elif MM==3:
		dum2=alb.invgam(x,tmp_a1, tmp_b1);
		dum3=alb.invgam(-1*x,tmp_a2, tmp_b2);
	
	dum2[xneg]=0;
	dum3[xpos]=0; 
	D1=np.multiply(np.ones(size(x))*tmp_PI[0],pGa); D1[np.where(D1<np.power(10,-14))]=eps
	D2=np.multiply(np.ones(size(x))*tmp_PI[1],dum2);D2[np.where(D2<np.power(10,-14))]=eps
	D3=np.multiply(np.ones(size(x))*tmp_PI[2],dum3);D3[np.where(D3<np.power(10,-14))]=eps
	D=D1+D2+D3;
	R1=np.divide(D1,np.ones(size(x))*D);
	R2=np.divide(D2,np.ones(size(x))*D);
	R3=np.divide(D3,np.ones(size(x))*D);
	resp= sc.column_stack([R1,R2,R3]);	
  	#M step
	N=np.ones(3)	
	N[0]=sum(R1)
	N[1]=sum(R2)
	N[2]=sum(R3)
	tmp_PI[0]=N[0]/sum(N)
	tmp_PI[1]=N[1]/sum(N)
	tmp_PI[2]=N[2]/sum(N)		
	#tmp_lik[it]=sum( np.multiply(resp[:,0],(log(tmp_PI[0])+log(pGa))) + np.multiply(resp[:,1],(log(tmp_PI[1])+log(dum2)))  + np.multiply(resp[:,2],(log(tmp_PI[2])+log(dum3)))       );#bishop
	real_lik[it]=sum(log(np.multiply(tmp_PI[0],pGa)+  np.multiply(tmp_PI[1],dum2)+ np.multiply(tmp_PI[2],dum3) )  )
	trol=np.zeros([3,x.shape[0]])
	trol[0,:]=np.multiply(D1,tmp_PI[0]);
	trol[1,:]=np.multiply(D2,tmp_PI[1]);
	trol[2,:]=np.multiply(D3,tmp_PI[2]);
	real_lik[it]=np.sum(np.log(trol.sum(0))) 
 
 
 
	#ITERATE
	flag=0
	while flag==0:
		it=it+1

		#update gaussian mean and variance
		tmp_mu1=[]
		tmp_mu1=sum(np.multiply(resp[:,0],x))/N[0];
		tmp_v1=[];
		tmp_v1=sum(np.multiply(resp[:,0],pow(x-tmp_mu1,2)))/N[0];
		#if tmp_v <= 0.5:
		#	tmp_v=0.5
		#UPDATE EACH INVERSE GAMMA. 
		tmp_mu2=[]
		tmp_mu2=sum(np.multiply(resp[:,1],x))/N[1];
		tmp_v2=[];
		tmp_v2=sum(np.multiply(resp[:,1],pow(x-tmp_mu2,2)))/N[1];
		#if tmp_v2< 0.1:#pow(10,-1):
			#tmp_v2=0.1
		tmp_mu3=[]
		tmp_mu3=sum(np.multiply(resp[:,2],x))/N[2];
		tmp_v3=[];
		tmp_v3=sum(np.multiply(resp[:,2],pow(x-tmp_mu3,2)))/N[2];
		#if tmp_v3< 0.1:
			#tmp_v3=0.1
		if MM==2:
			tmp_a1 =alb.alphaGm(tmp_mu2,tmp_v2)
			tmp_b1 =alb.betaGm(tmp_mu2,tmp_v2)
			tmp_a2 =alb.alphaGm(-1*tmp_mu3,tmp_v3)
			tmp_b2 =alb.betaGm(-1*tmp_mu3,tmp_v3)	
		elif MM==3:	
			tmp_a1 =alb.alphaIG(tmp_mu2,tmp_v2)
			tmp_b1 =alb.betaIG(tmp_mu2,tmp_v2)
			tmp_a2 =alb.alphaIG(-1*tmp_mu3,tmp_v3)
			tmp_b2 =alb.betaIG(-1*tmp_mu3,tmp_v3)




		#print 'it_num',it
		Nobj=sc.stats.norm(tmp_mu1,np.power(tmp_v1,0.5));
		pGa=Nobj.pdf(x);
		pGa[pGa==0]=np.power(10,-14);
		if MM==1:
			1
		elif MM==2:
			dum2=alb.gam(x,tmp_a1, tmp_b1);
			dum3=alb.gam(-1*x,tmp_a2, tmp_b2);
		elif MM==3:
			dum2=alb.invgam(x,tmp_a1, tmp_b1);
			dum3=alb.invgam(-1*x,tmp_a2, tmp_b2);
			
		dum2[xneg]=0;
		dum3[xpos]=0; 
  
		dum2[np.isnan(dum2)] = 0; dum2[np.isinf(dum2)] = 0;dum2[dum2==0]=np.power(10,-14)
		dum3[np.isnan(dum3)] = 0; dum3[np.isinf(dum3)] = 0; dum3[dum3==0]=np.power(10,-14);
            
		D1=np.multiply(np.ones(size(x))*tmp_PI[0],pGa); D1[np.where(D1<np.power(10,-14))]=eps
		D2=np.multiply(np.ones(size(x))*tmp_PI[1],dum2);D2[np.where(D2<np.power(10,-14))]=eps
		D3=np.multiply(np.ones(size(x))*tmp_PI[2],dum3);D3[np.where(D3<np.power(10,-14))]=eps
		#D3[xpos]=0;
  		#D2[xneg]=0;

		
		
		D=D1+D2+D3;
		R1=np.divide(D1,np.ones(size(x))*D);
		R2=np.divide(D2,np.ones(size(x))*D);
		R3=np.divide(D3,np.ones(size(x))*D);
		resp= sc.column_stack([R1,R2,R3]);	
          #M step
		N=np.ones(3)	
		N[0]=sum(R1)
		N[1]=sum(R2)
		N[2]=sum(R3)
		tmp_PI[0]=N[0]/sum(N)
		tmp_PI[1]=N[1]/sum(N)
		tmp_PI[2]=N[2]/sum(N)	;tmp_PI[np.where(tmp_PI<np.power(10,-14))]=eps;	
		#tmp_lik[it]=sum( np.multiply(resp[:,0],(log(tmp_PI[0])+log(pGa))) + np.multiply(resp[:,1],(log(tmp_PI[1])+log(dum2)))  + np.multiply(resp[:,2],(log(tmp_PI[2])+log(dum3)))       );#bishop
		real_lik[it]=sum(log(np.multiply(tmp_PI[0],pGa)+  np.multiply(tmp_PI[1],dum2)+ np.multiply(tmp_PI[2],dum3) )  )
		trol=np.zeros([3,x.shape[0]])
		trol[0,:]=np.multiply(D1,tmp_PI[0]);
		trol[1,:]=np.multiply(D2,tmp_PI[1]);
		trol[2,:]=np.multiply(D3,tmp_PI[2]);
		real_lik[it]=np.sum(np.log(trol.sum(0)))

		

		


		if (abs((real_lik[it]-real_lik[it-1])/real_lik[it-1] )< tol) | (it > maxiters):
			flag=1
		

	stmu1=tmp_mu1;#mu1[it];
	stv1=tmp_v1#v1[it];
	stmu2=tmp_mu2#mu2[it];
	stv2=tmp_v2#v2[it];
	stmu3=tmp_mu3#mu3[it];
	stv3=tmp_v3#v3[it];
	stPI=tmp_PI#PI[it,:];
	lik=real_lik[0:it] #tmp_lik[it];

	return stmu1,stv1,stmu2,stv2, stmu3,stv3,stPI,lik,it,resp;#,tmp_lik;# a1,b1,a2,b2,mu,v,PI,tmp_lik;



def mmfit2(x, maxiters,tol,MM):
	print MM
	all_params=[0,1,2,1]
	init_mu1=all_params[0]
	init_v1=all_params[1]
	init_mu2=all_params[2]
	init_v2=all_params[3]
	init_PI=np.zeros(2)
	init_PI[0]=0.5
	init_PI[1]=0.5
	#First estimation initial parameters for inv gammas: alphas y betas

	#if MM==1:
	#	1#fix for gmm
	#el
	if MM==2:	
		init_a1 =alb.alphaGm(init_mu2,init_v2)
		init_b1 =alb.betaGm(init_mu2,init_v2)
	elif MM==3:	
		init_a1 =alb.alphaIG(init_mu2,init_v2)
		init_b1 =alb.betaIG(init_mu2,init_v2)

	#rename parameters for iteration
	tmp_mu1=init_mu1;
	tmp_v1=init_v1;
	tmp_mu2=init_mu2;
	tmp_v2=init_v2;


	tmp_a1=init_a1;
	tmp_b1=init_b1;
	tmp_PI=init_PI;
	#make structures to save the parameters estimates at each iteration
	mu1=np.zeros(maxiters+2)
	v1=np.zeros(maxiters+2)
	mu2=np.zeros(maxiters+2)
	v2=np.zeros(maxiters+2)


	a1=np.zeros(maxiters+2)
	b1=np.zeros(maxiters+2)
	tmp_lik=np.zeros(maxiters+2)
	real_lik=np.zeros(maxiters+2)
	PI=np.zeros(2*(maxiters+2));#3 because we fit 2 components
	PI=np.reshape(PI,[maxiters+2,2]); 
	#save first values of this structures as the initialized values
	mu1[0]=tmp_mu1;
	v1[0]=tmp_v1;
	mu2[0]=tmp_mu2;
	v2[0]=tmp_v2;


	a1[0]=tmp_a1;
	b1[0]=tmp_b1;
	PI[0,0]=tmp_PI[0];
	PI[0,1]=tmp_PI[1];
	flag=0
	it=-1;
	#indexes of samples to assign 0 prob wrt non-gauss components
	xneg=find(x<pow(10,-14));
	while flag==0:
		it=it+1
		#print it
		#for it in range (0,maxiters):
		#print 'it1',it

		Nobj=sc.stats.norm(tmp_mu1,pow(tmp_v1,(1/2)));
		pGa=Nobj.pdf(x);
		pGa[pGa==0]=pow(10,-14);
		if MM==1:
			1
		elif MM==2:
			dum2=alb.gam(x,tmp_a1, tmp_b1);
		elif MM==3:
			dum2=alb.invgam(x,tmp_a1, tmp_b1);
			
		
		dum2[xneg]=0;dum2[np.isnan(dum2)] = 0; dum2[np.isinf(dum2)] = 0;dum2[dum2==0]=pow(10,-14);
		D1=np.multiply(np.ones(size(x))*tmp_PI[0],pGa); 
		D2=np.multiply(np.ones(size(x))*tmp_PI[1],dum2);
		D2[xneg]=0;
		D=D1+D2
		R1=np.divide(D1,np.ones(size(x))*D);
		R2=np.divide(D2,np.ones(size(x))*D);
		resp= sc.column_stack([R1,R2]);			
		#tmp_lik[it]=sum( np.multiply(resp[:,0],(log(tmp_PI[0])+log(pGa))) + np.multiply(resp[:,1],(log(tmp_PI[1])+log(dum2))));#bishop
		real_lik[it]=sum(log(np.multiply(tmp_PI[0],pGa)+  np.multiply(tmp_PI[1],dum2)))
		#N=np.ones(2)	
		#N[0]=sum(R1)
		#N[1]=sum(R2)
		#PI[0,0]=N[0]/sum(N)
		#PI[0,1]=N[1]/sum(N)
		#if it < maxiters-1:
		#M step
		N=np.ones(2)	
		N[0]=sum(R1)
		N[1]=sum(R2)
		tmp_PI[0]=N[0]/sum(N)
		tmp_PI[1]=N[1]/sum(N)
		#print tmp_PI
		#print tmp_lik[it]
		#update gaussian mean and variance
		tmp_mu1=[]
		tmp_mu1=sum(np.multiply(resp[:,0],x))/N[0];
		tmp_v1=[];
		tmp_v1=sum(np.multiply(resp[:,0],pow(x-tmp_mu1,2)))/N[0];
		#if tmp_v <= 0.5:
		#	tmp_v=0.5
		#UPDATE EACH INVERSE GAMMA. 
		tmp_mu2=[]
		tmp_mu2=sum(np.multiply(resp[:,1],x))/N[1];
		tmp_v2=[];
		tmp_v2=sum(np.multiply(resp[:,1],pow(x-tmp_mu2,2)))/N[1];
		if tmp_v2< 0.2:
			tmp_v2= 0.2

		if MM==2:	
			tmp_a1 =alb.alphaGm(tmp_mu2,tmp_v2)
			tmp_b1 =alb.betaGm(tmp_mu2,tmp_v2)
		elif MM==3:	
			tmp_a1 =alb.alphaIG(tmp_mu2,tmp_v2)
			tmp_b1 =alb.betaIG(tmp_mu2,tmp_v2)

		if it > 20: 
			if abs(real_lik[it]-real_lik[it-1]) < tol:
				flag=1
				print it
		
		if it > (maxiters-1):
				flag=1
				#print it

		if flag == 0:			
			mu1[it+1]=tmp_mu1;
			v1[it+1]=tmp_v1;
			mu2[it+1]=tmp_mu2;
			v2[it+1]=tmp_v2;
			a1[it+1]=alphaIG(tmp_mu2,tmp_v2);
			b1[it+1]=betaIG(tmp_mu2,tmp_v2);
			PI[it+1,:]=tmp_PI;

	stmu1=mu1[it];
	stv1=v1[it];
	stmu2=mu2[it];
	stv2=v2[it];
	stPI=PI[it,:];
	lik=tmp_lik[it];

	stPI=PI[it,:]
	return stmu1,stv1,stmu2,stv2,stPI,lik,it,resp;#,tmp_lik;# a1,b1,a2,b2,mu,v,PI,tmp_lik;

def SIN_init_VB_MM(data,opts):
	import scipy.special as sp;import copy
	#SIN_init_VB_MM does
	#              - fit a mixture model using ML (EM + MM algorithms (mmfit.m))
	#              - initialize VB parameters of mixture model using EM fit as
	#              initial posteriors
	#inputs: -data : vector normalized to mean zero and unit std
	#	 -opts: list with options and values 
	#		-MM = GIM or GGM( default =GIM)
	#		-MLMMits = max number of iterations allowed to ML algorithm before initialize VB (default =1)
	#		-MLMMtol = tolerance for convergence of ML algorithm before initialize VB
	#ouptput: mix1 is a list containg the initialized priors, and the posterior estimations given the ML initialization 
	#example:opts=[];opts.append({'MM': 'GIM', 'MLMMits': 1, 'MLMMtol': 10^-5});
	#mix=SIN_init_VB_MM(data,opts); 
	#From matlab...IN PROGRESS
	

	if 'MM' not in opts[0]:
		MM='GIM'
	else:
		MM=opts[0]['MM']

	if 'MLMMits' not in opts[0]:
		MLMMits=1
	else:
		MLMMits= opts[0]['MLMMits']

	if 'MLMMtol' not in opts[0]:
		MLMMtol=0.00001
	else:
		MLMMtol=opts[0]['MLMMtol']
	#SET PRIORS
	#set mixing priors.
	prior=[];
	mmm=10;#(the mean of component)
	vvv=10;#(the variance of the component)
	if MM =='GGM': 
		#set GAMMA prior on rates (shape and rate)
		Erate= np.true_divide(1,alb.betaGm(mmm,vvv));
		d_0= copy.deepcopy(Erate)
		e_0=1;
		Erate=np.true_divide(d_0,e_0);
		#set shapes conditional prior (fancy)
		Eshape=alb.alphaGm(mmm,vvv);
		dum_v=np.copy(Eshape);#allow variance on shape to be of size of mean shape
		dum_p=np.true_divide(1,dum_v);
		#from laplace approx b=prec/psi'(map(s))
		b_0=np.true_divide(dum_p,sp.polygamma(1,Eshape))
		c_0=copy.deepcopy(b_0);	
		loga_0=((b_0* sp.polygamma(0,Eshape))-(c_0*log(Erate))) 
	elif MM=='GIM':
		#set GAMMA prior on scale (shape d and rate e)
		Escale=alb.betaIG(mmm,vvv);
		d_0=copy.deepcopy(Escale);#shape
		e_0=1;#rate
		Escale=np.true_divide(d_0,e_0);
  
		#set component 2 and 3 shape conditional prior (fancy)
		Eshape=alb.alphaIG(mmm,vvv);
		dum_v=np.copy(Eshape);#allow variance on shape to be of size of mean shape
		dum_p=np.true_divide(1,dum_v);
		b_0=np.true_divide(dum_p,sp.polygamma(1,Eshape));#from laplace approx b=prec/psi'(map(s))
		c_0=copy.deepcopy(b_0);
		loga_0=(-(b_0* sp.polygamma(0,Eshape))+(c_0*np.log(Escale)))

	prior.append({'lambda_0': 5, 'm_0': 0,  'tau_0': 100,  'c0': 0.001, 'b0': 100 , 'd_0': d_0, 'e_0': e_0 , 'loga_0': loga_0, 'b_0': b_0 , 'c_0': c_0 })
	prior=prior[0]	


	#SET POSTERIORS initializations using ML mixture models
	if MM=='GGM':
		mmtype=2
	elif MM=='GIM':
		mmtype =3
	else:
		mmtype =1;#dummy, never used gmm 
	ML=[];
	[mu1,v1,mu2,v2, mu3,v3,pipi,lik,numits,resp]=alb.mmfit3(data, MLMMits,MLMMtol,mmtype)
	ML_param= [ mu1,v1,mu2,v2, mu3,v3]
	ML.append({'init': ML_param,'pi': pipi,'LIK': lik})
	ML=ML[0]

	#mix1.ML

	#INIT POSTERIORS BASED IN ML MIX MODEL
	post=[]
	#[dum; b]=max(resp)
	q=resp.argmax(1)
	gammas=np.copy(resp)
	#lambda=sum(resp,2)'

	lambdap = resp.sum(0)

	#COMPONENT 1: Gaussian component
	#hyperparam. on mean
	m0=ML_param[0]
	tau0=np.true_divide(1, np.true_divide(ML_param[0] +  ML_param[2] +  np.absolute(ML_param[4]),3))

	#hyperparam. on precission
	init_prec=np.true_divide(1,ML_param[1])
	init_var_prec=np.var([np.true_divide(1,ML_param[1]) ,  np.true_divide(1,ML_param[3]) , np.true_divide(1,ML_param[5]) ], ddof=1)
	c0=alb.alphaGm(init_prec,init_var_prec );#shape
	b0=alb.betaGm(init_prec,init_var_prec );#scale

	#COMPONENTS 2 AND 3: gamma or inverse gamma
	if MM=='GGM':
		#hyperparam. on rates
		init_rates=[  np.true_divide(1, alb.betaGm(np.absolute(ML_param[2]),ML_param[3])) ,   np.true_divide(1,alb.betaGm(np.absolute(ML_param[4]), ML_param[5]))  ]  ;
		dum_var_r= np.multiply(0.1,init_rates)#(init_rates)* 0.1;#    var(init_rates);
		d_0=alb.alphaGm(init_rates,dum_var_r);#shape
		e_0=np.true_divide(1, alb.betaGm(init_rates,dum_var_r));#rate
		Erates=np.true_divide(d_0,e_0) # == init_rates


		#hyperparam. on shapes
		init_shapes=[  alb.alphaGm(np.absolute(ML_param[2]), ML_param[3])  ,  alb.alphaGm(np.absolute(ML_param[4]), ML_param[5])  ]  ;
		#b_0=[1 1];c_0=b_0;  
		#b_0=sum(resp(2:3,:),2)';c_0=b_0;  
		b_0=resp.sum(0)[1:3]
		c_0=b_0
		#loga_0=((b_0* sp.polygamma(0,init_shapes)-(c_0*log(Erates))); 
		loga_0=np.multiply(b_0,sp.polygamma(0,init_shapes)) - (np.multiply(c_0,np.log(Erates)))
		#MAP_shapes=invpsi((loga_0+ (c_0 .* log(Erates))) ./ b_0) # == init_shapes

	elif MM=='GIM':
		#hyperparam. on scales (inverse gamma) --> scale is r in the text,
		#r ~ inv gamma distr
		init_scales=[  alb.betaIG(np.absolute(ML_param[2]), ML_param[3]) ,   alb.betaIG(np.absolute(ML_param[4]), ML_param[5])  ]  ;
		dum_var_sc=np.multiply(0.1,init_scales);#var(init_scales);
		d_0=alb.alphaGm(init_scales,dum_var_sc);#gamma shape
		e_0=np.divide(1, alb.betaGm(init_scales,dum_var_sc));#gamma rate
		Escales=np.divide(d_0,e_0); # == init_scales

		#hyperparam. on shapes
		init_shapes=[  alb.alphaIG(np.absolute(ML_param[2]), ML_param[3]) ,   alb.alphaIG(np.absolute(ML_param[4]), ML_param[5])  ]  ;
		#b_0=[1 1];c_0=b_0;  
		sumgam=resp.sum(0)
		b_0=sumgam[1:3]
		c_0=b_0;  
		loga_0=-np.multiply(b_0,sp.polygamma(0,init_shapes)) + (np.multiply(c_0,np.log(Escales)))
		#MAP_shapes=invpsi((-loga_0+ (c_0 .* log(Escales))) ./ b_0) # == init_shapes

	post.append({'lambda': lambdap, 'm_0': m0,  'tau_0': tau0,  'c0': c0, 'b0': b0 , 'd_0': d_0, 'e_0': e_0 , 'loga_0': loga_0, 'b_0': b_0 , 'c_0': c_0 })	
	post=post[0]
	#Save posterior expectations for initialization of VB mixModel

	mix1=[];
	if MM=='GGM':
		shapes=[0,0]
		rates=[0,0]
		#shapes=alphaGm(ML_param[2:3], ML_param(2:3,2))';
		#rates= 1./  betaGm(ML_param(2:3,1), ML_param(2:3,2))' ;
		shapes[0]=alb.alphaGm(abs(ML_param[2]), ML_param[3])
		shapes[1]=alb.alphaGm(abs(ML_param[4]), ML_param[5])
		rates[0]=np.divide( 1,  betaGm(abs(ML_param[2]), ML_param[3])) 
		rates[1]=np.divide( 1,  betaGm(abs(ML_param[4]), ML_param[5])) 
		mix1.append({'gammas': resp, 'lambda': lambdap,'pi': pipi,'mu1': ML_param[0], 'tau1':np.true_divide(1, ML_param[1]),'shapes': shapes,'rates': rates,'q': q,'prior': prior,'post':post,'ML': ML ,'opts': opts})
	elif MM=='GIM':
		shapes=[0,0]
		scales=[0,0]
	    	shapes[0]=[alphaIG(abs(ML_param[2]), ML_param[3])] 
		shapes[1]=[alphaIG(abs(ML_param[4]), ML_param[5])] 
		scales[0]=  [betaIG(abs(ML_param[2]), ML_param[3])];#   betaIG(ML_param(3,1), ML_param(3,2))  ];
		scales[1]=  [betaIG(abs(ML_param[4]), ML_param[5])];
		mix1.append({'gammas': resp, 'lambda': lambdap,'pi': pipi,'mu1': ML_param[0],'tau1': np.true_divide(1,ML_param[1]),'shapes': shapes,'scales': scales,'q': q,'prior': prior,'post':post,'ML': ML ,'opts': opts })
		
	mix1=mix1[0]




	return mix1

def   invpsi(X): 
# Inverse digamma (psi) function.  The digamma function is the derivative of the log gamma function.  
#This calculates the value Y > 0 for a value X such that digamma(Y) = X.
#This algorithm is from Paul Fackler: http://www4.ncsu.edu/~pfackler/
	import scipy.special as sp  
	import numpy as np 
	L = 1;
	Y = np.exp(X);
	while L > 10e-9:
		Y = Y + L*np.sign(X-sp.polygamma(0,Y));
		L = np.true_divide(L , 2);
	return Y

def f2(b,alpha):
	import scipy.special as sp  
	out= np.multiply(b , sp.polygamma(1,alpha));# Necesaty for Laplace approx in VB
	return out



