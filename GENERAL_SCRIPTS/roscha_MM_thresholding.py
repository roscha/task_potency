import numpy as np
import scipy.stats
import scipy.integrate
import alb_MM_functions as alb

def selectionMM(mat,step=50,iteration=3,method='weighted_pFDR'):
	#mat should be a 2D matrix NxN
    number=len(mat)
    def norm2(x,stmu1,stv1):
        return scipy.integrate.quad(norminit,x,np.inf,args=(stv1,stmu1))[0]
    def norminit(x,stv1,stmu1):
        return scipy.stats.norm(stmu1,stv1).pdf(x)
    vecnorm2=np.vectorize(norm2)
    pmin=0.025
    maxiters=100
    tol=0.000001 #relative tolerance for convergence
    MM=2 #2 is'GGM', 3 is 'GIM'
    #initial normalization
    try:
        data=mat[np.triu_indices(number,1)]
        tri=1
    except:
        data=copy.deepcopy(mat)
        tri=0
    mini,stini=[np.mean(data),np.std(data)]
    data=(data-mini)/stini
    #run mm            
    output=alb.mmfit3(data, maxiters,tol,MM)
    stmu1,stv1=[output[0],output[1]]
    if method=='symmetric':
        init=stmu1+2*stv1
        last=np.max(np.abs(data))
        for i in range(iteration):
            t2=np.arange(init,last+(last-init)/step,(last-init)/step)
            t=t2[:step]
            #find the values closest to the threshold (above and bellow 0.05)
            test=np.divide(2*vecnorm2(t,stmu1,stv1) , (np.mean(np.repeat([np.abs(data- stmu1)],step,0).T>=t-stmu1,0))  ) < pmin
            
            limthr=np.max(np.where(test==False))
            init=t2[limthr]
            last=t2[limthr+1]
        

        selectionfromtriuIndex=[np.where(np.abs(data-stmu1)>init-stmu1)]
        tmax=np.min(mat[np.triu_indices(number,1)][np.where(mat[np.triu_indices(number,1)]>(init*stini+mini))])
        tmin=np.max(mat[np.triu_indices(number,1)][np.where(mat[np.triu_indices(number,1)]<((-init+2*stmu1)*stini+mini))])
    else:
        if method=='equal_pFDR':
            pmin1=0.025
            pmin2=0.025
        elif method=='weighted_pFDR':
            pmin1=0.05*output[6][1]/(output[6][1]+output[6][2])
            pmin2=0.05*output[6][2]/(output[6][1]+output[6][2])
        #up
        init=stmu1+2*stv1
        last=np.max(data)
        try:
            for i in range(iteration):
                t2=np.arange(init,last+(last-init)/step,(last-init)/step)
                t=t2[:step]
            #find the values closest to the threshold
                test=np.divide(vecnorm2(t,stmu1,stv1) , (np.mean(np.repeat([(data- stmu1)],step,0).T>=t-stmu1,0))  ) < pmin1
            
                if test[len(test)-1]==False:
                    limthrMax=len(test)-1
                else:
                    limthrMax=np.max(np.where(test==False))
                init=t2[limthrMax]
                last=t2[limthrMax+1]
        except:
            init=init
        #down
        
        init2=-(stmu1-2*stv1)
        last2=np.max(-data)
        try:
            for i in range(iteration):
                t2=np.arange(init2,last2+(last2-init2)/step,(last2-init2)/step)
                t=t2[:step]
            #find the values closest to the threshold

                test=np.divide(vecnorm2(t,stmu1,stv1) , (np.mean(np.repeat([-(data- stmu1)],step,0).T>=t+stmu1,0))  ) < pmin2
            
                if test[len(test)-1]==False:
                    limthrMin=len(test)-1
                else:
                    limthrMin=np.max(np.where(test==False))
            
                init2=t2[limthrMin]
                last2=t2[limthrMin]+(last2-init2)/step
        except:
            init2=init2
        

        selectionfromtriuIndex=[np.where((data>init)+(data<-init2)==1)]
		prov=np.zeros((number,number)) #prov would be a binary matrix with selected edges
		prov2=np.zeros(len(np.triu_indices(number,1)))
		prov2[selectionfromtriuIndex]=1
		prov[np.triu_indices(number,1)]=prov2
		prov=prov+prov.T
        try:
            if tri==1:
                tmax=np.min(mat[np.triu_indices(number,1)][np.where(mat[np.triu_indices(number,1)]>(init*stini+mini))])
            else:
                tmax=np.min(mat[np.where(mat>(init*stini+mini))])
        except:
            tmax=np.inf
        try:
            if tri==1:
                tmin=np.max(mat[np.triu_indices(number,1)][np.where(mat[np.triu_indices(number,1)]<((-init2+2*stmu1)*stini+mini))])
            else:
                tmin=np.max(mat[np.where(mat<((-init2+2*stmu1)*stini+mini))])
        except:
            tmin=-np.inf
    
    return prov,selectionfromtriuIndex,tmin,tmax
