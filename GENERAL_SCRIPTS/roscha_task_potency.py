#!/home/mrstats/maamen/Software/python/bin/python2.7


import sklearn
import sklearn.covariance
import numpy as np

#function
def cov_to_corr(M):
    M2=M * np.diag(M) ** (-1./2) * (np.diag(M) ** (-1./2))[...,np.newaxis]
    for i in range(len(M2)):
        M2[i][i]=1
    return M2

def corr_to_arc(M,l): 
    Z=M 
    Z2=np.diag(np.diag(Z))
    Z[np.where(Z2!=0)]= np.nan
    Z = np.arctanh(Z)
    Z[np.where(Z2!=0)]=1.
    return Z

def corr_to_Z(M,l):
    Z=M 
    Z2=np.diag(np.diag(Z))
    Z[np.where(Z2!=0)]= np.nan
    Z = np.arctanh(Z)*np.sqrt(l-3)
    Z[np.where(Z2!=0)]=1.
    return Z
    
def makemat(ts,method):
    if method=="basic":
        shr=sklearn.covariance.ShrunkCovariance()
        shr=shr.fit(ts.T)
        covM=shr.covariance_
        parCor=shr.precision_
    elif method=="ledoit_wolf":
        covM,lw=sklearn.covariance.ledoit_wolf(ts.T)
        parCor=np.linalg.inv(covM)
        tmp = np.tile(np.sqrt(abs(np.diag(parCor))),[len(covM),1])
        parCor=parCor / (tmp*tmp.T)
    elif method=="":
        covM=np.cov(ts)
        parCor=covM/np.sqrt(np.average(np.diag(covM)**2))
        alpha=.5#??
        parCor=np.linalg.inv(covM+ alpha * np.eye(len(covM)))#(cov_to_corr(np.linalg.inv(corM)))
        tmp = np.tile(np.sqrt(abs(np.diag(parCor))),[len(covM),1])
        parCor=parCor / (tmp*tmp.T)
    else:
       
        print "method not recognized"
        
    arcparCor=corr_to_arc(parCor,len(ts[0]))
    ZparCor=corr_to_Z(parCor,len(ts[0]))
    corM=np.corrcoef(ts)
    if method=="ledoit_wolf":
        r=[corM,ZparCor,arcparCor,covM,parCor,lw]
    else:
        r=[corM,ZparCor,arcparCor,covM,parCor]
    return r

# do regression
def regression(data, design, mask, demean=True, desnorm=False, resids=False):
    
    import numpy as np
        
    # Y = Xb + e
    # process Y
    Y = data[mask==1,:]
    
    # process X
    if design.shape[0] == Y.shape[1]:
        X = design
    else:
        X = design[mask==1,:]
        
    
    if demean == True:
        #demean Y
        if Y.shape[0] == X.shape[0]:
            Y = Y - np.average(Y,axis=0)
        else:
            # demean the data, subtract mean over time from each voxel
            Y = Y - np.tile(np.average(Y, axis=1), (Y.shape[1],1)).T
    
        # demean the design
        X = X - X.mean(axis=0)
    
    if desnorm == True:
        # variance normalize the design
        X = X/X.std(axis=0, ddof=1)

    # add constant to X
    constant = np.ones(X.shape[0])
    X = np.column_stack((constant,X))
    
    if Y.shape[1] == X.shape[0]:
        # put time in rows for regression against time course
        Y = Y.T
    
    # obtain betas
    B = np.linalg.pinv(X).dot(Y)
    # obtain residuals
    #print Y.shape, X.shape, B.shape
    eta = Y - X.dot(B)
    #print eta.shape
    
    # put betas back into image if needed
    if max(B.shape) == max(Y.shape):
        bi = np.zeros((B.shape[0],max(data.shape)))
        bi[:,mask==1] = B
        B = bi
    
    # put residuals back into image
    if resids == True:
        ei = np.zeros_like(data)
        ei[mask==1,:] = eta.T
        eta = ei
        
    # return betas and design
    # discard first beta, this is the constant
    if resids == True:
        return B[1:,:], eta
    else:
        return B[1:,:]

def potency(niftipathtask,niftipathrest,maskpathtask,maskpathrest,atlas4dpath,potency='indiv',savepath='',savelevel=1):
    '''author . roselyne chauvin
        niftipathtask : list of path path or path to the 4d nifti file corresponding to the preprocessed task acquisition
        niftipathrest : list of path or one path to 4d nifti file(s) corresponding to the preprocessed resting state acquisition of the same subject or to the population of interest
        potency : can be 'indiv' for 'individual potency' as the substraction of rest connectivity (**niftipathrest is a path to a unique file**) or 'population' for population potency' as the standardization by the resting distribution (**niftipath is a list of path**)
        atlas4dpath : path to the nifti file with the atlas (one area per volume) 
        savepath : folder location to save a .npy file with the potency matrix
        savelevel : if ==2 will return rest and task normalized matrices
        '''
    import sys
    import os, glob, random
    import numpy as np
    import subprocess
    import nibabel
    ##parameters for mixture modelling
    sys.path.append('/home/mrstats/roscha/Scripts/TEN/scripts/')
    import alb_MM_functions as alb
    maxiters=100
    tol=0.000001 #relative tolerance for convergence
    MM=2 #2 is'GGM', 3 is 'GIM'


        ##verification
    if potency=='indiv' and type(niftipathtask)==list:
        if len(niftipathtask)!=len(niftipathrest):
            print stop
    if type(niftipathtask)!=list:
        niftipathtask=np.array([niftipathtask])
    if type(maskpathtask)!=list:
        maskpathtask=np.array([maskpathtask])
    if type(maskpathrest)!=list:
        maskpathrest=np.array([maskpathrest])
    if type(niftipathrest)!=list:
        niftipathrest=np.array([niftipathrest])
    if type(atlas4dpath)!=list:
        atlas_path=atlas4dpath
        area=nibabel.load(atlas_path).shape
        indivatlas=False
    else:
        indivatlas=True
        area=nibabel.load(atlas_path[0]).shape
            
        #for each nifti task file if more than one
    taskmat=np.zeros((len(niftipathtask),area[3],area[3]))
    for i,f in enumerate(niftipathtask):

            #read data
        if indivatlas==True:
            atlas_path=atlas4dpath[i]
        a= nibabel.load(atlas_path).get_data()
        number=a.T.shape[0]
        tri=np.zeros((number,number))
        tri[np.triu_indices(number,1)]=1
        d= nibabel.load(niftipathtask[i]).get_data()
        m= nibabel.load(maskpathtask[i]).get_data()
            
            #extract time serie from atlas (reg weitghted)
        try:
            prov=regression(d, a, m)
        
            #demean
            ts=(prov-np.repeat([np.mean(prov,1)],len(prov[0]),axis=0).T)/(np.repeat([np.std(prov,1)],len(prov[0]),axis=0).T)
        #compute ledoit wolf covariance
        #compute partial correlation
        #compute Zpartial correlation
            [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(ts,'ledoit_wolf')
        
        #mixture modelling on it
            prov=ZparCor
            
            prov2=prov[np.triu_indices(number,1)]  
            prov2_norm=(prov2-np.mean(prov2))/np.std(prov2)   
            output=alb.mmfit3(prov2_norm, maxiters,tol,MM)   
            m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
            var_gaus=(np.std(prov2)**2)*output[1] 

                
        #normalize matrix by main gaussian
            prov[np.triu_indices(number,1)]=((prov[np.triu_indices(179,1)]-m_gaus)/np.sqrt(var_gaus))      



            prov=prov*tri+(prov*tri).T
        #clean up
            taskmat[i]=prov
        except:
            print 'fail'

        #redo that for each resting state (or just one)
    restmat=np.zeros((len(niftipathrest),area[3],area[3]))
    if niftipathrest!='':
        for i,f in enumerate(niftipathrest):
        
            #read data
            if indivatlas==True:
                atlas_path=atlas4dpath[i]
            a= nibabel.load(atlas_path).get_data()
            number=a.T.shape[0]
            tri=np.zeros((number,number))
            tri[np.triu_indices(number,1)]=1
            d= nibabel.load(niftipathrest[i]).get_data()
            m= nibabel.load(maskpathrest[i]).get_data()
            
            #extract time serie from atlas (reg weitghted)
            prov=regression(d, a, m)
        
            #demean
            ts=(prov-np.repeat([np.mean(prov,1)],len(prov[0]),axis=0).T)/(np.repeat([np.std(prov,1)],len(prov[0]),axis=0).T)
        #compute ledoit wolf covariance
        #compute partial correlation
        #compute Zpartial correlation
            [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(ts,'ledoit_wolf')
        
        #mixture modelling on it
            prov=ZparCor
            prov2=prov[np.triu_indices(number,1)]  
            prov2_norm=(prov2-np.mean(prov2))/np.std(prov2)   
            output=alb.mmfit3(prov2_norm, maxiters,tol,MM)   
            m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
            var_gaus=(np.std(prov2)**2)*output[1] 

                
        #normalize matrix by main gaussian
            prov[np.triu_indices(number,1)]=((prov[np.triu_indices(179,1)]-m_gaus)/np.sqrt(var_gaus))      
            prov=prov*tri+(prov*tri).T
        #clean up
            restmat[i]=prov

    if savelevel==2:
        if savepath!='':
            np.save(savepath+'/rest_normalizedMat.npy',restmat)
            np.save(savepath+'/task_normalizedMat.npy',taskmat)
        return [taskmat,restmat]
    else:        
        #potency 
        if potency=='indiv' : #substract one by one
            if savepath!='':
                np.save(savepath+'/indivStandardized_potency_mat.npy',[taskmat[i]-restmat[i] for i in range(taskmat)])
            return [taskmat[i]-restmat[i] for i in range(taskmat)]
        else:
                #compute mean and standard deviation per edge and normaliwe task matrix one after another

            [m,st]=[np.mean(restmat,0),np.std(restmat,0)]
            if savepath!='': 
                np.save(savepath+'/groupStandardized_potency_mat.npy',[(taskmat[i]-m)/st for i in range(len(taskmat))])
            return [(taskmat[i]-m)/st for i in range(len(taskmat))]
       
