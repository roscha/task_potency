#!/home/mrstats/maamen/Software/python/bin/python2.7


#example use:
#import sys
#sys.path.append('folderpathwherethisscriptislocated')
#rc_dyn.dynamic_potency(['/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-Stop_run-1_echo-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz','/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-Stop_run-1_echo-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz'],'/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-rs_run-1_echo-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz' ,'/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-Stop_run-1_echo-1_space-MNI152NLin6Asym_desc-brain_mask.nii.gz' ,'/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-rs_run-1_echo-1_space-MNI152NLin6Asym_desc-brain_mask.nii.gz', '/project/3022011.02/NeuroIMAGE/atlases/ICPAtlas_v4_fine_208parcels_4Dbis.nii.gz' ,2300,2300,task='related',savepath='/project/3022043.01/DELTA/3T/fmriprep/test' ,toreturn=2 )




import sklearn
import sklearn.covariance
import numpy as np
import sys
sys.path.append('/home/mrstats/roscha/Scripts/TEN/scripts/')
import alb_MM_functions as alb
import roscha_MM_thresholding as rcthresh###**###

#function
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

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

#if True:
#    niftipathtask=['/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-Stop_run-1_echo-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz','/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-Stop_run-1_echo-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz']
#    niftipathrest='/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-rs_run-1_echo-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz' 
#    maskpathtask='/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-Stop_run-1_echo-1_space-MNI152NLin6Asym_desc-brain_mask.nii.gz' 
#    maskpathrest='/project/3022043.01/DELTA/3T/fmriprep/fmriprep/sub-026/ses-mri01/func/sub-026_ses-mri01_task-rs_run-1_echo-1_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
#    atlas4dpath= '/project/3022011.02/NeuroIMAGE/atlases/ICPAtlas_v4_fine_208parcels_4Dbis.nii.gz' 
#    TRrest=2300
#    TRtask=2300
#    task='related'
#    savepath='/project/3022043.01/DELTA/3T/fmriprep/test'
#    toreturn=2 
#    savecontent=[1,2,3,4]
def dynamic_potency(niftipathtask,niftipathrest,maskpathtask,maskpathrest,atlas4dpath,TRrest,TRtask,name='dynamic_potency',task='indep',savepath='',toreturn=0, savecontent=[1,2,3,4]):
    '''author . roselyne chauvin
        potency is calculated for a single subject, the function needs to be run for each subject but a list of task/block can be computed at once
        task: can be "indep" independant list of tasks, or "related" one single fixed matrix will be calculated from concatenation of task list, dynamic will be calculated for each items of the task list (example use : block design)        
        niftipathtask : list of path or path to the 4d nifti file corresponding to the preprocessed task acquisition 
        niftipathrest : path to 4d nifti file corresponding to the preprocessed resting state acquisition of the subject
        
        atlas4dpath : path to the nifti file with the atlas (one area per volume) 
        TRrest : TR of the rest acquisition in ms
        TRtask : TR of the task acquisition in ms. If list of task with different TR, provide a list of TR []
        savepath : folder location to .txt files and .npy files
        toreturn : 0 save files 1 return files 2 both save and return
        savecontent :  1  task dynamic potency 2  task and rest dynamic 3  task and rest fixed matrices and 4 ledoit_wolf convergence parameter 
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
    
    if type(niftipathtask)!=list:
        niftipathtask=np.array([niftipathtask])
    if type(maskpathtask)!=list:
        maskpathtask=np.array([maskpathtask])
    if len(maskpathtask)==1:
        maskpathtask=np.array(np.repeat(maskpathtask,len(niftipathtask)))
    else:
        if len(niftipathtask)!=len(maskpathtask):
            print('mismatch between len of task list and len of mask list')
            exit()
    #if type(maskpathrest)!=list:
    #    maskpathrest=np.array([maskpathrest])#necessary?
    
        
    if type(TRtask)!=list:
        TRtask=np.array(np.repeat(TRtask,len(niftipathtask)))
    else:
        if len(niftipathtask)!=len(TRtask):
            print('mismatch between len of task list and len of TR list')
            exit()
    
            
    area=nibabel.load(atlas4dpath).shape
                
        #for each nifti task file if more than one
    
    a= nibabel.load(atlas4dpath).get_data()
    number=a.T.shape[0]
    tri=np.zeros((number,number))
    tri[np.triu_indices(number,1)]=1
        
    ####### compute Time Series and fixed connectivity matrices
    ts=[[] for i in range(len(niftipathtask))]
    for i,f in enumerate(niftipathtask):

            #read data
        
        
        d= nibabel.load(niftipathtask[i]).get_data()
        m= nibabel.load(maskpathtask[i]).get_data()
            
            #extract time serie from atlas (reg weitghted)
        prov=regression(d, a, m)
        
            #demean
        ts[i]=(prov-np.repeat([np.mean(prov,1)],len(prov[0]),axis=0).T)/(np.repeat([np.std(prov,1)],len(prov[0]),axis=0).T)
    #compute ledoit wolf covariance
    #compute partial correlation
    #compute Zpartial correlation
    
    if task=='indep':
        taskmat=np.zeros((len(niftipathtask),number,number))
        taskmatfordyn=np.zeros((len(niftipathtask),number,number))
        lwALL=np.zeros(len(niftipathtask))        
        for i,f in enumerate(niftipathtask):
            [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(ts[i],'ledoit_wolf')
            taskmatfordyn[i]=parCor
            #mixture modelling on it
            prov=ZparCor
            
            prov2=prov[np.triu_indices(number,1)]  
            prov2_norm=(prov2-np.mean(prov2))/np.std(prov2)   
            output=alb.mmfit3(prov2_norm, maxiters,tol,MM)   
            m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
            var_gaus=(np.std(prov2)**2)*output[1] 

                
            #normalize matrix by main gaussian
            prov[np.triu_indices(number,1)]=((prov[np.triu_indices(number,1)]-m_gaus)/np.sqrt(var_gaus))      



            prov=prov*tri+(prov*tri).T
        #clean up
            taskmat[i]=prov
            
            lwALL[i]=lw
        
    if task=='related':
        [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(np.concatenate(ts,axis=1),'ledoit_wolf')
        taskmatfordyn=parCor        
            #mixture modelling on it
        prov=ZparCor
            
        prov2=prov[np.triu_indices(number,1)]  
        prov2_norm=(prov2-np.mean(prov2))/np.std(prov2)   
        output=alb.mmfit3(prov2_norm, maxiters,tol,MM)   
        m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
        var_gaus=(np.std(prov2)**2)*output[1] 

                
            #normalize matrix by main gaussian
        prov[np.triu_indices(number,1)]=((prov[np.triu_indices(number,1)]-m_gaus)/np.sqrt(var_gaus))      



        prov=prov*tri+(prov*tri).T
        #clean up
        taskmat=prov
            
        lwALL=lw

    ######redo that for the resting state
    
            #read data
        
        
    d= nibabel.load(niftipathrest).get_data()
    m= nibabel.load(maskpathrest).get_data()
            
            #extract time serie from atlas (reg weitghted)
    prov=regression(d, a, m)
        
            #demean
    tsrest=(prov-np.repeat([np.mean(prov,1)],len(prov[0]),axis=0).T)/(np.repeat([np.std(prov,1)],len(prov[0]),axis=0).T)
    #compute ledoit wolf covariance
    #compute partial correlation
    #compute Zpartial correlation
    

    [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(tsrest,'ledoit_wolf')
    restmatfordyn=parCor
            #mixture modelling on it
    prov=ZparCor
            
    prov2=prov[np.triu_indices(number,1)]  
    prov2_norm=(prov2-np.mean(prov2))/np.std(prov2)   
    output=alb.mmfit3(prov2_norm, maxiters,tol,MM)   
    m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
    var_gaus=(np.std(prov2)**2)*output[1] 

                
            #normalize matrix by main gaussian
    prov[np.triu_indices(number,1)]=((prov[np.triu_indices(number,1)]-m_gaus)/np.sqrt(var_gaus))      



    prov=prov*tri+(prov*tri).T
        #clean up
    restmat=prov
            
    lwrest=lw

        
        
    ###### compute Multiplication of Temporal Derivative
    #compute temporal derivative 
    tsderiv=[np.zeros((number,len(ts[i][0])-1)) for i in range(len(ts))]
    tstime=[np.zeros(len(ts[i][0])-1) for i in range(len(ts))]
    for i,f in enumerate(ts):
        
        tsderiv[i]=np.array([ts[i].T[n]-ts[i].T[n-1] for n in range(1,len(ts[i][0]))]).T
        tstime[i]=[(n-0.5)*TRtask[i] for n in range(1,len(ts[i][0]))]
        
    tsderivrest=np.array([tsrest.T[n]-tsrest.T[n-1] for n in range(1,len(tsrest[0]))]).T
    tstimerest=[(n-0.5)*TRrest for n in range(1,len(tsrest[0]))]
        
    
    #compute multiplication of temporal derivative
    mtd=[np.zeros((len(ts[i][0])-1,number,number)) for i in range(len(ts))]
    
    for i,f in enumerate(tsderiv):    
        for n in range(len(tsderiv[i][0])):

            prov=np.outer(tsderiv[i].T[n],tsderiv[i].T[n])
            if task=="indep":
                prov3=corr_to_arc(-taskmatfordyn[i]*(prov)*taskmatfordyn[i],2)
            if task=='related':
                prov3=corr_to_arc(-taskmatfordyn*(prov)*taskmatfordyn,2)

                    #MMnormalized
            prov2=prov3[np.triu_indices(number,1)]  

            prov2[np.where(np.isnan(prov2))]=0#if the atlas is well defined, this should not be necessary
            prov2_norm=(prov2-np.mean(prov2))/np.std(prov2) 
            output=alb.mmfit3(prov2_norm, maxiters,tol,MM) 
            m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
            var_gaus=(np.std(prov2)**2)*output[1] 

                
                    #normalize matrix by main gaussian
            
            prov2=((prov2-m_gaus)/np.sqrt(var_gaus)) 
            prov3=np.zeros((number,number))
            prov3[np.triu_indices(number,1)]=prov2
            mtd[i][n]=prov3*tri+(prov3*tri).T
    
    mtdrest=np.zeros((len(tsrest[0])-1,number,number))
    
        
    for n in range(len(tsderivrest[0])):
            
        prov=np.outer(tsderivrest.T[n],tsderivrest.T[n])
        prov3=corr_to_arc(-restmatfordyn*(prov)*restmatfordyn,2)
            
                    #MMnormalized
        prov2=prov3[np.triu_indices(number,1)]  
        prov2[np.where(np.isnan(prov2))]=0#if the atlas is well defined, this should not be necessary

        prov2_norm=(prov2-np.mean(prov2))/np.std(prov2)   
        output=alb.mmfit3(prov2_norm, maxiters,tol,MM)   
        m_gaus=np.std(prov2)*output[0]+np.mean(prov2)     
        var_gaus=(np.std(prov2)**2)*output[1] 

                
                    #normalize matrix by main gaussian
        prov2=((prov2-m_gaus)/np.sqrt(var_gaus)) 
        prov3=np.zeros((number,number))
        prov3[np.triu_indices(number,1)]=prov2
        mtdrest[n]=prov3*tri+(prov3*tri).T


        
    ## compute the dynamic task potency
    meanrest=np.mean(mtdrest,0)
    stdrest=np.std(mtdrest,0)
    
    dynamictask=[[(i - meanrest)/stdrest for i in mtd[n]] for n in range(len(mtd))]
    
    ##adjust sign: we want to express the change in task connectivity relateive to rest
	#define rest connection that are positive, negative or no signal
	[selection,tmin,tmax]=rcthresh.selectionMM(restmat)###**###
    signrestmat=np.zeros((number,number))###**###
	prov=np.zeros(len(np.triu_indices(number,1)|0]))###**###
	prov=np.sign(restmat)[np.triu_indices(number,1)]###**###
	prov[selection]=np.sign(restmat)[np.triu_indices(number,1)][selection]###**###
	signrestmat[np.triu_indices(number,1)]=prov###**###
	signrestmat=signrestmat+signrestmat.T###**###
	antisign=np.zeros((number,number))###**###
	antisign[np.where(signrestmat==0)]=1###**###
	#adjust sign of dynamic potency:
	sign_dynamictask=[[(np.abs(i)*antisign + i*signrestmat) for i in mtd[n]] for n in range(len(mtd))]###**###
	
	
    #savecontent :  1  task dynamic potency 2  task and rest dynamic 3  task and rest fixed matrices and 4 ledoit_wolf convergence parameter 
    resultdict=dict()
    
    for n in range(len(niftipathtask)):
        resultdict['task_'+np.str(n)]={'path':niftipathtask[n]}
###
        resultdict['task_'+np.str(n)].update({'tsderiv':tsderiv[n]})
        
        if 1 in savecontent:
            resultdict['task_'+np.str(n)].update({'dynamic_task_potency':dynamictask[n]})
			resultdict['task_'+np.str(n)].update({'sign_adjusted_dynamic_task_potency':sign_dynamictask[n]}) ###**###
            resultdict['task_'+np.str(n)].update({'timing_task':tstime[n]})
        if 2 in savecontent:
            resultdict['task_'+np.str(n)].update({'pseudo_dynamic_partialcorrelation_task':mtd[n]})
        if 3 in savecontent:
            if task=='related':
                resultdict['task_'+np.str(n)].update({'normalized_Z_partialcorrelation_task':taskmat})
            if task=='indep':
                resultdict['task_'+np.str(n)].update({'normalized_Z_partialcorrelation_task':taskmat[n]})
        if 4 in savecontent:
            if task=='related':
                resultdict['task_'+np.str(n)].update({'ledoitwolf_convergence_parameter':lwALL})
            else:
                resultdict['task_'+np.str(n)].update({'ledoitwolf_convergence_parameter':lwALL[n]})
    resultdict['rest']={'path':niftipathrest}
###    
    resultdict['rest'].update({'tsderivrest':tsderivrest})
    resultdict['rest'].update({'sign_connectivity_rest':signrestmat})###**###
    if 2 in savecontent:
        resultdict['rest'].update({'pseudo_dynamic_partialcorrelation_rest':mtdrest})
        resultdict['rest'].update({'timing_rest':tstimerest})
    if 3 in savecontent:
        resultdict['rest'].update({'normalized_Z_partialcorrelation_rest':restmat})
    if 4 in savecontent:
        resultdict['rest'].update({'ledoitwolf_convergence_parameter':lwrest})
    
    if toreturn==0 or toreturn==2:
        save_obj(resultdict,savepath+name)
    if toreturn==1 or toreturn==2:
        return resultdict
    
    