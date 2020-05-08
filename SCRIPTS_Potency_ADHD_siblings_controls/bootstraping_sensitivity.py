#!/home/mrstats/maamen/Software/python/bin/python2.7

# In[ ]:

import sys
sys.path.append('/home/mrstats/maamen/DCCN/Scripts/NeuroImage/')
import mm_helpers
import argparse, os, glob, random,sys
import numpy as np
import mm_GetInfoFromDb
import subprocess
import nibabel
sys.path.append('/home/mrstats/roscha/Scripts/TEN/scripts/')
import rc_helpers
import copy
import random
import matplotlib.pyplot as plt
import rc_GetInfoFromDb
import numpy as np
import seaborn as sns

import scipy.stats
import scipy.integrate
import alb_MM_functions as alb

import scipy

# parameter

# In[ ]:


import os, subprocess, argparse
#-----------------
parser = argparse.ArgumentParser(
                    description='''Script to bootstrap for the taskpotency submission
                    You can run from to commandline  or run from within ipython, where you have allinfo defined, run like this %run -i ~/analyse....py [options]''')
# required options                    
reqoptions = parser.add_argument_group('Required Arguments')
reqoptions.add_argument('-instance', action="store", dest="instance", required=True, help='de 0 to 19')


# parse arguments
args = parser.parse_args()
##-----------------------------------------------------------------------------
instance=args.instance

# In[3]:

cond='179'
atlas='atlas_complete_GM'

matrices='ZpartialCorrelation'
method='_ledoit_wolf'
family=0




# find and read data

# In[4]:

data=['/home/mrstats/roscha/NeuroIMAGE/preproc/fMRI_RS/SUBJECT/matrices/preprocessing.feat/times_series/ICA_AROMA_nonaggr_nr_hp_'+atlas+'_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160NORM95.txt',
            '/home/mrstats/roscha/NeuroIMAGE/preproc/fMRI_STOP/SUBJECT/matrices.feat/nuissance/denoised_func_data_nonaggr/times_series/denoised_func_data_nonaggr_hp_'+atlas+'_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160NORM95.txt',
            '/home/mrstats/roscha/NeuroIMAGE/preproc/fMRI_REWARD/SUBJECT/matrices.feat/nuissance/denoised_func_data_nonaggr/times_series/denoised_func_data_nonaggr_hp_'+atlas+'_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160NORM95.txt',
            '/home/mrstats/roscha/NeuroIMAGE/preproc/fMRI_WM/SUBJECT/matrices.feat/nuissance/denoised_func_data_nonaggr/times_series/denoised_func_data_nonaggr_hp_'+atlas+'_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160NORM95.txt',
            ]
NeuroIMAGE=['RS','STOP','REWARD','WM']
fold='results'
#Subject selection
if True:
    suj=[np.unique([mm_helpers.find_subject(i,'NeuroIMAGE') for i in glob.glob(j.replace('SUBJECT','*').replace('CONDITION','*'))]) for j in data]
    sujCONTROL=[np.unique([mm_helpers.find_subject(i,'NeuroIMAGE') for i in glob.glob(j.replace('SUBJECT','*').replace('CONDITION','*'))]) for j in data]
    sujADHD=[np.unique([mm_helpers.find_subject(i,'NeuroIMAGE') for i in glob.glob(j.replace('SUBJECT','*').replace('CONDITION','*'))]) for j in data]
    sujSibADHD=[np.unique([mm_helpers.find_subject(i,'NeuroIMAGE') for i in glob.glob(j.replace('SUBJECT','*').replace('CONDITION','*'))]) for j in data]
    selec=len(NeuroIMAGE)
    allinfo=[dict() for t in range(selec)]
    allinfoADHD=[dict() for t in range(selec)]
    allinfoSibADHD=[dict() for t in range(selec)]
    allinfoCONTROL=[dict() for t in range(selec)]

    for t in range(selec):
        for s in suj[t]:
            allinfo[t].update({s : {s:s,'run':[]}})
            allinfoCONTROL[t].update({s : {s:s,'run':[]}})
            allinfoADHD[t].update({s : {s:s,'run':[]}})
            allinfoSibADHD[t].update({s : {s:s,'run':[]}})


        
        #if adhd==1:
        allinfoADHD[t]=rc_helpers.selectADHD(allinfoADHD[t])
        allinfoSibADHD[t]=rc_helpers.selectSibADHD(allinfoSibADHD[t])
        allinfoCONTROL[t]=rc_helpers.selectControlnotADHDsib(allinfoCONTROL[t])
        
        allinfo[t]=rc_helpers.check_movers(allinfo[t])
        allinfoCONTROL[t]=rc_helpers.check_movers(allinfoCONTROL[t])
        allinfoADHD[t]=rc_helpers.check_movers(allinfoADHD[t])
        allinfoSibADHD[t]=rc_helpers.check_movers(allinfoSibADHD[t])
        
        
        suj[t]=sorted(allinfo[t].keys())
        sujCONTROL[t]=sorted(allinfoCONTROL[t].keys())
        sujADHD[t]=sorted(allinfoADHD[t].keys())
        sujSibADHD[t]=sorted(allinfoSibADHD[t].keys())


# In[6]:

## read phenotype info and other

myquery2= """SELECT `Subject`,`age` FROM `Phenotype_info`"""#number,ScanAge,_derived
allinfo_age = rc_GetInfoFromDb.get_query(myquery2,['subject','age'],"neuroimage")

myquery2= """SELECT `Subject`,`Gender` FROM `Phenotype_info`"""
allinfo_gender = rc_GetInfoFromDb.get_query(myquery2,['Subject','gender'],"neuroimage")

queryT = "SELECT `subject_number`,`SeriesNumber`,`ScanProtocol`,`rmsFD_Jenkinson`  FROM `MRI_series_info` "
allinfo_rms=rc_GetInfoFromDb.get_query(queryT, ['subject','sn','sp','jen'], 'neuroimage', True)
#myquery2= """SELECT `subject_number`,`rmsFD_Jenkinson` FROM `MRI_series_info`"""
#allinfo_rms = mm_GetInfoFromDb.get_query(myquery2,['subject','rms'],"neuroimage")
queryT = "SELECT `subject_number`,`TR_fMRI`,`TR_rsfMRI`  FROM `MRI_session_info` "
allinfo_TR=rc_GetInfoFromDb.get_query(queryT, ['subject','tr','trrs'], 'neuroimage', True)

queryT = "SELECT `Subject`,`Family`  FROM `Phenotype_info` "
allinfo_family=rc_GetInfoFromDb.get_query(queryT, ['subject','family'], 'neuroimage', True)


# In[7]:

##load matrices for ADHD

UpmatricesADHD=[[] for t in range(len(NeuroIMAGE))]

tri=np.zeros((number,number))
tri[np.triu_indices(number,1)]=1
keepsujADHD=[[] for i in range(len(sujADHD))]
familyADHD=[[] for i in range(len(sujADHD))]

for t in range(len(NeuroIMAGE)):
    
    for s in sujADHD[t]:
        dat=glob.glob(data[t].replace('SUBJECT',s))
        for f in dat:
            do=1
            if s in sujADHD[0]:
                prov=np.loadtxt(f)
                provrest=np.loadtxt(glob.glob(data[0].replace('SUBJECT',s))[0])
            else:
                do=0
            if do==1:
                keepsujADHD[t]+=[s]
            
                prov=np.loadtxt(f)
                if t!=0:
                    provrest=np.loadtxt((glob.glob(data[0].replace('SUBJECT',s))[0]))
					prov=prov-provrest
  
                UpmatricesADHD[t]+=[prov]
                familyADHD[t]+=[(allinfo_family[k]['family']) for k in allinfo_family.keys() if allinfo_family[k]['subject']==s]
  
# In[8]:

##load data for siblings

UpmatricesSibADHD=[[] for t in range(len(NeuroIMAGE))]

keepsujSibADHD=[[] for i in range(len(sujSibADHD))]
familySibADHD=[[] for i in range(len(sujSibADHD))]

for t in range(len(NeuroIMAGE)):
    
    for s in sujSibADHD[t]:
        dat=glob.glob(data[t].replace('SUBJECT',s))
        for f in dat:
            do=1
            if s in sujSibADHD[0]:
                prov=np.loadtxt(f)
                provrest=np.loadtxt(glob.glob(data[0].replace('SUBJECT',s))[0])
            else:
                do=0
            if do==1:
                keepsujSibADHD[t]+=[s]
				prov=np.loadtxt(f)
                if t!=0:
                    provrest=np.loadtxt((glob.glob(data[0].replace('SUBJECT',s))[0]))
					prov=prov-provrest
                UpmatricesSibADHD[t]+=[prov]

                familySibADHD[t]+=[(allinfo_family[k]['family']) for k in allinfo_family.keys() if allinfo_family[k]['subject']==s]



##load data for control


UpmatricesCONTROL=[[] for t in range(len(NeuroIMAGE))]

keepsujCONTROL=[[] for i in range(len(sujCONTROL))]
familyCONTROL=[[] for i in range(len(sujCONTROL))]

for t in range(len(NeuroIMAGE)):
    prov2=[]
    for s in sujCONTROL[t]:
        dat=glob.glob(data[t].replace('SUBJECT',s))
        for f in dat:
            do=1
            if s in sujCONTROL[0]:
                prov=np.loadtxt(f)
                provrest=np.loadtxt(glob.glob(data[0].replace('SUBJECT',s))[0])
            else:
                do=0
            if do==1:
                keepsujCONTROL[t]+=[s]
				prov=np.loadtxt(f)
                if t!=0:
                    provrest=np.loadtxt((glob.glob(data[0].replace('SUBJECT',s))[0]))
					prov=prov-provrest
                UpmatricesCONTROL[t]+=[prov]

                familyCONTROL[t]+=[(allinfo_family[k]['family']) for k in allinfo_family.keys() if allinfo_family[k]['subject']==s]
 
# defined family members and choose one participant per family
if family==0:
    #ADHD and siblings
    linkfamilyADHD=[np.zeros(len(keepsujADHD[i])) for i in range(len(keepsujADHD))]
    linkfamilySibADHD=[np.zeros(len(keepsujSibADHD[i])) for i in range(len(keepsujSibADHD))]
    linkfamilychosenADHD=[np.zeros(len(keepsujSibADHD[i])) for i in range(len(keepsujSibADHD))]
    linkfamilychosenSib=[np.zeros(len(keepsujADHD[i])) for i in range(len(keepsujADHD))]
    
    for t in range(len(keepsujADHD)):
        Nadhd=len(np.unique(familyADHD[t])) #ADHD family
        NadhdS=len(np.unique(familySibADHD[t])) #with siblings
        NadhdSA=len(np.unique([j for j in np.concatenate([familySibADHD[t],familyADHD[t]],axis=0) if j in familySibADHD[t] and j in familyADHD[t]])) #with siblings
        print NadhdSA,NadhdS,Nadhd
        a=-1 #alternate keep sib or ADHD
        for i in range(len(keepsujADHD[t])):
            if linkfamilyADHD[t][i]==0:
                linkfamilyADHD[t][np.where(np.array(familyADHD[t])==familyADHD[t][i])]=-1
                so=np.where(np.array(familySibADHD[t])==familyADHD[t][i])
                if len(so[0])!=0:
                    linkfamilySibADHD[t][so]=-1
                    
                    if a==0 or NadhdS-Nadhd>0:#keep ADHD reject siblings
                        linkfamilyADHD[t][i]=1
                        linkfamilychosenSib[t][i]=so[0][0]
                        print so[0][0]
                        if a==-1:
                            Nadhd+=1
                            NadhdSA+=-1
                        else:
                            a=1
                    elif a==1 or Nadhd-NadhdS>0: #keep one sibling, reject other and ADHD
                        linkfamilyADHD[t][i]=-1
                        linkfamilySibADHD[t][so[0][0]]=1
                        linkfamilychosenADHD[t][so[0][0]]=i
                        
                        if a==-1:
                            NadhdS+=1
                            NadhdSA+=-1
                        else:
                            a=0
                            
                    
                else:
                    linkfamilyADHD[t][i]=1
                    if a==-1:
                        Nadhd+=1
                        NadhdSA+=-1
                    else:
                        a=1
                if NadhdS-Nadhd==0 and a==-1:
                    print 'now'
                    a=0
                    NadhdSA=-1
        for i in range(len(keepsujSibADHD[t])):
            if linkfamilySibADHD[t][i]==0:
                linkfamilySibADHD[t][np.where(np.array(familySibADHD[t])==familySibADHD[t][i])]=-1
                linkfamilySibADHD[t][i]=1
        print NadhdSA,NadhdS,Nadhd     
    linkfamilyCONTROL=[np.zeros(len(keepsujCONTROL[i])) for i in range(len(keepsujCONTROL))]
    for t in range(len(keepsujCONTROL)):
        #keep only one
        
        for i in range(len(keepsujCONTROL[t])):
            if linkfamilyCONTROL[t][i]==0:
                linkfamilyCONTROL[t][np.where(np.array(familyCONTROL[t])==familyCONTROL[t][i])]=-1
                linkfamilyCONTROL[t][i]=1




## remove matrice for related family members, i.e. that are -1 in linkfamily

if family==0:
    
    for t in range(len(UpmatricesCONTROL)):
        UpmatricesCONTROL[t]=[UpmatricesCONTROL[t][i] for i in range(len(UpmatricesCONTROL[t])) if linkfamilyCONTROL[t][i]==1]
        
    for t in range(len(UpmatricesADHD)):
        UpmatricesADHD[t]=[UpmatricesADHD[t][i] for i in range(len(UpmatricesADHD[t])) if linkfamilyADHD[t][i]==1]
       
    for t in range(len(UpmatricesSibADHD)):
        UpmatricesSibADHD[t]=[UpmatricesSibADHD[t][i] for i in range(len(UpmatricesSibADHD[t])) if linkfamilySibADHD[t][i]==1]
 
if not os.path.exists('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/dist_pdf'):
    os.makedirs('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/dist_pdf')






# # edge selection

def selectionMM(mat,step=50,iteration=3,method='weighted_pFDR'):
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
        #down
        
        init2=-(stmu1-2*stv1)
        last2=np.max(-data)
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
        

        selectionfromtriuIndex=[np.where((data>init)+(data<-init2)==1)]
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
    
    return selectionfromtriuIndex,tmin,tmax







bootstrap=500 #this script is runned 20 times, then combined in the ipython notebook

#permutation testing

# 7 =shared/unique : shared between ADHD SIblings Controls, shared between ADHD and control, unique to ADHD, shares between ADHD and Siblings, unique to siblings, shared between Siblnigs and controls, unique to controls
# 3 tasks = stop, reward, working memory
# 3 mix tasks = stop/reward, stop/wm, reward/wm
pairedtask=[[0,1],[0,2],[1,2]]


tri=np.zeros((number,number))
tri[np.triu_indices(number,1)]=1
p=np.triu_indices(number,1)
#percentage relative to category (sens, psingle,pmix,pall)
modulatebysens=np.zeros((3,7,bootstrap))#3 task, shared/unique, bootstrap
modulatebyspe=np.zeros((3,7,bootstrap))#Psingle: 3 task, shared/unique, bootstrap
modulatebyun=np.zeros((3,7,bootstrap))#Pmix, 3mix task, shared/unique, bootstrap
modulatebycom=np.zeros((7,bootstrap))# shared/unique, bootstrap
#percentage relative to sensitivity
modulatebysensG=np.zeros((3,7,bootstrap))#=[[[] for i in range(7)] for j in range(3)]#ASC,AC,A,AS,S,SC,C, for three tasks
modulatebyspeG=np.zeros((3,3,bootstrap))#[[[] for i in range(7)] for j in range(3)]#ASC,AC,A,AS,S,SC,C, for three tasks
modulatebyunG=np.zeros((3,3,bootstrap))#[[[] for i in range(7)] for j in range(3)]#ASC,AC,A,AS,S,SC,C, for three tasks, for 2 possible pairs of tasks 
modulatebycomG=np.zeros((3,3,bootstrap))#[[] for i in range(7)]#ASC,AC,A,AS,S,SC,C groups
#selectivity for each connections
SUMsensitivity=np.zeros((3,7,bootstrap,number,number))#[[np.zeros((number,number)) for i in range(7)] for j in range(3)]#ASC,AC,A,AS,S,SC,C, for three tasks
SUMspecificity=np.zeros((3,7,bootstrap,number,number))#[[np.zeros((number,number)) for i in range(7)] for j in range(3)]#ASC,AC,A,AS,S,SC,C, for three tasks
SUMundefined=np.zeros((3,7,bootstrap,number,number))#[[np.zeros((number,number)) for i in range(7)] for j in range(3)]
SUMcommon=np.zeros((7,bootstrap,number,number))#[np.zeros((number,number)) for i in range(7)]#ASC,AC,A,AS,S,SC,C groups
#
modulatebysens2=np.zeros((3,7,bootstrap))
modulatebyspe2=np.zeros((3,7,bootstrap))
modulatebyun2=np.zeros((3,7,bootstrap))
modulatebycom2=np.zeros((7,bootstrap))

modulatebysensG2=np.zeros((3,7,bootstrap))
modulatebyspeG2=np.zeros((3,3,bootstrap))
modulatebyunG2=np.zeros((3,3,bootstrap))
modulatebycomG2=np.zeros((3,3,bootstrap))
SUMsensitivity2=np.zeros((3,7,bootstrap,number,number))

SUMspecificity2=np.zeros((3,7,bootstrap,number,number))
SUMundefined2=np.zeros((3,7,bootstrap,number,number))
SUMcommon2=np.zeros((7,bootstrap,number,number))

#threshold for each task and group , bootstrap, negative and positive
threshold_group=np.zeros((3,3,bootstrap,2))
threshold_group2=np.zeros((3,3,bootstrap,2))


#1 split control in 3
orderADHD=np.array([np.array(range(len(UpmatricesADHD[i+1]))) for i in range(3)])
orderSibADHD=np.array([np.array(range(len(UpmatricesSibADHD[i+1]))) for i in range(3)])
orderCONTROL=np.array([np.array(range(len(UpmatricesCONTROL[i+1]))) for i in range(3)])

#selected edges and mean values: 3 tasks, 1000bootstrap, upper triangle
GENEADHDsel=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENEADHDval=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENESIBsel=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENESIBval=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENECONsel=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENECONval=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))

#SAME FOR NULL DISTRIBUTION
GENEADHDsel2=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENEADHDval2=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENESIBsel2=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENESIBval2=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENECONsel2=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))
GENECONval2=np.zeros((3,bootstrap,len(np.triu_indices(number,1)[0])))



tot=0
b=0
groupS=[[0,1,2,3],[0,3,4,5],[0,1,5,6]]

while b<bootstrap:
    
    #2 comput mean potency for each group, 80% of sample with replacement
    
    for i in range(3):
        random.shuffle(orderADHD[i])
        random.shuffle(orderSibADHD[i])
        random.shuffle(orderCONTROL[i])
    listsuj=[[[orderADHD[i][[int(len(orderADHD[i])*0.8*random.random()) for m in range(int(len(orderADHD[i])*0.8))]]],[orderSibADHD[i][[int(len(orderSibADHD[i])*0.8*random.random()) for m in range(int(len(orderSibADHD[i])*0.8))]]],[orderCONTROL[i][[int(len(orderCONTROL[i])*0.8*random.random()) for m in range(int(len(orderCONTROL[i])*0.8))]]]] for i in range(3)]
    #mean potency per group for this bootstrap
    Upmatricessplit=copy.deepcopy([[np.mean(np.array(UpmatricesADHD[i+1])[listsuj[i][0]]*np.sqrt(int(len(orderADHD[i])*0.8)),0),np.mean(np.array(UpmatricesSibADHD[i+1])[listsuj[i][1]]*np.sqrt(int(len(orderSibADHD[i])*0.8)),0),np.mean(np.array(UpmatricesCONTROL[i+1])[listsuj[i][2]]*np.sqrt(int(len(orderCONTROL[i])*0.8)),0)] for i in range(3)])

    #create null distribution by randomising the previously made selection but keeping group size constant
    randomlistS=[np.concatenate([np.zeros(len(listsuj[i][0][0])),np.ones(len(listsuj[i][1][0])),2*np.ones(len(listsuj[i][2][0]))]) for i in range(3)]
    for i in range(3):
        random.shuffle(randomlistS[i])
    NrandomlistS=[[[len(np.where(randomlistS[i][:len(listsuj[i][0][0])]==a)[0]) for a in range(3)],
[len(np.where(randomlistS[i][len(listsuj[i][0][0]):len(listsuj[i][0][0])+len(listsuj[i][1][0])]==a)[0]) for a in range(3)],
[len(np.where(randomlistS[i][len(listsuj[i][0][0])+len(listsuj[i][1][0]):]==a)[0]) for a in range(3)]] for i in range(3)]#number in relabelling #group #tasks 
	#mean potency null representation for this bootstrap
    Upmatricessplit2=copy.deepcopy([[np.mean(np.concatenate([np.array(UpmatricesADHD[i+1])[listsuj[i][0][0][:NrandomlistS[i][0][0]]],np.array(UpmatricesSibADHD[i+1])[listsuj[i][1][0][:NrandomlistS[i][0][1]]],np.array(UpmatricesCONTROL[i+1])[listsuj[i][2][0][:NrandomlistS[i][0][2]]]],0)*np.sqrt(int(len(listsuj[i][0][0]))),0),np.mean(np.concatenate([np.array(UpmatricesADHD[i+1])[listsuj[i][0][0][NrandomlistS[i][0][0]:NrandomlistS[i][0][0]+NrandomlistS[i][1][0]]],np.array(UpmatricesSibADHD[i+1])[listsuj[i][1][0][NrandomlistS[i][0][1]:NrandomlistS[i][0][1]+NrandomlistS[i][1][1]]],np.array(UpmatricesCONTROL[i+1])[listsuj[i][2][0][NrandomlistS[i][0][2]:NrandomlistS[i][0][2]+NrandomlistS[i][1][2]]]],0)*np.sqrt(int(len(listsuj[i][1][0]))),0),np.mean(np.concatenate([np.array(UpmatricesADHD[i+1])[listsuj[i][0][0][NrandomlistS[i][0][0]+NrandomlistS[i][1][0]:]],np.array(UpmatricesSibADHD[i+1])[listsuj[i][1][0][NrandomlistS[i][0][1]+NrandomlistS[i][1][1]:]],np.array(UpmatricesCONTROL[i+1])[listsuj[i][2][0][NrandomlistS[i][0][2]+NrandomlistS[i][1][2]:]]],0)*np.sqrt(int(len(listsuj[i][2][0]))),0)] for i in range(3)])
	
	

    try:
        for i in range(3): #for each task
            #define group selection threshold for each group and select the least conservative one
            for s in range(3):
            
                prov=selectionMM(Upmatricessplit[i][s])
                threshold_group[i][s][b]=[prov[1],prov[2]]

            provthr=[np.min([threshold_group[i][s][b][0] for s in range(3)]),np.max([threshold_group[i][s][b][1] for s in range(3)])]
			#SAME FOR NULL DISTRIBUTION
            for s in range(3):
            
                prov=selectionMM(Upmatricessplit2[i][s])
                threshold_group2[i][s][b]=[prov[1],prov[2]]

            provthr2=[np.min([threshold_group2[i][s][b][0] for s in range(3)]),np.max([threshold_group2[i][s][b][1] for s in range(3)])]
                
			#apply threshold for each of the group mean potency
            for s in range(3):
            

                prov3=np.zeros(len(p[0]))
                prov3[np.where(Upmatricessplit[i][s][np.triu_indices(number,1)]<=provthr[0])]+=1
                prov3[np.where(Upmatricessplit[i][s][np.triu_indices(number,1)]>=provthr[1])]+=1
                prov3=np.sign(prov3)

                prov2=np.zeros((number,number))
                prov2[np.triu_indices(number,1)]=prov3
				#same selection and mean potency for each group
                if s==0:
                    GENEADHDsel[i][b]=copy.deepcopy(prov3)
                    GENEADHDval[i][b]=copy.deepcopy(Upmatricessplit[i][s][np.triu_indices(number,1)]  )
                elif s==1:
                    GENESIBsel[i][b]=copy.deepcopy(prov3)
                    GENESIBval[i][b]=copy.deepcopy(Upmatricessplit[i][s][np.triu_indices(number,1)]  )
                else:
                    GENECONsel[i][b]=copy.deepcopy(prov3)
                    GENECONval[i][b]=copy.deepcopy(Upmatricessplit[i][s][np.triu_indices(number,1)]  )
                Upmatricessplit[i][s]=copy.deepcopy(prov2)
                #SAME FOR NULL DISTRIBUTION
                prov3=np.zeros(len(p[0]))
                prov3[np.where(Upmatricessplit2[i][s][np.triu_indices(number,1)]<=provthr2[0])]+=1
                prov3[np.where(Upmatricessplit2[i][s][np.triu_indices(number,1)]>=provthr2[1])]+=1
                prov3=np.sign(prov3)

                prov2=np.zeros((number,number))
                prov2[np.triu_indices(number,1)]=prov3
                if s==0:
                    GENEADHDsel2[i][b]=copy.deepcopy(prov3)
                    GENEADHDval2[i][b]=copy.deepcopy(Upmatricessplit2[i][s][np.triu_indices(number,1)]  )
                elif s==1:
                    GENESIBsel2[i][b]=copy.deepcopy(prov3)
                    GENESIBval2[i][b]=copy.deepcopy(Upmatricessplit2[i][s][np.triu_indices(number,1)]  )
                else:
                    GENECONsel2[i][b]=copy.deepcopy(prov3)
                    GENECONval2[i][b]=copy.deepcopy(Upmatricessplit2[i][s][np.triu_indices(number,1)]  )
                Upmatricessplit2[i][s]=copy.deepcopy(prov2)
		#overall sum of selected edges across group (task sensitivity percentage)
        totalSENS=[np.float(len(np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]!=0)[0])) for i in range(3)]
        totalSENS=[np.max([q,1.]) for q in totalSENS]
		#SAME FOR NULL DISTRIBUTION
		totalSENS2=[np.float(len(np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]!=0)[0])) for i in range(3)]
        totalSENS2=[np.max([q,1.]) for q in totalSENS2]
		#sum of selected edges per group
        totalSENSG=[[np.float(len(np.where((Upmatricessplit[i][g])[np.triu_indices(number,1)]!=0)[0])) for g in range(3)] for i in range(3)]
        totalSENSG=[[np.max([q,1.]) for q in q2] for q2 in totalSENSG]
		#SAME FOR NULL DISTRIBUTION
		totalSENSG2=[[np.float(len(np.where((Upmatricessplit2[i][g])[np.triu_indices(number,1)]!=0)[0])) for g in range(3)] for i in range(3)]
        totalSENSG2=[[np.max([q,1.]) for q in q2] for q2 in totalSENSG2]
        
        ###specificity
		#instanciation
        UpmatricessplitSPE=np.sum(np.array(Upmatricessplit),0) #Psingle
        UpmatricessplitCOM=copy.deepcopy(UpmatricessplitSPE) #Pall
        UpmatricessplitUN=copy.deepcopy(UpmatricessplitSPE) #Pmix
        #Psingle
		UpmatricessplitSPE[np.where(UpmatricessplitSPE!=1)]=0
        UpmatricessplitSPE=[UpmatricessplitSPE*Upmatricessplit[i] for i in range(3)]
		#sum of Psingle edges for percentage calculation
        totalSPE=[np.float(len(np.where(np.sum(UpmatricessplitSPE[i],0)[np.triu_indices(number,1)]!=0)[0])) for i in range(3)]
        totalSPE=[np.max([q,1.]) for q in totalSPE]
        #SAME FOR NULL DISTRIBUTION
        UpmatricessplitSPE2=np.sum(np.array(Upmatricessplit2),0)
        UpmatricessplitCOM2=copy.deepcopy(UpmatricessplitSPE2)
        UpmatricessplitUN2=copy.deepcopy(UpmatricessplitSPE2)
        UpmatricessplitSPE2[np.where(UpmatricessplitSPE2!=1)]=0
        UpmatricessplitSPE2=[UpmatricessplitSPE2*Upmatricessplit2[i] for i in range(3)]
        totalSPE2=[np.float(len(np.where(np.sum(UpmatricessplitSPE2[i],0)[np.triu_indices(number,1)]!=0)[0])) for i in range(3)]
        totalSPE2=[np.max([q,1.]) for q in totalSPE2]
        
        
        # Pmix
        UpmatricessplitUN[np.where(UpmatricessplitUN!=2)]=0
        UpmatricessplitUNG=np.sign([UpmatricessplitUN*Upmatricessplit[i] for i in range(3)])
        UpmatricessplitUN=np.sign([UpmatricessplitUN*Upmatricessplit[a[0]]*Upmatricessplit[a[1]] for a in pairedtask])
        totalUN=[np.float(len(np.where(np.sum(UpmatricessplitUN[i],0)[np.triu_indices(number,1)]!=0)[0])) for i in range(3)]
        totalUN=[np.max([q,1.]) for q in totalUN]
        #SAME FOR NULL DISTRIBUTION
        UpmatricessplitUN2[np.where(UpmatricessplitUN2!=2)]=0
        UpmatricessplitUN2G=np.sign([UpmatricessplitUN2*Upmatricessplit2[i] for i in range(3)])
        UpmatricessplitUN2=np.sign([UpmatricessplitUN2*Upmatricessplit2[a[0]]*Upmatricessplit2[a[1]] for a in pairedtask])
        totalUN2=[np.float(len(np.where(np.sum(UpmatricessplitUN2[i],0)[np.triu_indices(number,1)]!=0)[0])) for i in range(3)]
        totalUN2=[np.max([q,1.]) for q in totalUN2]
        #common
        UpmatricessplitCOM[np.where(UpmatricessplitCOM!=3)]=0
        UpmatricessplitCOM=np.sign(UpmatricessplitCOM)
        totalCOM=np.float(len(np.where(np.sum(UpmatricessplitCOM,0)[np.triu_indices(number,1)]!=0)[0]))
        totalCOM=np.max([totalCOM,1.])
        #SAME FOR NULL DISTRIBUTION
        UpmatricessplitCOM2[np.where(UpmatricessplitCOM2!=3)]=0
        UpmatricessplitCOM2=np.sign(UpmatricessplitCOM2)
        totalCOM2=np.float(len(np.where(np.sum(UpmatricessplitCOM2,0)[np.triu_indices(number,1)]!=0)[0]))
        totalCOM2=np.max([totalCOM2,1.])
        
        # for no percentages : 1.
        #3 compute groups overlap
        #for one group
        Ns=np.array([2,4,6])
        for s in range(3):
            for i in range(3):
                #sensitivity
                SUMsensitivity[i][Ns[s]][b][np.where(Upmatricessplit[i][s]*np.sum(Upmatricessplit[i],0)==1)]=1
                modulatebysens[i][Ns[s]][b]=np.sum(Upmatricessplit[i][s][np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]==1)])/totalSENS[i]
                modulatebysensG[i][Ns[s]][b]=np.sum(Upmatricessplit[i][s][np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]==1)])
                #SAME FOR NULL DISTRIBUTION
                SUMsensitivity2[i][Ns[s]][b][np.where(Upmatricessplit2[i][s]*np.sum(Upmatricessplit2[i],0)==1)]=1
                modulatebysens2[i][Ns[s]][b]=np.sum(Upmatricessplit2[i][s][np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]==1)])/totalSENS2[i]
                modulatebysensG2[i][Ns[s]][b]=np.sum(Upmatricessplit2[i][s][np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]==1)])
            
                #Psingle
                SUMspecificity[i][Ns[s]][b][np.where(UpmatricessplitSPE[i][s]*np.sum(UpmatricessplitSPE[i],0)==1)]=1
                modulatebyspe[i][Ns[s]][b]=np.sum(UpmatricessplitSPE[i][s][np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitSPE[i],0)[np.triu_indices(number,1)]==1)])/totalSPE[i]
                modulatebyspeG[i][s][b]=np.sum(np.sign(UpmatricessplitSPE[i][s][np.triu_indices(number,1)]))/totalSENSG[i][s]
                #SAME FOR NULL DISTRIBUTION
                SUMspecificity2[i][Ns[s]][b][np.where(UpmatricessplitSPE2[i][s]*np.sum(UpmatricessplitSPE2[i],0)==1)]=1
                modulatebyspe2[i][Ns[s]][b]=np.sum(UpmatricessplitSPE2[i][s][np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitSPE2[i],0)[np.triu_indices(number,1)]==1)])/totalSPE2[i]
                modulatebyspeG2[i][s][b]=np.sum(np.sign(UpmatricessplitSPE2[i][s][np.triu_indices(number,1)]))/totalSENSG2[i][s]
                #Pmix
                SUMundefined[i][Ns[s]][b][np.where(UpmatricessplitUN[i][s]*np.sum(UpmatricessplitUN[i],0)==1)]=1
                modulatebyun[i][Ns[s]][b]=np.sum(UpmatricessplitUN[i][s][np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitUN[i],0)[np.triu_indices(number,1)]==1)])/totalUN[i]
                modulatebyunG[i][s][b]=np.sum(np.sign(UpmatricessplitUNG[i][s][np.triu_indices(number,1)]))/totalSENSG[i][s]
                #SAME FOR NULL DISTRIBUTION
                SUMundefined2[i][Ns[s]][b][np.where(UpmatricessplitUN2[i][s]*np.sum(UpmatricessplitUN2[i],0)==1)]=1
                modulatebyun2[i][Ns[s]][b]=np.sum(UpmatricessplitUN2[i][s][np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitUN2[i],0)[np.triu_indices(number,1)]==1)])/totalUN2[i]
                modulatebyunG2[i][s][b]=np.sum(np.sign(UpmatricessplitUN2G[i][s][np.triu_indices(number,1)]))/totalSENSG2[i][s]
				#Pall
                modulatebycomG[i][s][b]=np.sum(np.sign(UpmatricessplitCOM[s][np.triu_indices(number,1)]))/totalSENSG[i][s]
                modulatebycomG2[i][s][b]=np.sum(np.sign(UpmatricessplitCOM2[s][np.triu_indices(number,1)]))/totalSENSG2[i][s]
            #sum for percentage calculation
			SUMcommon[Ns[s]][b][np.where(UpmatricessplitCOM[s]*np.sum(UpmatricessplitCOM,0)==1)]=1
            modulatebycom[Ns[s]][b]=np.sum(UpmatricessplitCOM[s][np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitCOM,0)[np.triu_indices(number,1)]==1)])/totalCOM
    
            #SAME FOR NULL DISTRIBUTION
            SUMcommon2[Ns[s]][b][np.where(UpmatricessplitCOM2[s]*np.sum(UpmatricessplitCOM2,0)==1)]=1
            modulatebycom2[Ns[s]][b]=np.sum(UpmatricessplitCOM2[s][np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitCOM2,0)[np.triu_indices(number,1)]==1)])/totalCOM2
        #two groups
        Ns=np.array([3,1,5])
        for e,s in enumerate(pairedtask):
            for i in range(3):
                #sensitivity
                SUMsensitivity[i][Ns[e]][b][np.where((Upmatricessplit[i][s[0]]+Upmatricessplit[i][s[1]])*np.sum(Upmatricessplit[i],0)==4)]=1
                modulatebysens[i][Ns[e]][b]=len(np.where((Upmatricessplit[i][s[0]]+Upmatricessplit[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]==2)]==2)[0])/totalSENS[i]
                modulatebysensG[i][Ns[e]][b]=len(np.where((Upmatricessplit[i][s[0]]+Upmatricessplit[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]==2)]==2)[0])
                #SAME FOR NULL DISTRIBUTION
                SUMsensitivity2[i][Ns[e]][b][np.where((Upmatricessplit2[i][s[0]]+Upmatricessplit2[i][s[1]])*np.sum(Upmatricessplit2[i],0)==4)]=1
                modulatebysens2[i][Ns[e]][b]=len(np.where((Upmatricessplit2[i][s[0]]+Upmatricessplit2[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]==2)]==2)[0])/totalSENS2[i]
                modulatebysensG2[i][Ns[e]][b]=len(np.where((Upmatricessplit2[i][s[0]]+Upmatricessplit2[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]==2)]==2)[0])
            
                #Psingle
                SUMspecificity[i][Ns[e]][b][np.where((UpmatricessplitSPE[i][s[0]]+UpmatricessplitSPE[i][s[1]])*np.sum(UpmatricessplitSPE[i],0)==4)]=1
                modulatebyspe[i][Ns[e]][b]=len(np.where((UpmatricessplitSPE[i][s[0]]+UpmatricessplitSPE[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitSPE[i],0)[np.triu_indices(number,1)]==2)]==2)[0])/totalSPE[i]
                #SAME FOR NULL DISTRIBUTION
                SUMspecificity2[i][Ns[e]][b][np.where((UpmatricessplitSPE2[i][s[0]]+UpmatricessplitSPE2[i][s[1]])*np.sum(UpmatricessplitSPE2[i],0)==4)]=1
                modulatebyspe2[i][Ns[e]][b]=len(np.where((UpmatricessplitSPE2[i][s[0]]+UpmatricessplitSPE2[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitSPE2[i],0)[np.triu_indices(number,1)]==2)]==2)[0])/totalSPE2[i]
                #Pmix
                SUMundefined[i][Ns[e]][b][np.where((UpmatricessplitUN[i][s[0]]+UpmatricessplitUN[i][s[1]])*np.sum(UpmatricessplitUN[i],0)==4)]=1
                modulatebyun[i][Ns[e]][b]=len(np.where((UpmatricessplitUN[i][s[0]]+UpmatricessplitUN[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitUN[i],0)[np.triu_indices(number,1)]==2)]==2)[0])/totalUN[i]
                #SAME FOR NULL DISTRIBUTION
                SUMundefined2[i][Ns[e]][b][np.where((UpmatricessplitUN2[i][s[0]]+UpmatricessplitUN2[i][s[1]])*np.sum(UpmatricessplitUN2[i],0)==4)]=1
                modulatebyun2[i][Ns[e]][b]=len(np.where((UpmatricessplitUN2[i][s[0]]+UpmatricessplitUN2[i][s[1]])[np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitUN2[i],0)[np.triu_indices(number,1)]==2)]==2)[0])/totalUN2[i]

			#overall sum for percentage
            SUMcommon[Ns[e]][b][np.where((UpmatricessplitCOM[s[0]]+UpmatricessplitCOM[s[1]])*np.sum(UpmatricessplitCOM,0)==4)]=1
            modulatebycom[Ns[e]][b]=len(np.where((UpmatricessplitCOM[s[0]]+UpmatricessplitCOM[s[1]])[np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitCOM,0)[np.triu_indices(number,1)]==2)]==2)[0])/totalCOM
            #SAME FOR NULL DISTRIBUTION
            SUMcommon2[Ns[e]][b][np.where((UpmatricessplitCOM2[s[0]]+UpmatricessplitCOM2[s[1]])*np.sum(UpmatricessplitCOM2,0)==4)]=1
            modulatebycom2[Ns[e]][b]=len(np.where((UpmatricessplitCOM2[s[0]]+UpmatricessplitCOM2[s[1]])[np.triu_indices(number,1)][np.where(np.sum(UpmatricessplitCOM2,0)[np.triu_indices(number,1)]==2)]==2)[0])/totalCOM2
        #three groups
        for i in range(3):
            #sensitivity
            SUMsensitivity[i][0][b][np.where(np.sum(Upmatricessplit[i],0)==3)]=1
            modulatebysens[i][0][b]=len(np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]==3)[0])/totalSENS[i]
            modulatebysensG[i][0][b]=len(np.where(np.sum(Upmatricessplit[i],0)[np.triu_indices(number,1)]==3)[0])
            #Psingle
            SUMspecificity[i][0][b][np.where(np.sum(UpmatricessplitSPE[i],0)==3)]=1
            modulatebyspe[i][0][b]=len(np.where(np.sum(UpmatricessplitSPE[i],0)[np.triu_indices(number,1)]==3)[0])/totalSPE[i]
            #Pmix
            SUMundefined[i][0][b][np.where(np.sum(UpmatricessplitUN[i],0)==3)]=1
            modulatebyun[i][0][b]=len(np.where(np.sum(UpmatricessplitUN[i],0)[np.triu_indices(number,1)]==3)[0])/totalUN[i]
            #SAME FOR NULL DISTRIBUTION
            SUMsensitivity2[i][0][b][np.where(np.sum(Upmatricessplit2[i],0)==3)]=1
            modulatebysens2[i][0][b]=len(np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]==3)[0])/totalSENS2[i]
            modulatebysensG2[i][0][b]=len(np.where(np.sum(Upmatricessplit2[i],0)[np.triu_indices(number,1)]==3)[0])
            #
            SUMspecificity2[i][0][b][np.where(np.sum(UpmatricessplitSPE2[i],0)==3)]=1
            modulatebyspe2[i][0][b]=len(np.where(np.sum(UpmatricessplitSPE2[i],0)[np.triu_indices(number,1)]==3)[0])/totalSPE2[i]
            #
            SUMundefined2[i][0][b][np.where(np.sum(UpmatricessplitUN2[i],0)==3)]=1
            modulatebyun2[i][0][b]=len(np.where(np.sum(UpmatricessplitUN2[i],0)[np.triu_indices(number,1)]==3)[0])/totalUN2[i]
		#Pall
        SUMcommon[0][b][np.where(np.sum(UpmatricessplitCOM,0)==3)]=1
        modulatebycom[0][b]=len(np.where(np.sum(UpmatricessplitCOM,0)[np.triu_indices(number,1)]==3)[0])/totalCOM
		#SAME FOR NULL DISTRIBUTION
        SUMcommon2[0][b][np.where(np.sum(UpmatricessplitCOM2,0)==3)]=1
        modulatebycom2[0][b]=len(np.where(np.sum(UpmatricessplitCOM2,0)[np.triu_indices(number,1)]==3)[0])/totalCOM2
        tot+=1
        b+=1
        
    except:
        print b,'fail'
        #b=b-1
        tot+=1
    if b/10.-b/10==0.:
        print b,tot

np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSENS80percent_1030_'+instance,modulatebysens)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSPE80percent_1030_'+instance,modulatebyspe)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapCOM80percent_1030_'+instance,modulatebycom)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapUN80percent_1030_'+instance,modulatebyun)

np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSENS80percent_group_1030_'+instance,modulatebysensG)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSPE80percent_group_1030_'+instance,modulatebyspeG)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapCOM80percent_group_1030_'+instance,modulatebycomG)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapUN80percent_group_1030_'+instance,modulatebyunG)

np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapUN80percent_alledges_1030_'+instance,np.mean(SUMundefined,2))
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSENS80percent_alledges_1030_'+instance,np.mean(SUMsensitivity,2))
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSPE80percent_alledges_1030_'+instance,np.mean(SUMspecificity,2))
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapCOM80percent_alledges_1030_'+instance,np.mean(SUMcommon,1))


np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSENS80percent_randomlabel_1030_'+instance,modulatebysens2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSPE80percent_randomlabel_1030_'+instance,modulatebyspe2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapCOM80percent_randomlabel_1030_'+instance,modulatebycom2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapUN80percent_randomlabel_1030_'+instance,modulatebyun2)

np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSENS80percent_randomlabel_group_1030_'+instance,modulatebysensG2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSPE80percent_randomlabel_group_1030_'+instance,modulatebyspeG2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapCOM80percent_randomlabel_group_1030_'+instance,modulatebycomG2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapUN80percent_randomlabel_group_1030_'+instance,modulatebyunG2)

np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapUN80percent_alledges_randomlabel_1030_'+instance,np.mean(SUMundefined2,2))
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSENS80percent_alledges_randomlabel_1030_'+instance,np.mean(SUMsensitivity2,2))
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapSPE80percent_alledges_randomlabel_1030_'+instance,np.mean(SUMspecificity2,2))
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapCOM80percent_alledges_randomlabel_1030_'+instance,np.mean(SUMcommon2,1))




np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/selec80potency_ADHD_randomlabel_1030_'+instance,GENEADHDsel2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/mean80potency_ADHD_randomlabel_1030_'+instance,GENEADHDval2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/selec80potency_SIBLING_randomlabel_1030_'+instance,GENESIBsel2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/mean80potency_SIBLING_randomlabel_1030_'+instance,GENESIBval2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/selec80potency_CONTROL_randomlabel_1030_'+instance,GENECONsel2)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/mean80potency_CONTROL_randomlabel_1030_'+instance,GENECONval2)


np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/selec80potency_ADHD_1030_'+instance,GENEADHDsel)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/mean80potency_ADHD_1030_'+instance,GENEADHDval)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/selec80potency_SIBLING_1030_'+instance,GENESIBsel)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/mean80potency_SIBLING_1030_'+instance,GENESIBval)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/selec80potency_CONTROL_1030_'+instance,GENECONsel)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/mean80potency_CONTROL_1030_'+instance,GENECONval)


np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapthreshold_group_80percent_1030_'+instance,threshold_group)
np.save('/home/mrstats/roscha/NeuroIMAGE/TasksVsRS_MatriceResult/reordering/'+fold+'/'+np.str(number)+'/permutationtestingoverlapthreshold_group_randomlabel_80percent_1030_'+instance,threshold_group2)


