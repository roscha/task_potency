#!/home/mrstats/maamen/Software/python/bin/python2.7


#run as : 

#run /matrice_subjects_GENERIC.py -data /fMRI_TASK/SUBJECT/RUN.feat/nuissance/denoised_func_data_nonaggr/times_series/denoised_func_data_nonaggr_hp_atlas_subcort_cerebellum_1mm_4D_warped_th_reg.txt -atlas /atlas_subcort_cerebellum_1mm_4D_warped_th.nii.gz -mask /fMRI_TASK/SUBJECT/RUN.feat/mask.nii.gz -NeuroIMAGE TASK -rejet 149 -method ledoit_wolf



import sys
sys.path.append('/')
import mm_helpers
import argparse, os, glob, random,sys
import numpy as np
import mm_GetInfoFromDb
import subprocess
from sklearn.decomposition import PCA
import sklearn.covariance
import nibabel
import rc_helpers
import copy

##-----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
                    description='''make and store covariance, correlation, partial correlation and z parial correlation matrices for a given time series file, design for beta serie hierarchie with subject, run, condition''') 

# required options 
reqoptions = parser.add_argument_group('Required Arguments')
reqoptions.add_argument('-data', action='store', dest = 'data', required = True, help = 'full path to time_serie.txt file, with SUBJECT/RUN(/CONDITION)')
reqoptions.add_argument('-mask', action='store', dest = 'mask', required = True, help = 'full path subject/run specific mask (be careful of the same registration than atlas is), with SUBJECT/RUN(/CONDITION)')
reqoptions.add_argument('-atlas', action='store', dest = 'atlas', required = True, help = 'full path atlas (be careful of the same registration than mask is),if needed with SUBJECT/RUN(/CONDITION)')

# optional options
optoptions = parser.add_argument_group('Optional Arguments')
optoptions.add_argument('-outDir', action='store', dest = 'outDir', required = False, default='', help = 'if not define, files will be stored in run directory/matrices because RUN will be concatenated. Full path to the folder name to store matrix python file with SUBJECT/CONDITION if necessary e.g SUBJECT/matrices/CONDITION, output file will be timeSerieName_matriceName_areaRejected')
optoptions.add_argument('-NeuroIMAGE', action='store', dest = 'NeuroIMAGE', required = False, default='', help = 'can be set as "NeuroIMAGE" or "STOP" or "REWARD" or "WM" or "RS" for NeuroIMAGE, enable to catch info from DB, else info will be catch in raw data (longer)')
optoptions.add_argument('-listSubject', action='store', dest = 'listSubject', required = False, default='', help = 'full path to a subject list than you want to process')
optoptions.add_argument('-rejet', action='store', dest = 'rejet', required = False, default='', help = 'area numeros to not consider (as indexed 0 the first area) separate by "/" e.g 32/149')
optoptions.add_argument('-onsetNames', action='store', dest = 'onsetNames', required = False, default='', help = 'names of onset of interest if CONDITION in data, so under betaserie condition. [if not define, take existing one] separate by "/" e.g Cue/Feedback')
optoptions.add_argument('-method', action='store', dest = 'method', required = False, default='basic', help = 'method to estimate covariance, can be "ledoit_wolf" or "basic" (default : shrinkage with 0.1) or a number for basic shrinkage e.g "0.3", or "" to do it in the old way, matrices name notice the method')
optoptions.add_argument('-maxTP', action='store', dest = 'maxTP', required = False, default='0', help = 'maximum number of time point consider to make the matrices : 200 is what we consider, if not define, will take all time point')
optoptions.add_argument('-Call', action='store', dest = 'Call', required = False, default='0', help = 'take all condition to build one connectivity matrix')
optoptions.add_argument('-db', action='store', dest = 'db', required = False, default='neuroimage', help = 'neuroimage or neuroimage2')
#_main_
## parse arguments
args = parser.parse_args()
Call=args.Call
method=args.method
data=args.data
mask=args.mask
atlas=args.atlas
onsetNames=args.onsetNames.split('/')
maxTP=np.int(args.maxTP)
rejet=args.rejet
db=args.db
if rejet!='':
    rejet=[int(i) for i in rejet.split('/')]


outDir=args.outDir
if outDir=='':
    outDir=os.path.dirname(data).replace('RUN','matrices')+'/CONDITION/'

NeuroIMAGE =args.NeuroIMAGE
listSubject=args.listSubject

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
        alpha=.5
        parCor=np.linalg.inv(covM+ alpha * np.eye(len(covM)))
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


if listSubject=='':
    if NeuroIMAGE!='':
        suj=np.unique([i[data.index('SUBJECT'):data.index('SUBJECT')+i.replace(data[:data.index('SUBJECT')],'').find('/')] for i in glob.glob(data.replace('SUBJECT','*').replace('RUN','*').replace('CONDITION','*'))])
        
        allinfo=dict()
        for s in suj:
            allinfo.update({s : {s:s,'run':[]}})

        alldata=glob.glob(data.replace('SUBJECT','*').replace('RUN','*').replace('CONDITION','*'))
        for f in alldata:
            r=f.replace(f[:f.find(mm_helpers.find_subject(f,'NeuroIMAGE'))]+mm_helpers.find_subject(f,'NeuroIMAGE')+'/','').replace(f[f.find('.feat'):],'').replace('/preprocessing','')
            if r not in allinfo[mm_helpers.find_subject(f,'NeuroIMAGE')]['run']:
                allinfo[mm_helpers.find_subject(f,'NeuroIMAGE')]['run']+=[r]


        allinfo=rc_helpers.dataSelection(allinfo,onsetNames,database=db)
        suj=sorted(allinfo.keys())

    else:
        suj=np.unique([i[data.index('SUBJECT'):data.index('SUBJECT')+i.replace(data[:data.index('SUBJECT')],'').find('/')] for i in glob.glob(data.replace('SUBJECT','*').replace('RUN','*').replace('CONDITION','*'))])
else :
    suj=open(listSubject).read().split()
    if NeuroIMAGE!='':
        
        allinfo=dict()
        for s in suj:
            allinfo.update({s : {s:s,'run':[]}})

            alldata=glob.glob(data.replace('SUBJECT',s).replace('RUN','*').replace('CONDITION','*'))
            for f in alldata:
                r=f.replace(f[:f.find(s)]+s+'/','').replace(f[f.find('.feat'):],'').replace('/preprocessing','')
                if r not in allinfo[s]['run']:
                    allinfo[s]['run']+=[r]


        allinfo=rc_helpers.dataSelection(allinfo,onsetNames,database=db)
        suj=sorted(allinfo.keys())



fail=[]
for s in suj :

    print s
    #defining condition if exist
    if onsetNames==['']:

        if NeuroIMAGE!='':
            r=allinfo[s]['run'][0]
        else:

            f=glob.glob(data.replace('SUBJECT',s).replace('RUN','*').replace('CONDITION','*'))[0]
            r=f[data.replace('SUBJECT',s).find('RUN'):data.replace('SUBJECT',s).find('RUN')+f.replace(f[:data.replace('SUBJECT',s).find('RUN')],'').find('/')].replace('.feat','')

        cond=glob.glob(data.replace('SUBJECT',s).replace('RUN',r).replace('CONDITION','*'))
        if len(cond)!=0:
            Icond=data.replace('SUBJECT',s).replace('RUN',r).find('CONDITION')
            cond=[i[Icond:len(i) - (len(data)-data.find('CONDITION')+9)] for i in cond]
        else:
            cond=['']
    else :
        cond=onsetNames
    #if Call=='1':
    tsall=[]
    for nameC in cond:
        print nameC
        try:#if True
            print nameC
            fout=outDir.replace('SUBJECT',s).replace('CONDITION',nameC)
            if not os.path.exists(fout):
                os.makedirs(fout)
            if NeuroIMAGE!='':
            
                fold=[glob.glob(data.replace('SUBJECT',s).replace('RUN',r).replace('CONDITION',nameC))[0] for r in allinfo[s]['run']]
        
            else:
                fold=glob.glob(data.replace('SUBJECT',s).replace('RUN','*').replace('CONDITION',nameC))
            
            ts=[]
            coverage=np.zeros(len(np.loadtxt(fold[0])))
            for f in fold:
                nameR=f[data.replace('SUBJECT',s).replace('CONDITION',nameC).find('RUN'):data.replace('SUBJECT',s).replace('CONDITION',nameC).find('RUN')+f.replace(f[:data.replace('SUBJECT',s).replace('CONDITION',nameC).find('RUN')],'').find('/')].replace('.feat','')
                print nameR
                if nameC=='':
                    name=data.replace(os.path.dirname(data),'').replace('/','').replace('.txt','')
                else :
                    name=f[f.find(nameC):len(f)-4]#f[f.find(nameC,-1):]
            #read, concatenate
                if ts==[]:
                    prov=np.loadtxt(f)#to normalize
                    ts=(prov-np.repeat([np.mean(prov,1)],len(prov[0]),axis=0).T)/(np.repeat([np.std(prov,1)],len(prov[0]),axis=0).T)
                else:
                    prov=np.loadtxt(f)#to normalize
                    ts=np.concatenate((ts,(prov-np.repeat([np.mean(prov,1)],len(prov[0]),axis=0).T)/(np.repeat([np.std(prov,1)],len(prov[0]),axis=0).T)),axis=1)

            #get coverage
                maskdat=nibabel.load(mask.replace('SUBJECT',s).replace('RUN',nameR).replace('CONDITION',nameC)).get_data()
                maskdat[np.where(maskdat!=0.)]=1
                atlasdat=nibabel.load(atlas.replace('SUBJECT',s).replace('RUN',nameR).replace('CONDITION',nameC)).get_data()
                atlasdat[np.where(atlasdat!=0.)]=1
                        
                coverage=np.sum([coverage,np.sum(np.sum(np.sum((atlasdat.T*maskdat.T).T,0),0),0)],0)

            if maxTP!=0:
                if f.find('HCP')!=-1:
                    ts=ts.T[[3*t for t in range(maxTP)]].T
                else:
                    ts=ts.T[:maxTP].T

        #reject area if needed, make matrices
            if rejet!='':
                ts=np.array([ts[i] for i in range(len(ts)) if i not in rejet])
                coverage=np.array([coverage[i] for i in range(len(coverage)) if i not in rejet])
            if method=='ledoit_wolf':
                [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(ts,method)
            else:
                [corM,ZparCor,arcparCor,covM,parCor]=makemat(ts, method)
  
            if maxTP!=0:
                tp='tp'+np.str(maxTP)
            else:
                tp=''
            if method!="":
                method2=np.str(method)+'_'
            else:
                method2=""
            if method=="ledoit_wolf":
                np.save(fout+name+'_ledoitworf_'+'_'.join([np.str(i) for i in rejet])+tp,lw)            
            np.savetxt(fout+name+'_covariance_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',covM)
            np.savetxt(fout+name+'_correlation_'+'_'.join([np.str(i) for i in rejet])+tp+'.txt',corM)
            np.savetxt(fout+name+'_partialCorrelation_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',parCor)
            np.savetxt(fout+name+'_ArcpartialCorrelation_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',arcparCor)
            np.savetxt(fout+name+'_ZpartialCorrelation_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',ZparCor)
            print fout+name
            if tsall==[]:
                tsall=ts
            else:
                tsall=np.concatenate((tsall,ts),axis=1)
        except :
            fail+=[s,nameC]
            print fail
    if (tsall!=[]) & (Call=='1'):
        if method=='ledoit_wolf':
            [corM,ZparCor,arcparCor,covM,parCor,lw]=makemat(tsall,method)
        else:
            [corM,ZparCor,arcparCor,covM,parCor]=makemat(tsall,method)
        if maxTP!=0:
            tp='tp'+np.str(maxTP)
        else:
            tp=''
        if method!="":
            method2=np.str(method)+'_'
        else:
            method2=""
        if method=="ledoit_wolf":
            np.save(os.path.dirname(fout[:len(fout)-1])+'/'+'_'.join(cond)+'_ledoitworf_'+'_'.join([np.str(i) for i in rejet])+tp,lw)            
        np.savetxt(os.path.dirname(fout[:len(fout)-1])+'/'+'_'.join(cond)+'_covariance_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',covM)
        np.savetxt(os.path.dirname(fout[:len(fout)-1])+'/'+'_'.join(cond)+'_correlation_'+'_'.join([np.str(i) for i in rejet])+tp+'.txt',corM)
        np.savetxt(os.path.dirname(fout[:len(fout)-1])+'/'+'_'.join(cond)+'_partialCorrelation_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',parCor)
        np.savetxt(os.path.dirname(fout[:len(fout)-1])+'/'+'_'.join(cond)+'_ArcpartialCorrelation_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',arcparCor)
        np.savetxt(os.path.dirname(fout[:len(fout)-1])+'/'+'_'.join(cond)+'_ZpartialCorrelation_'+method2+'_'.join([np.str(i) for i in rejet])+tp+'.txt',ZparCor)

