#!/python2.7

## catch all matrices to perform a mixture modelling using MELODIC and save the output

# In[2]:

import sys
sys.path.append('/')
import mm_helpers
import argparse, os, glob
import numpy as np
import mm_GetInfoFromDb
import subprocess
import nibabel
import numpy as np





serie='full'
cond='179'
atlas='atlas_complete_GM'
matrices='ZpartialCorrelation'
if matrices =='ZpartialCorrelation' or matrices =='partialCorrelation' or matrices =='ArcpartialCorrelation':
    method='_ledoit_wolf'
else:
    method=''
movement=False


# In[5]:

if cond=='179':
    number=np.int(cond)
    if serie=='full':
        if cond=='179':
            data=['/fMRI_RS/SUBJECT/ICA_AROMA_nonaggr_nr_hp_atlas_subcort_cerebellum_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160.txt',
            '/fMRI_STOP/SUBJECT/denoised_func_data_nonaggr_hp_atlas_subcort_cerebellum_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160.txt',
            '/fMRI_REWARD/SUBJECT/denoised_func_data_nonaggr_hp_atlas_subcort_cerebellum_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160.txt',
            '/fMRI_WM/SUBJECT/denoised_func_data_nonaggr_hp_atlas_subcort_cerebellum_1mm_4D_warped_th_reg_'+matrices+''+method+'_149_152_155_158_159_160.txt',
            ]
        NeuroIMAGE=['RS','STOP','REWARD','WM']
if atlas=='atlas_complete_GM':
    for i in range(len(data)):
        data[i]=data[i].replace('atlas_subcort_cerebellum','atlas_complete_GM')

    
# In[3]:
if True:
    suj=[np.unique([mm_helpers.find_subject(i,'NeuroIMAGE') for i in glob.glob(j.replace('SUBJECT','*').replace('CONDITION','*'))]) for j in data]
    selec=len(NeuroIMAGE)
    allinfo=[dict() for t in range(selec)]

    for t in range(selec):
        for s in suj[t]:
            allinfo[t].update({s : {s:s,'run':[]}})
        
        suj[t]=sorted(allinfo[t].keys())



keep=[]
Upmatrices=[[] for t in range(len(NeuroIMAGE))]

tri=np.zeros((number,number))
tri[np.triu_indices(number,1)]=1

for t in range(len(NeuroIMAGE)):
    print NeuroIMAGE[t]
    prov2=[]
    for s in suj[t]:
        
        dat=glob.glob(data[t].replace('SUBJECT',s).replace('CONDITION','*'))
        print s,len(dat)

        for f in dat:
            try:
                prov=np.loadtxt(f)
                #mixture modelling using a nifti to input in Mixture Modelling from MELODIC
                test=nibabel.load('/test.nii.gz')
                nibabel.save(nibabel.spatialimages.SpatialImage(prov[np.triu_indices(number,1)],test.get_affine(),test.get_header()),'/test.nii.gz')
                nibabel.save(nibabel.spatialimages.SpatialImage(np.sign(np.abs(prov[np.triu_indices(number,1)])),test.get_affine(),test.get_header()),'/test2.nii.gz')
                subprocess.call(['mm','-m','/test2.nii.gz','--sdf=/test.nii.gz','--ns','-l','/testMM'])
            #normalization
                prov[np.triu_indices(number,1)]=(prov[np.triu_indices(number,1)]-np.loadtxt('/home/mrstats/roscha/NeuroIMAGE/atlases/testMM/mu_mean')[0])/np.sqrt(np.loadtxt('/home/mrstats/roscha/NeuroIMAGE/atlases/testMM/var_mean')[0])
                prov=prov*tri+(prov*tri).T
            #clean up
                subprocess.call(['rm','-r','/testMM'])
                np.savetxt(f.replace('.txt','NORM.txt'),prov)
            except:
                keep+=[np.str(t)+'_'+s]
                #clean up
                subprocess.call(['rm','-r','/testMM'])

print keep
