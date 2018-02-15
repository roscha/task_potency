#!/python2.7


#run as :
#run /reg_subjects_GENERIC.py -data /fMRI_TASK/SUBJECT/RUN.feat/nuissance/denoised_func_data_nonaggr/denoised_func_data_nonaggr_hp.nii.gz -atlas /fMRI_TASK/SUBJECT/RUN.feat/atlases/atlas_subcort_cerebellum_1mm_4D_warped_th.nii.gz -mask /fMRI_TASK/SUBJECT/RUN.feat/mask.nii.gz -NeuroIMAGE REWARD -outDir //fMRI_TASK/SUBJECT/RUN.feat/nuissance/denoised_func_data_nonaggr/times_series

import sys
sys.path.append('/')
import mm_helpers
import argparse, os, glob, random,sys
import numpy as np
import mm_GetInfoFromDb
import subprocess
from sklearn.decomposition import PCA
import nibabel


##-----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
					description='''do a regression (time serie according to an atlas), design for beta serie hierarchie with subject, run, condition''') 

# required options 
reqoptions = parser.add_argument_group('Required Arguments')
reqoptions.add_argument('-data', action='store', dest = 'data', required = True, help = 'full path to 4D file to extract times series from, with SUBJECT/RUN')
reqoptions.add_argument('-atlas', action='store', dest = 'atlas', required = True, help = 'full path to 4D atlas.nii.gz (one region of interest per volume) than we want to use for define times series with SUBJECT/RUN if necessary')
reqoptions.add_argument('-mask', action='store', dest = 'mask', required = True, help = 'full path to 3D mask.nii.gz  with SUBJECT/RUN if necessary')

# optional options
optoptions = parser.add_argument_group('Optional Arguments')
optoptions.add_argument('-outDir', action='store', dest = 'outDir', required = False, default='', help = 'if not define, files will be stored in data directoroy/time_series, full path to the folder name to store the times series as matrix python file with SUBJECT/RUN if necessary e.g SUBJECT/RUN/times_series, output file will be named dataName_atlasName_TSmethod')
optoptions.add_argument('-NeuroIMAGE', action='store', dest = 'NeuroIMAGE', required = False, default='', help = 'can be set as "STOP" or "REWARD" or "WM" for NeuroIMAGE, enable to catch info from DB, else info will be catch in raw data (longer)')
optoptions.add_argument('-listSubject', action='store', dest = 'listSubject', required = False, default='', help = 'full path to a subject list than you want to process')
optoptions.add_argument('-TReg', action='store', dest = 'TReg', required = False, default='0', help = 'also do a triple regression (1), default 0')

#_main_
## parse arguments
args = parser.parse_args()
data=args.data
mask_path=args.mask
atlas=args.atlas
TReg=args.TReg

outDir=args.outDir
if outDir=='':
	outDir=os.path.dirname(data)

NeuroIMAGE =args.NeuroIMAGE
listSubject=args.listSubject





if listSubject=='':
	if NeuroIMAGE!='':
		queryT = "SELECT `subject_number`  FROM `MRI_series_info` WHERE `ScanProtocol` = 'fMRI_"+NeuroIMAGE+"'"
		suj=np.unique(mm_GetInfoFromDb.get_query(queryT,['subject']))
		suj=np.unique([i[data.index('SUBJECT'):data.index('SUBJECT')+i.replace(data[:data.index('SUBJECT')],'').find('/')] for i in glob.glob(data.replace('SUBJECT','*').replace('RUN','*').replace('CONDITION','*'))])
		allinfo=dict()
		for s in suj:
			allinfo.update({s : s})
		allinfo=mm_helpers.T1_selection(allinfo)
		suj=sorted(allinfo.keys())

	else:
		suj=np.unique([i[data.index('SUBJECT'):data.index('SUBJECT')+i.replace(data[:data.index('SUBJECT')],'').find('/')] for i in glob.glob(data.replace('SUBJECT','*').replace('RUN','*').replace('CONDITION','*'))])
else :
	suj=open(listSubject).read().split()

for s in suj :
	try:
		fold=glob.glob(data[:data.index('CONDITION')].replace('SUBJECT',s).replace('RUN','*'))
	except:
		fold=glob.glob(data.replace('SUBJECT',s).replace('RUN','*'))
	for f in fold:
		if f.replace(f[:data.replace('SUBJECT',s).find('RUN')],'').find('/')!=-1:
			nameR=f[data.replace('SUBJECT',s).find('RUN'):data.replace('SUBJECT',s).find('RUN')+f.replace(f[:data.replace('SUBJECT',s).find('RUN')],'').find('/')].replace('.feat','')
		elif data.find('RUN')!=-1:
			nameR=f[data.replace('SUBJECT',s).find('RUN'):len(f)-len(data[data.replace('SUBJECT',s).find('RUN')+4:])].replace('.feat','')

		cond=glob.glob(data.replace('SUBJECT',s).replace('RUN',nameR).replace('CONDITION','*'))
		for c in cond :
			nameC=c[data.replace('SUBJECT',s).replace('RUN',nameR).find('CONDITION'):len(c)-7]#data.replace('SUBJECT',s).replace('RUN',nameR).find('CONDITION')+f.replace(f[:data.replace('SUBJECT',s).replace('RUN',nameR).find('CONDITION')],'').find('/')].replace('.feat','')
			if nameC=='':
				name=data.replace(os.path.dirname(data),'').replace('/','').replace('.nii.gz','')+'_'+atlas.replace(os.path.dirname(atlas),'').replace('.nii.gz','').replace('/','')
			else :
				name=nameC
			fout=outDir.replace('SUBJECT',s).replace('RUN',nameR).replace('CONDITION',nameC)
			if not os.path.exists(fout):
				os.makedirs(fout)

			test = False 

			if True:not test:#len(glob.glob(fout+'/times_series/*.txt'))!=4 and len(glob.glob(fout+'/times_series/*.nii.gz'))!=3:
				print s, nameR
#				break
		

				# use mm_helpers.py to create temporal regression and triple regression times series.
				if True:#not os.path.exists(fout+'/'+name+'_reg.txt') and not os.path.exists(fout+'/'+name.replace('RESTRUN','RUNREST')+'_reg.txt'):
					if os.path.exists(fout+'/'+name+'_regression.txt'):
						os.rename(fout+'/'+name+'_regression.txt',fout+'/'+name+'_reg.txt')
					else:
						try:	#if True:#
							atlas_path=atlas.replace('SUBJECT',s).replace('RUN',nameR)
							a= nibabel.load(atlas_path).get_data()
							data_path=glob.glob(data.replace('SUBJECT',s).replace('RUN',nameR).replace('CONDITION',nameC))[0]
							d= nibabel.load(data_path).get_data()
							m= nibabel.load(mask_path.replace('SUBJECT',s).replace('RUN',nameR)).get_data()
						
							mat=mm_helpers.regression(d, a, m)
							if nameC!='':
								np.savetxt(fout+'/'+name+'_'+atlas.replace(os.path.dirname(atlas),'').replace('.nii.gz','').replace('/','')+'_reg.txt',mat)
							else:
								np.savetxt(fout+'/'+name+'_reg.txt',mat)
						except:
							print s,nameR,nameC,'R failed'


				if not os.path.exists(fout+'/'+name+'_triple_reg.txt') and TReg=='1':
					try:	
						atlas_path=atlas.replace('SUBJECT',s).replace('RUN',nameR)
						
						data_path=data.replace('SUBJECT',s).replace('RUN',nameR).replace('CONDITION',nameC)
						
						mat=mm_helpers.triple_regression(atlas_path, data_path)
						np.savetxt(fout+'/'+name+'_triple_reg.txt',mat)
					except:
						print s,nameR,nameC,'TR failed'
			
		
	
