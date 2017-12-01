#!/python2.7


import argparse, os, glob, random,sys
import numpy as np

def check_movers(allinfo,percent=5.,database='neuroimage'):
    import sys
    sys.path.append('/home/mrstats/maamen/DCCN/Scripts/NeuroImage/')
    sys.path.append('/home/mrstats/roscha/Scripts/TEN/scripts/')
    import rc_GetInfoFromDb
    queryT = "SELECT `subject_number`,`SeriesNumber`,`ScanProtocol`,`rmsFD_Jenkinson`  FROM `MRI_series_info` "
    jen=rc_GetInfoFromDb.get_query(queryT, ['subject','sn','sp','jen'], database, True)
    keep=[]
    suj=allinfo.keys()
    keepjen=np.zeros(len(suj))

    for k in jen.keys():
        if jen[k]['subject'] in suj and np.str(jen[k]['sn'])+'_'+np.str(jen[k]['sp']) in allinfo[jen[k]['subject']]['run']:

            keepjen[suj.index(jen[k]['subject'])]+=np.float(jen[k]['jen'])/np.float(len(allinfo[jen[k]['subject']]['run']))#[i for i in range(len(suj)) if suj[i]==jen[k]['subject']
    Nrejet=int(len(suj)*percent/100.)
    sortKJ=[i[0] for i in sorted(enumerate(keepjen), key=lambda x:x[1])]
    for k in sortKJ[len(sortKJ)-Nrejet:]:
        keep+=[suj[k]]
        del allinfo[suj[k]]

    print keep
    return allinfo


def selectControl(allinfo,database='neuroimage'):
    import sys
    sys.path.append('/home/mrstats/maamen/DCCN/Scripts/NeuroImage/')
    sys.path.append('/home/mrstats/roscha/Scripts/TEN/scripts/')
    import rc_GetInfoFromDb
    queryT = "SELECT `Subject`,`type`,`Diagnosis_Overall_Type2`  FROM `Phenotype_info` "
    typ=rc_GetInfoFromDb.get_query(queryT, ['subject','type','Diag'], database, True)
    print len(allinfo.keys())
    for i in typ.keys():
        if typ[i]['subject'] in allinfo.keys() and typ[i]['type']!='controle' and typ[i]['Diag']!=1:
            del allinfo[typ[i]['subject']]
    print len(allinfo.keys())
    return allinfo
