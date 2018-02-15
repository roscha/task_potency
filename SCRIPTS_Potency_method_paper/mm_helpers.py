#!/python2.7

## Various Helper functions
## Written by Maarten Mennes
## 

# find NeuroIMAGE subjectnumber in a path or string
def find_subject(mystring, sample, fcon_add=False):

	import string
	# make translation table to check for not allowed chars
	trans_table = string.maketrans('','')
	
	# dictionary of allowed and not allowed characters
	samples = dict(NeuroIMAGE = dict(yes = ['19-', '98-'], no = string.ascii_letters + '_. '),
					ADHD200 = dict(yes = ['0','1','2','3','4','5','6','7','8','9'], no = string.ascii_letters + '-_. '),
					ABIDE = dict(yes = ['005'], no = string.ascii_letters + '-_. '),
					fcon1000 = dict(yes = ['_sub0','_sub1','_sub2','_sub3','_sub4','_sub5','_sub6','_sub7','_sub8','_sub9'], no = '-. '),
					BIG = dict(yes = ['BIG0','BIG1','BIG2','BIG3','BIG4','BIG5','BIG6','BIG7','BIG8','BIG9'], no = '-_.txfcon '),
					bigc = dict(yes = ['bigc0', 'bigc1'], no = '-_.txfon '),
					s = dict(yes = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9'], no = '-_.qwertyuiopadfghjklzxcvbnm '),
					S = dict(yes = ['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9'], no = '-_.qwertyuiopadfghjklzxcvbnm '),
					sub = dict(yes = ['sub'], no = '-_.qwertyiopadfghjklzxcvnm '),
					NumbersOnly = dict(yes = ['0','1','2','3','4','5','6','7','8','9'], no = string.ascii_letters + '-_. '),
					Compuls = dict(yes = ['10-', '35-', '50-', '65-'], no = string.ascii_letters + '_. '),
					KCL = dict(yes = ['TRADA0','TRADA1','TRADA2'], no ='-_.qwertyuiopadfghjklzxcvbnm '),
					pfizer = dict(yes = ['B7441004_','B7441007_'], no ='-.qwertyuiopadfghjklzxcvbnm '),
					Matrics = dict(yes = ['0','1','2','3','4','5','6','7','8','9'], no = string.ascii_letters + '_. '))
	
	# split string based on 
	def determine(parts, samples, trans_table):
		for part in parts:
			for condition in samples[sample]['yes']:
				if condition in part:
					if part.translate(trans_table, samples[sample]['no']) == part:
						return part
					elif not sample == 'fcon1000':
						newparts = part.split('_')
						if len(newparts) == 1:
							break
						else:
							out = determine(newparts, samples, trans_table)
							return out
						
	parts = mystring.split('/')
	out = determine(parts, samples, trans_table)
	
	if fcon_add == True:
		print mystring
		site = mystring.split(out)[0].split('/')[-2]
		out = site + '_' + out
	
	if out == None:
		return ''
	else:
		return out


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


# do triple regression to get subject specific time-courses of subject specific spatial maps
def triple_regression(templates, mydata):
    
    import nibabel as nb
    import numpy as np
    
    # load templates
    img = nb.load(templates)
    templates = img.get_data()
    if len(img.shape) == 4:
        templates = np.reshape(templates, (np.prod(img.shape[:3]), img.shape[3]))
    else:
        templates = np.reshape(templates, (np.prod(img.shape)))
    
    # load data
    try:
        mydata.shape
        #print mydata.shape
    except AttributeError:
        img = nb.load(mydata)
        mydata = img.get_data()
        mydata = np.reshape(mydata, (np.prod(img.shape[:3]), img.shape[3]))
        #print mydata.shape
    
    # obtain mask
    mask = np.sign(np.min(np.abs(mydata), axis=1))

    # do first regression to get template specific time courses
    #print mydata.shape, templates.shape, mask.shape
    ts = regression(mydata, templates, mask, demean=True, desnorm=False)
    #print ts.shape
    
    # do second regression to get subject specific spatial maps
    #print mydata.shape, ts.T.shape, mask.shape
    sm = regression(mydata, ts.T, mask, demean=True, desnorm=True)
    #print sm.shape
    
    # do third regression to get subject specific time courses
    sts = regression(mydata, sm.T, mask, demean=True, desnorm=False)
    #print sts.shape
    
    return sts
    
    
    
    
    
    
