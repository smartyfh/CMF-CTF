import numpy as np
import os, sys, time
import multiprocessing
sys.path.append('src')
#print(sys.path.append('src'))
# Build external model
if not os.path.isfile('src/core.so'):
	print('Lack of core.so (built from the C++ module).') 
	print('Please first build the C++ code into core.so by using: ')
	print('>> python setup.py build_ext --inplace')
	sys.exit()

from utilities import *
import evaluator
import dataloader
 

#########################################################
# config area
#
para = {'dataType': 'rt', # set the dataType as 'rt' or 'tp'
		'dataPath': '',
		'outPath': 'res35/',
		'metrics': ['MAE', 'RMSE'], # delete where appropriate
		'density': [0.5], # matrix density
		'rounds': 5, # how many runs are performed at each matrix density
		'dimension': 15, # dimenisionality of the latent factors
		'gamma': 35, # maybe 10 for tp
		'lambdau': 0.1, # regularization parameter have tried 0.1
		'lambdas': 0.1,
		'lambdat': 0.1,
		'maxIter': 350, # the max iterations
		'saveTimeInfo': False, # whether to keep track of the running time
		'saveLog': False, # whether to save log into file
		'debugMode': False, # whether to record the debug info
		'parallelMode': False, # whether to leverage multiprocessing for speedup
		'outlier_fra': [0.02, 0.04, 0.06, 0.08, 0.1, 0.2] #list(np.arange(0.02, 0.11, 0.02))     
		}
initConfig(para)
#########################################################


startTime = time.clock() # start timing
logger.info('==============================================')
logger.info('Approach: Cauchy Tensor Factorization (CTF).')

# load the dataset
dataTensor = dataloader.load(para)
logger.info('Loading data done.')

# run for each density
# if para['parallelMode']: # run on multiple processes
# 	pool = multiprocessing.Pool()
# 	for density in para['density']:
# 		pool.apply_async(evaluator.execute, (dataTensor, density, para))
# 	pool.close()
# 	pool.join()
# else: # run on single processes
# 	for density in para['density']:
# 		evaluator.execute(dataTensor, density, para)


       
# run for each outlier_fra
if para['parallelMode']: # run on multiple processes
	pool = multiprocessing.Pool()
	for outlier_fra in para['outlier_fra']:
		pool.apply_async(evaluator.execute_outlier, (dataTensor, density, para))
	pool.close()
	pool.join()
else: # run on single processes
	for density in para['density']:
		evaluator.execute_outlier(dataTensor, density, para)


logger.info(time.strftime('All done. Total running time: %d-th day - %Hhour - %Mmin - %Ssec.',
		 time.gmtime(time.clock() - startTime)))
logger.info('==============================================')
sys.path.remove('src')
