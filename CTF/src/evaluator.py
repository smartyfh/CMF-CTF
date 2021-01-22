import numpy as np 
from numpy import linalg as LA
import time
import random
import core
from utilities import *


########################################################
# Function to run the prediction approach at each density
# 
def execute(tensor, density, para):
	startTime = time.clock()
	[numUser, numService, numTime] = tensor.shape
	rounds = para['rounds']
	logger.info('Data size: %d users * %d services * %d timeslices'\
		%(numUser, numService, numTime))
	logger.info('Run the algorithm for %d rounds: density = %.2f.'%(rounds, density))
	evalResults = np.zeros((rounds, len(para['metrics']))) 
	timeResults = np.zeros((rounds, 1))
	
	for k in range(rounds):
		logger.info('----------------------------------------------')
		logger.info('%d-round starts.'%(k + 1))
		logger.info('----------------------------------------------')

		# remove the entries of data to generate trainTensor and testTensor
		(trainTensor, testTensor, indtrainTensor, indtestTensor) = removeTensor(tensor, density, k, para)
		logger.info('Removing data entries done.')
     # remove outliers from the testing data
		(testTensor, indtestTensor) = merge_test_outlier(testTensor, indtestTensor, para['outlier_fra'], para)
		logger.info('Merge outlier done.')

		# invocation to the prediction function
		iterStartTime = time.clock() # to record the running time for one round				
		predictedTensor = core.predict(trainTensor, para) 
		timeResults[k] = time.clock() - iterStartTime

		# calculate the prediction error
		evalResults[k, :] = cal_metric(testTensor, predictedTensor, indtestTensor)
		print(evalResults)

		logger.info('%d-round done. Running time: %.2f sec'%(k + 1, timeResults[k]))
		logger.info('----------------------------------------------')

	outFile = '%savg_%sResult_density%.2f.txt'%(para['outPath'], para['dataType'], density)
	saveResult(outFile, evalResults, timeResults, para)

	logger.info('Density = %.2f done. Running time: %.2f sec'
			%(density, time.clock() - startTime))
	logger.info('==============================================')
########################################################


########################################################
# Function to run the prediction approach at each outlier
# 
def execute_outlier(tensor, density, para):
	startTime = time.clock()
	[numUser, numService, numTime] = tensor.shape
	rounds = para['rounds']
	logger.info('Data size: %d users * %d services * %d timeslices'\
		%(numUser, numService, numTime))
	logger.info('Run the algorithm for %d rounds: density = %.2f.'%(rounds, density))
	evalResults = np.zeros((len(para['outlier_fra']), rounds, len(para['metrics']))) 
	timeResults = np.zeros((rounds, 1))
	
	for k in range(rounds):
		logger.info('----------------------------------------------')
		logger.info('%d-round starts.'%(k + 1))
		logger.info('----------------------------------------------')

		# remove the entries of data to generate trainTensor and testTensor
		(trainTensor, testTensor0, indtrainTensor, indtestTensor0) = removeTensor(tensor, density, k, para)
		logger.info('Removing data entries done.')

		# invocation to the prediction function
		iterStartTime = time.clock() # to record the running time for one round				
		predictedTensor = core.predict(trainTensor, para) 
		timeResults[k] = time.clock() - iterStartTime
		# calculate the prediction error

		o = 0
		for otf in para['outlier_fra']:
			# testTensor1 = merge_test_outlier(testTensor0, otf, para)
			# logger.info('Merge outlier done.')
			# result = np.zeros((numTime, len(para['metrics'])))

			# for i in range(numTime):
			# 	testMatrix = testTensor1[:, :, i]
			# 	predictedMatrix = predictedTensor[:, :, i]
			# 	(testVecX, testVecY) = np.where(testMatrix)
			# 	testVec = testMatrix[testVecX, testVecY]
			# 	predVec = predictedMatrix[testVecX, testVecY]
			# 	result[i, :] = errMetric(testVec, predVec, para['metrics'])
			# evalResults[o, k, :] = np.average(result, axis=0)
			(testTensor1, indtestTensor1) = merge_test_outlier(testTensor0, indtestTensor0, otf, para)
			logger.info('Merge outlier done.')
			evalResults[o, k, :] = cal_metric(testTensor1, predictedTensor, indtestTensor1)
			o = o + 1

		logger.info('%d-round done. Running time: %.2f sec'%(k + 1, timeResults[k]))
		logger.info('----------------------------------------------')

	o = 0
	for outlf in para['outlier_fra']:
		outFile = '%savg_%sResult_outlier_fra%.2f.txt'%(para['outPath'], para['dataType'], outlf)
		saveResult(outFile, evalResults[o, :, :], timeResults, para)
		o = o + 1

	logger.info('Density = %.2f done. Running time: %.2f sec'
			%(density, time.clock() - startTime))
	logger.info('==============================================')
########################################################


########################################################
# Function merge_test_outlier
#
def merge_test_outlier(testTensor, indtestTensor, outlier_fra, para):
	m, n, t = testTensor.shape  
	outlier_filename = para['dataPath']+"outlier_"+para['dataType']+"_new/Full_Time_I_outlier_fra_"+str(outlier_fra)+"-new.npy"
	logger.info('Load outlier file.')
	I_outlier = np.load(outlier_filename)
	for i in range(m):
		for j in range(n):
			for k in range(t):
				if I_outlier[i][j][k] == 1:
					testTensor[i][j][k] = 0
					indtestTensor[i][j][k] = 0
	return testTensor, indtestTensor
########################################################

########################################################
# Function to remove the entries of data tensor
# Return the trainTensor and the corresponding testTensor
#
def removeTensor(tensor, density, round, para):
	numTime = tensor.shape[2]
	trainTensor = np.zeros(tensor.shape)
	I_trainTensor = np.zeros(tensor.shape)
	testTensor = np.zeros(tensor.shape)
	I_testTensor = np.zeros(tensor.shape)
	for i in range(numTime):
		seedID = round + i * 100
		(trainMatrix, testMatrix, I_trainMatrix, I_testMatrix) = removeEntries(tensor[:, :, i], density, seedID)
		trainTensor[:, :, i] = trainMatrix
		testTensor[:, :, i] = testMatrix
		I_trainTensor[:, :, i] = I_trainMatrix
		I_testTensor[:, :, i] = I_testMatrix
	return trainTensor, testTensor, I_trainTensor, I_testTensor
########################################################


########################################################
# Function to remove the entries of data matrix
# Return the trainMatrix and the corresponding testing data
#
def removeEntries(matrix, density, seedID):
	(vecX, vecY) = np.where(matrix > 0)
	vecXY = np.c_[vecX, vecY]
	numRecords = vecX.size
	numAll = matrix.size
	random.seed(seedID)
	randomSequence = list(range(0, numRecords))
	random.shuffle(randomSequence) # one random sequence per round
	numTrain = int(numRecords * density)
	# by default, we set the remaining QoS records as testing data					   
	numTest = numRecords - numTrain
	trainXY = vecXY[randomSequence[0 : numTrain], :]
	testXY = vecXY[randomSequence[- numTest :], :]

	trainMatrix = np.zeros(matrix.shape)
	trainMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
	testMatrix = np.zeros(matrix.shape)
	testMatrix[testXY[:, 0], testXY[:, 1]] = matrix[testXY[:, 0], testXY[:, 1]]
	I_trainMatrix = np.zeros(matrix.shape)
	I_trainMatrix[trainXY[:, 0], trainXY[:, 1]] = 1
	I_testMatrix = np.zeros(matrix.shape)
	I_testMatrix[testXY[:, 0], testXY[:, 1]] = 1

	# ignore invalid testing			 
	idxX = (np.sum(trainMatrix, axis=1) == 0)
	testMatrix[idxX, :] = 0
	I_testMatrix[idxX, :] = 0
	idxY = (np.sum(trainMatrix, axis=0) == 0)
	testMatrix[:, idxY] = 0	 
	I_testMatrix[:, idxY] = 0  
	return trainMatrix, testMatrix, I_trainMatrix, I_testMatrix
########################################################


########################################################
# Function to compute the evaluation metrics
#
def errMetric(realVec, predVec, metrics):
	result = []
	absError = np.abs(predVec - realVec) 
	mae = np.sum(absError)/absError.shape
	for metric in metrics:
		if 'MAE' == metric:
			result = np.append(result, mae)
		if 'RMSE' == metric:
			rmse = LA.norm(absError) / np.sqrt(absError.shape)
			result = np.append(result, rmse)
	return result
########################################################


########################################################
# Function to compute the evaluation metrics
#
def cal_metric(testData, predData, indTensor):
	resData = indTensor * (testData - predData)
	numData = np.sum(indTensor)

	mae = np.sum(abs(resData)) / numData

	rmse = np.sum(resData * resData) / numData
	rmse = rmse**0.5

	return mae, rmse
########################################################
