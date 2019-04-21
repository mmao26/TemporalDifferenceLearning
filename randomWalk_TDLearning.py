### Author Name: Manqing Mao
### GTID: mmao33
### Temporal Difference (Lambda)

import numpy as np
import random as rand
import matplotlib.pyplot as plt 
from math import sqrt

def sequence_generator():              # One sequence of random walk generator
    xi = [0., 0., 0., 1., 0., 0., 0.]; # Initialization of xi (at center D)
    Seq = [np.array(xi)]
    while True:
        nxtwalk = rand.randint(0, 1)   # left 0, right 1
        curPos = xi.index(1.)
        nxtPos = curPos+1 if nxtwalk else curPos-1
        xi[nxtPos] = 1.
        xi[curPos] = 0.
        Seq.append(np.array(xi))
        if nxtPos in [0, len(xi)-1]:   # sequence done at A or G
            break
    return Seq
    
def trainingSet_generator(num_seq):    # One training set of random walk generator
    Seqs_TrainSet = []
    for i in range(num_seq):
        Seqs_TrainSet.append(sequence_generator())
    return Seqs_TrainSet

def allTrainingData_generator(num_trainSet, num_seq):  # All training data with several training sets
    Seqs_all = []
    for i in range(num_trainSet):
        Seqs_all.append(trainingSet_generator(num_seq))
    return Seqs_all

def weightUpdateAfterSet(alpha, lamda, oneTrainSet, Weight):     # Experimental 1
    # Repeated 4ever until convergence
    while True:                                        
        deltaWeight = np.zeros(7)
        for oneSeq in oneTrainSet:
            for t in range(len(oneSeq)-1):
                Pred_t1 = np.dot(Weight.transpose(), oneSeq[t+1])
                Pred_t0 = np.dot(Weight.transpose(), oneSeq[t])

                sum_LambdaPk = np.zeros(7)           
                for k in range (1, t+1):
                    sum_LambdaPk += pow(lamda, t-k) * oneSeq[k]
                # Accumulate delta weight
                deltaWeight += alpha * (Pred_t1 - Pred_t0) * sum_LambdaPk
        # Update weight after complete a training set
        Weight += deltaWeight
        # Convergence Condition: delta weight is small enough
        if np.sqrt(np.dot(deltaWeight.transpose(), deltaWeight)) <= 0.001:       
            break       
    return Weight


def weightUpdateAfterSeq(alpha, lamda, oneTrainSet, Weight):      # Experimental 2
    Weights = Weight.copy()
    for oneSeq in oneTrainSet:
        
        for t in range(len(oneSeq)-1):
            Pred_t1 = np.dot(Weights.transpose(), oneSeq[t+1])
            Pred_t0 = np.dot(Weights.transpose(), oneSeq[t])

            sum_LambdaPk = np.zeros(7)
            for k in range (1, t+1):
                sum_LambdaPk += pow(lamda, t-k) * oneSeq[k]            
            # Update weight after complete Squence
            Weights += alpha * (Pred_t1 - Pred_t0) * sum_LambdaPk
    return Weights

def rmseCal(prediction, ideal):                         # RMSE calculation
    return np.sqrt(((prediction - ideal) ** 2).mean())

if __name__ == "__main__":

    plot_fig = 5                     # 3 for Fig. 3, 4 for Fig. 4 and 5 for Fig. 5

    rand.seed(1)                     # Set a random seed
    num_TrainingSet = 100;           # The number of training set
    num_Seqs = 10;                   # The number of squences per training set
    trainingData = allTrainingData_generator(num_TrainingSet, num_Seqs)
    print(len(trainingData))
    idealPred = [1./6., 1./3., 1./2., 2./3., 5./6.]             # ideal predictions
    initalPred = np.array([0., 0.5, 0.5, 0.5, 0.5, 0.5, 1.])    # initial predictions
    
    if plot_fig == 3:
        lambdaSet = [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.]
        alpha = 0.005
        avgRMSE = []
        for lamda in lambdaSet:
            RMSE = []
            for eachTrainSet in trainingData:
                weightPred1 = weightUpdateAfterSet(alpha, lamda, eachTrainSet, initalPred)
                RMSE.append(rmseCal(weightPred1[1:6], idealPred))
            avgRMSE.append(np.mean(RMSE))
            
        plt.figure(1)
        plt.plot(lambdaSet, avgRMSE, marker='8')
        plt.xlabel(r'$\lambda$', fontsize=14)
        plt.ylabel('ERROR USING BEST ' r'$\alpha$', fontsize=13)
        plt.show()

    elif plot_fig == 4:
        lambdaSet = [0., 0.3, 0.8, 1.]
        alphaSet = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
        all_avgRMSE = []
        for lamda in lambdaSet:
            avgRMSE = []
            for alpha in alphaSet:
                RMSE = []
                for eachTrainSet in trainingData:
                    weightPred = weightUpdateAfterSeq(alpha, lamda, eachTrainSet, initalPred)
                    RMSE.append(rmseCal(weightPred[1:6], idealPred))
                avgRMSE.append(np.mean(RMSE))
            all_avgRMSE.append(avgRMSE)
            
        plt.figure(2)
        for i in range(4):
            plt.plot(alphaSet, all_avgRMSE[i], marker='8')
        plt.xlabel(r'$\alpha$', fontsize=15)
        plt.ylabel('ERROR', fontsize=13)
        plt.legend([r'$\lambda$ = 0', r'$\lambda$ = 0.3', r'$\lambda$ = 0.8', r'$\lambda$ = 1'])
        plt.ylim((0., 0.7))
        plt.show()

    elif plot_fig == 5:
        lambdaSet = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        alphaSet = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
        lowestRMSE = []
        
        for lamda in lambdaSet:
            errorBestAlpha = float("inf")
            for alpha in alphaSet:
                RMSE = []
                for eachTrainSet in trainingData:
                    weightPred = weightUpdateAfterSeq(alpha, lamda, eachTrainSet, initalPred)
                    RMSE.append(rmseCal(weightPred[1:6], idealPred))
                if np.mean(RMSE) < errorBestAlpha:
                    errorBestAlpha = np.mean(RMSE)
                    bestAlpha = alpha
            lowestRMSE.append(errorBestAlpha)

        plt.figure(3)
        plt.plot(lambdaSet, lowestRMSE, marker='8')
        plt.xlabel(r'$\lambda$', fontsize=14)
        plt.ylabel('ERROR USING BEST ' r'$\alpha$', fontsize=13)
        plt.show()

        
