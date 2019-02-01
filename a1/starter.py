import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    total_loss = 0
    
    N = len(x)
    yhat=np.dot(np.transpose(W),x)
    MSEloss = yhat-y.flatten()+ b #need to figure out how to do this
    MSEloss=MSEloss **2
    loss=np.sum(abs(MSEloss))
		

    weight_decay_loss = (reg / 2) * np.sum(np.linalg.norm(np.dot(W,W)) ** 2)
    total_loss = (loss / (2 * N)) + weight_decay_loss
    return total_loss



def gradMSE(W, b, x, y, reg):
    pass

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    pass

def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    pass

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass

#if __name__ == "__main__":
def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    x, y = trainData, trainTarget
    W = np.zeros(x.shape)

    print(W.shape)

main()
