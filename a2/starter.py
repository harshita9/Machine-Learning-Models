import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.maximum(x,0)

def softmax(x):
    sume=np.sum(np.exp(x))

    return np.exp(x)/sume



def computeLayer(X, W, b):

    return np.dot(np.transpose(W),X)+b

def CE(target, prediction):
    # total cross entropy loss
    cross_entropy_loss = 0

    yhat=prediction
    y=target

    N = (target.shape[0])
    K=(target.shape[1])

    # get z using the sigmax function

    z=softmax(yhat)
    innerloop=np.dot(y,np.log(z))


    # calculate total cross entropy loss
    cross_entropy_loss = np.sum(innerloop)/(-1*N)

    return cross_entropy_loss


def gradCE(target, prediction):

    yhat=prediction
    y=target
    N = (target.shape[0])
    z=softmax(yhat)

    #return (-1/N)*np.true_divide(np.transpose(y),z)
    return (-1/N)*(np.transpose(y)/z)

def outergradCE(target,prediction,x):
    yhat=prediction
    y=target
    N = (target.shape[0])
    z=softmax(yhat)
    gradb= (z-np.transpose(y))
    return np.dot(gradb,np.transpose(x)),np.sum(gradb,axis=1)



trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
newtrain, newvalid, newtest= convertOneHot(trainTarget, validTarget, testTarget)
x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))

print(x.shape)
print(x.shape)
print(softmax(x).shape)
deviation=np.sqrt(2/(x.shape[1]+newtrain.shape[1]))
W=np.random.normal(0,deviation, size=(x.shape[1],newtrain.shape[1]))
#W=np.transpose(W)
b=1
x=np.transpose(x)
prediction=computeLayer(x,W,b)
#print(prediction)
#print(CE(newtrain,prediction))
wg,bg=outergradCE(newtrain,prediction,x)
print(wg.shape,bg.shape)



    # TODO
