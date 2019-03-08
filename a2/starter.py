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

def reluDerivative(x):
    gradrelu=np.ones((x.shape[0],x.shape[1]))
    gradrelu[x<=0]=0
    return gradrelu

def softmax(x):
    sume=np.sum(np.exp(x),axis=0)

    return np.exp(x)/sume

def softmaxDerivative(x2,K,C):
    softmaxder=np.zeros(shape=x2.shape)

    for n in range(len(x2)):
        for i in range(K):
            for j in range(C):
                if i==j:
                    softmaxder[n][i]=x2[n][j] *(1-x2[n][j])
                else:
                    softmaxder[n][i]=-1*x2[n][j] * x2[n][j]

    return softmaxder


def computeLayer(X, W, b):
    return np.dot(np.transpose(W),X)+b

def CE(target, prediction):
    # total cross entropy loss
    cross_entropy_loss = 0
    #yhat is the final prediction in the final layer
    yhat=prediction
    y=target

    N = (target.shape[0])

    innerloop=np.dot(y,np.log(yhat))


    # calculate total cross entropy loss
    cross_entropy_loss = np.sum(innerloop)/(-1*N)

    return cross_entropy_loss


def gradCE(target, prediction): #check

    yhat=prediction
    y=target
    N = (target.shape[0])

    #get the gradient of error with respect to s
    return (-1/N)*(np.transpose(y)/yhat)

def outergradCE(target,x2,x1,s2,K,C):
    N = (target.shape[0])
    #get the gradient with respect to s
    gradS=gradCE(target,x2)
    #get the outer layer gradient
    gradsoftmax=softmaxDerivative(x2,K,C)
    #get the delta for outer layer
    deltaouter=np.dot(gradS,gradsoftmax)
    #get the gradient with respect to outer weights
    gradWouter=np.dot(x1,deltaouter)
    #get the gradient with respect to outer bias
    gradBouter=np.sum(deltaouter,axis=0)


    return deltaouter,gradWouter,gradBouter


def innergradCE(x, deltaouter,w2,s1):

    #get the delta for inner layer
    deltainner=np.matmul(reluDerivative(s1),np.dot(w2,deltaouter))
    #get the gradient with respect to inner layer weight
    gradwinner=np.dot(x,deltainner)
    #get the gradietn with respect to inner bias
    gradb=np.sum(deltainner,axis=0)

    return gradwinner,gradb

def forwardPropogation(x,w1,w2,b1,b2,y):
    #compute first layer prediction
    s1=computeLayer(x.T,w1,b1)
    #use the activation function
    x1=relu(s1)

    #compute the last layer prediction
    s2=computeLayer(x1,w2,b2)
    #use the softmax activation funtion
    prediction=softmax(s2)

    #get cross entropy loss
    CEloss=CE(y,prediction)

    return s1,x1,s2,prediction,CEloss


def backPropogation(x,x1,x2,w1,w2,s1,s2,y,K,C):
    #outer gradients
    deltaouter,gradWouter,gradBouter=outergradCE(y,x2,x1,s2,K,C)
    #inner gradients
    gradwinner,gradb=innergradCE(x, deltaouter,w2,s1)

    return gradWouter,gradBouter,gradwinner,gradb

def calculateAccuracy(prediction,y):

    predictedclasses=np.argmax(prediction,axis=1)
    count=0
    N=len(predictedclasses)
    for i in range(N):
        if prediction[i][predictedclasses[i]]!=0:
            count+=1

    return count/N * 100


def GD(trainingData, trainingLabels, alpha, iterations, gamma,K):

    # reshape x to be a 2D array (number of samples x 784)
    x = x=trainingData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    x = np.transpose(x)
    # 2/(F+K)
    deviation1=np.sqrt(2/(x.shape[1]+K))
    #2/(K+C)
    deviation2=np.sqrt(2/(trainingLabels.shape[1]+K))
    #initialize W vectors
    W1=np.random.normal(0,deviation1, size=(x.shape[1],K))
    W2=np.random.normal(0,deviation2, size=(K,trainingLabels.shape[1]))
    #initialize V vectors
    V1=np.ones((x.shape[1],K))
    V2=np.ones((K,trainingLabels.shape[1]))
    V1=V1*1e-5
    V2=V2*1e-5
    # Initialize b the bias vector to zero (1 x 1 array)
    b1 = np.zeros(1)
    b2 = np.zeros(1)

    y = trainingLabels

    i = 0
    train_loss = []
    accuracy = []

    C=trainingLabels.shape[1]

    for i in range(iterations):
        #do forward pass
        s1,x1,s2,prediction,loss=forwardPropogation(x,W1,W2,b1,b2,y)

        # get total loss based on lossType (default is MSE)
        acc = calculateAccuracy(prediction,y)
        accuracy.append(acc)
        # append loss to train_loss for plotting
        train_loss.append(loss)


        #Back back Propogation
        gradWouter,gradBouter,gradwinner,gradbinner=backPropogation(x,x1,prediction,W1,W2,s1,s2,y,K,C)
        print("grad outer W", gradWouter.shape)
        print("grad outer b", gradBouter.shape)
        print("grad inner W", gradwinner.shape)
        print("grad inner b", gradbinner.shape)
        #update value for inner weights
        V1=gamma*V1+(alpha*gradWinner)
        W1=W1-V1

        #update value for inner weights
        V2=gamma*V2+(alpha*gradWouter)
        W2=W2-V2

        #update value for Biases
        b1=b1-alpha*gradbinner
        b2=b2-alpha*gradBouter

    print("Final Loss: ", train_loss[len(train_loss)-1])
    print("Final Accuracy: ",accuracy[len(accuracy)-1],"%")
    return W1,W2,b1,b2,train_loss,accuracy


trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
newtrain, newvalid, newtest= convertOneHot(trainTarget, validTarget, testTarget)
x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
iterations=200
gamma=0.99
K=1000
alpha=0.01
GD(trainData, newtrain, alpha, iterations, gamma,K)
    # TODO
