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

    #innerloop=np.dot(np.log(yhat),y)
    innerloop=np.transpose(y)*np.log(yhat)


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
    #print("gradS: ", gradS.shape)
    #get the outer layer gradient
    gradsoftmax=softmaxDerivative(x2,K,C)
    #print("gradsoftmax: ", gradsoftmax.shape)
    #get the delta for outer layer
    deltaouter=np.multiply(gradS,gradsoftmax)
    #print("deltaouter: ", deltaouter.shape)
    #get the gradient with respect to outer weights
    gradWouter=np.dot(x1,np.transpose(deltaouter))
    #print("gradWouter: ", gradWouter.shape)
    #get the gradient with respect to outer bias
    gradBouter=np.sum(deltaouter,axis=1)
    #print("gradBouter: ", gradBouter.shape)


    return deltaouter,gradWouter,gradBouter


def innergradCE(x, deltaouter,w2,s1):

    #get the delta for inner layer
    deltainner=np.multiply(reluDerivative(s1),np.dot(w2,deltaouter))
    #get the gradient with respect to inner layer weight
    gradwinner=np.dot(x,np.transpose(deltainner))
    #print("gradwinner: ",gradwinner.shape)
    #get the gradietn with respect to inner bias
    gradb=np.sum(deltainner,axis=1)
    #print("gradb: ",gradb.shape)

    return gradwinner,gradb

def forwardPropogation(x,w1,w2,b1,b2,y):
    #compute first layer prediction
    s1=computeLayer(x,w1,b1.T)
    #print("S1: WX: ", s1.shape)
    #use the activation function
    x1=relu(s1)
    #print("X1: RElu: ",x1.shape)
    #compute the last layer prediction
    s2=computeLayer(x1,w2,b2.T)
    #print("S2: w2x: ",s2.shape)
    #use the softmax activation funtion
    prediction=softmax(s2)
    #print("prediction: softmax: ",prediction.shape)
    #print(prediction)
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

    predictedclasses=np.argmax(prediction,axis=0)
    #print(predictedclasses.shape)
    #print(predictedclasses)
    count=0.0
    N=(prediction.shape[0])
    for i in range(N):
        if y[i][predictedclasses[i]]==1:
            count+=1.0

    return count/N * 100.0


def GD(trainingData, trainingLabels, alpha, iterations, gamma,K):
    C=trainingLabels.shape[1]
    # reshape x to be a 2D array (number of samples x 784)
    x = x=trainingData.reshape(trainingData.shape[0],(trainingData.shape[1]*trainingData.shape[2]))
    x = np.transpose(x)
    # 2/(F+K)
    deviation1=np.sqrt(2/(x.shape[0]+K))
    #2/(K+C)
    deviation2=np.sqrt(2/(trainingLabels.shape[1]+K))
    #initialize W vectors
    W1=np.random.normal(0,deviation1, size=(x.shape[0],K))
    W2=np.random.normal(0,deviation2, size=(K,trainingLabels.shape[1]))
    #W2=np.random.normal(0,deviation2, size=(x.shape[1],trainingLabels.shape[0]))
    #initialize V vectors
    V1=np.ones((x.shape[0],K))
    V2=np.ones((K,trainingLabels.shape[1]))
    V1=V1*1e-5
    V2=V2*1e-5
    # Initialize b the bias vector to zero (1 x 1 array)
    b1 = np.zeros((1,K))
    b2 = np.zeros((1,C))

    y = trainingLabels

    i = 0
    train_loss = []
    accuracy = []
    #print(x.shape)


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
        #print("grad outer W", gradWouter.shape)
        #print("grad outer b", gradBouter.shape)
        #print("grad inner W", gradwinner.shape)
        #print("grad inner b", gradbinner.shape)
        #update value for inner weights
        V1=(gamma*V1)+(alpha*gradwinner)
        W1=W1-V1

        #update value for inner weights
        V2=(gamma*V2)+(alpha*gradWouter)
        W2=W2-V2

        #update value for Biases
        b1=b1-alpha*gradbinner
        b2=b2-alpha*gradBouter

    print(train_loss)
    print("Final Loss: ", train_loss[len(train_loss)-1])
    print("Final Accuracy: ",accuracy[len(accuracy)-1],"%")
    return W1,W2,b1,b2,train_loss,accuracy

def part1Main():
    trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
    newtrain, newvalid, newtest= convertOneHot(trainTarget, validTarget, testTarget)
    x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    iterations=200
    gamma=0.99
    K=1000
    alpha=0.0001
    W1,W2,b1,b2,train_loss,accuracy=GD(trainData, newtrain, alpha, iterations, gamma,K)
    W1,W2,b1,b2,train_loss1,accuracy1=GD(validData, newvalid, alpha, iterations, gamma,K)
    W1,W2,b1,b2,train_loss2,accuracy2=GD(testData, newtest, alpha, iterations, gamma,K)
    plt.close('all')

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plot the loss curve
    X_test = np.linspace(0, len(train_loss), len(train_loss))
    plt.title('Cross Entropy Loss for Training, Validation and Testing Data')
    plt.plot(X_test, train_loss, label='Training Data')
    plt.plot(X_test, train_loss1, label='Validation Data')
    plt.plot(X_test, train_loss2, label='Testing Data')
    plt.legend()

    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    #plot accuracy curve
    X_test = np.linspace(0, len(accuracy), len(accuracy))
    plt.title('Accuracy curve for Training, Validation and Testing Data')
    plt.plot(X_test, accuracy, label='Training Data')
    plt.plot(X_test, accuracy1, label='Validation Data')
    plt.plot(X_test, accuracy2, label='Testing Data')
    plt.legend()


    ####################### Part 1.4###################
    '''W1,W2,b1,b2,train_loss,accuracy=GD(trainData, newtrain, alpha, iterations, gamma,100)
    W1,W2,b1,b2,train_loss1,accuracy1=GD(trainData, newtrain, alpha, iterations, gamma,500)
    W1,W2,b1,b2,train_loss2,accuracy2=GD(trainData, newtrain, alpha, iterations, gamma,2000)
    plt.close('all')

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plot the loss curve
    X_test = np.linspace(0, len(trainloss), len(trainloss))
    plt.title('Cross Entropy Loss for Training Data using various units')
    plt.plot(X_test, train_loss, label='K=100')
    plt.plot(X_test, train_loss1, label='K=500')
    plt.plot(X_test, train_loss2, label='K=2000')
    plt.legend()

    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    #plot accuracy curve
    X_test = np.linspace(0, len(accuracy), len(accuracy))
    plt.title('Accuracy curve for Training Data using various units')
    plt.plot(X_test, accuracy, label='K=100')
    plt.plot(X_test, accuracy1, label='K=500')
    plt.plot(X_test, accuracy2, label='K=2000')
    plt.legend()'''

    plt.show()

part1Main()
