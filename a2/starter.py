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
    #print(x)
    gradrelu=np.ones((x.shape[0],x.shape[1]))
    gradrelu[x<=0]=0
    #print("*******************")
    #print(gradrelu)
    return gradrelu

def softmax(x):

    sume=np.sum(np.exp(x),axis=0)

    return np.exp(x)/sume

def softmaxDerivative(x2,C):
    softmaxder=np.zeros(shape=x2.shape)


    for n in range(x2.shape[1]):
        for i in range(C):
            for j in range(C):
                if i==j:
                    softmaxder[i][n]=x2[j][n] *(1-x2[j][n])
                else:
                    softmaxder[j][n]=-1*x2[i][n] * x2[j][n]

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
    gradsoftmax=softmaxDerivative(x2,C)
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
    #get the gradient with respect to inner bias
    gradb=np.sum(deltainner,axis=1)
    #print("gradb: ",gradb.shape)

    return gradwinner,gradb

def forwardPropogation(x,w1,w2,b1,b2,y):
    #compute first layer prediction
    s1=computeLayer(x,w1,np.transpose(b1))
    #print("S1: WX: ", s1.shape)
    #use the activation function
    x1=relu(s1)
    #print("X1: RElu: ",x1.shape)
    #compute the last layer prediction
    s2=computeLayer(x1,w2,np.transpose(b2))
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


def calculateAccuracyPart1(prediction,y):
    predictedclasses=np.argmax(prediction,axis=0)
    t=np.argmax(y,axis=1)

    count=0.0
    N=len(predictedclasses)
    for i in range(0,N):
        if t[i]==predictedclasses[i]:
            count+=1.0

    return count/float(N) * 100.0


def GD(trainingData, trainingLabels,ValidatinData,ValidationLabels,TestingData,TestingLabels, alpha, iterations, gamma,K):
    C=trainingLabels.shape[1]
    # reshape x to be a 2D array (number of samples x 784)
    x =trainingData.reshape(trainingData.shape[0],(trainingData.shape[1]*trainingData.shape[2]))
    x = np.transpose(x)
    x_valid =ValidatinData.reshape(ValidatinData.shape[0],(ValidatinData.shape[1]*ValidatinData.shape[2]))
    x_valid = np.transpose(x_valid)
    x_test =TestingData.reshape(TestingData.shape[0],(TestingData.shape[1]*TestingData.shape[2]))
    x_test = np.transpose(x_test)
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
    y_valid=ValidationLabels
    y_test=TestingLabels

    i = 0
    train_loss = []
    valid_loss = []
    test_loss = []
    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    #print(x.shape)


    for i in range(iterations):
        #do forward pass
        s1,x1,s2,prediction,loss=forwardPropogation(x,W1,W2,b1,b2,y)
        s_v1,x_v1,s_v2,prediction2,loss2=forwardPropogation(x_valid,W1,W2,b1,b2,y_valid)
        s_t1,x_t1,s_t2,prediction3,loss3=forwardPropogation(x_test,W1,W2,b1,b2,y_test)

        # get total loss based on lossType (default is MSE)
        acc = calculateAccuracyPart1(prediction,y)
        acc2 = calculateAccuracyPart1(prediction2,y_valid)
        acc3 = calculateAccuracyPart1(prediction3,y_test)
        train_accuracy.append(acc)
        valid_accuracy.append(acc2)
        test_accuracy.append(acc3)
        print("training")
        print(loss)
        print(acc,"%")
        print("validation")
        print(loss)
        print(acc,"%")
        print("testing")
        print(loss)
        print(acc,"%")
        # append loss to train_loss for plotting
        train_loss.append(loss)
        valid_loss.append(loss2)
        test_loss.append(loss3)


        #Back back Propogation
        gradWouter,gradBouter,gradwinner,gradbinner=backPropogation(x,x1,prediction,W1,W2,s1,s2,y,K,C)
        #print("grad outer W", gradWouter.shape)
        #print("grad outer b", gradBouter.shape)
        #print("grad inner W", gradwinner.shape)
        #print("grad inner b", gradbinner.shape)
        #update value for inner weights
        #norm_weight_1 = np.linalg.norm(gradwinner)
        #weight_direction1 =  gradwinner / norm_weight_1
        V1=(gamma*V1)+(alpha*gradwinner)
        #print(V1)
        W1=W1-V1
        #print(W1)
        #update value for inner weights
        #norm_weight_2 = np.linalg.norm(gradWouter)
        #weight_direction2 =  gradWouter / norm_weight_2
        V2=(gamma*V2)+(alpha*gradWouter)
        W2=W2-V2
        #print(W2)

        #update value for Biases inner
        #norm_bias_grad1 = np.linalg.norm(gradbinner)
        #bias_direction1 = gradbinner / norm_bias_grad1
        b1 = b1.flatten()
        b2 = b2.flatten()
        b1=b1-alpha*gradbinner

        #print(b1)
        #update value for biases outer
        #norm_bias_grad2 = np.linalg.norm(gradBouter)
        #bias_direction2 = gradBouter / norm_bias_grad2

        b2=b2-alpha*gradBouter

        b1 = b1.reshape((1, b1.shape[0]))

        b2 = b2.reshape((1, b2.shape[0]))
        #print(b2)

    s1,x1,s2,prediction,loss=forwardPropogation(x,W1,W2,b1,b2,y)
    s_v1,x_v1,s_v2,prediction2,loss2=forwardPropogation(x_valid,W1,W2,b1,b2,y_valid)
    s_t1,x_t1,s_t2,prediction3,loss3=forwardPropogation(x_test,W1,W2,b1,b2,y_test)

    # get total loss based on lossType (default is MSE)
    acc = calculateAccuracyPart1(prediction,y)
    acc2 = calculateAccuracyPart1(prediction2,y_valid)
    acc3 = calculateAccuracyPart1(prediction3,y_test)
    train_accuracy.append(acc)
    valid_accuracy.append(acc2)
    test_accuracy.append(acc3)
    # append loss to loss array for plotting
    train_loss.append(loss)
    valid_loss.append(loss2)
    test_loss.append(loss3)
    #print(train_loss)
    #print("****************************")
    #print(accuracy)
    print("Final Loss: ", train_loss[len(train_loss)-1])
    print("Final Accuracy: ",accuracy[len(accuracy)-1],"%")
    return W1,W2,b1,b2,train_loss,train_accuracy,valid_loss,valid_accuracy,test_loss,test_accuracy

def part1Main():
    trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
    newtrain, newvalid, newtest= convertOneHot(trainTarget, validTarget, testTarget)
    x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    iterations=200
    gamma=0.99
    K=1000
    alpha=0.0001
    W1,W2,b1,b2,train_loss,train_accuracy,valid_loss,valid_accuracy,test_loss,test_accuracy=GD(trainData, newtrain,validData,newvalid,testData,newtest, alpha, iterations, gamma,K)
    #W1,W2,b1,b2,train_loss1,accuracy1=GD(validData, newvalid, alpha, iterations, gamma,K)
    #W1,W2,b1,b2,train_loss2,accuracy2=GD(testData, newtest, alpha, iterations, gamma,K)
    plt.close('all')

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plot the loss curve
    X_test = np.linspace(0, len(train_loss), len(train_loss))
    plt.title('Cross Entropy Loss for Training, Validation and Testing Data')
    plt.plot(X_test, train_loss, label='Training Data')
    plt.plot(X_test, valid_loss, label='Validation Data')
    plt.plot(X_test, test_loss, label='Testing Data')
    plt.legend()

    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    #plot accuracy curve
    X_test = np.linspace(0, len(accuracy), len(accuracy))
    plt.title('Accuracy curve for Training, Validation and Testing Data')
    plt.plot(X_test, train_accuracy, label='Training Data')
    plt.plot(X_test, valid_accuracy, label='Validation Data')
    plt.plot(X_test, test_accuracy, label='Testing Data')
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












'''
def model_implementation(classes):



    tf.set_random_seed(421)



    # create placeholders for x, y, alpha and reg

    x = tf.placeholder(tf.float32, [784, None], name = "data")

    y = tf.placeholder(tf.float32, [None, classes], name = "labels")

    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    reg = tf.placeholder(tf.float32, name="reg")

    #x_reshaped = tf.reshape(x, [-1, 28, 28, 1], name = "reshaped_data")



    # 3x3 filter with 1 input channel and 32 filters

    conv_filter = tf.get_variable(shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, name="filter")

    conv_bias = tf.get_variable(shape=[32], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, name="conv_bias")



    W2 = tf.get_variable("Wf2", shape=[14*14*32,32],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)

    b2 = tf.get_variable("bf2", shape=[32],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)



    W3 = tf.get_variable("Wf3", shape=[32,classes_out],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)

    b3 = tf.get_variable("bf3", shape=[classes_out],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)





    conv = tf.nn.conv2d(x, conv_filter, strides=[1,1,1,1], padding="SAME", name="cnn_layer")

    #conv_with_bias = tf.nn.bias_add(conv, conv_bias)

    relu_activation = tf.nn.relu(conv + conv_bias)



    mean, var = tf.nn.moments(relu_activation)

    batch_norm = tf.nn.batch_normalization(conv1, mean, var)



    # 2x2 max pool

    batch_norm_pool = tf.nn.max_pool(value=batch_norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = "max_pool_layer")



    #flatten the output

    flattened = tf.reshape(max_pooling_layer, [-1, tf.shape(max_pooling_layer)[0]])



    # weights and bias values for fc layer

    wd1 = tf.get_variable(initializer=xavier_initializer, shape=[14 * 14 * 32, 784], name='wd1')

    bd1 = tf.get_variable(initializer=xavier_initializer, shape=784, name='bd1')

    dense_layer1 = tf.matmul(flattened, wd1) + bd1

    dense_layer1 = tf.nn.relu(dense_layer1)



    #setup some weights and bias values for fully connected and softmax layer, then activate with ReLU

    wd2 = tf.get_variable(initializer=xavier_initializer, shape=[tf.shape(dense_layer1),10], name='wd2')

    bd2 = tf.get_variable(initializer=xavier_initializer, shape=[10], name='bd2')

    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2

    y_pred = tf.argmax(input=dense_layer2, axis=1)



    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))



    adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name="Adam").minimize(total_loss)



    return predicted_y, x, y, total_loss, adam_optimizer, reg, learning_rate


'''
