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
    gradrelu[x<=0]=0.0

    return gradrelu

def softmax(x):

    sume=np.sum(np.exp(x),axis=0)

    return np.exp(x)/sume

def computeLayer(X, W, b):

    return np.dot(np.transpose(W),X)+b

def CE(target, prediction):
    # total cross entropy loss
    cross_entropy_loss = 0
    #yhat is the final prediction in the final layer
    yhat=prediction
    y=target
    N = (target.shape[0])

    innerloop=np.transpose(y)*np.log(yhat)
    # calculate total cross entropy loss
    cross_entropy_loss = np.sum(innerloop)/(-1.0*N)

    return cross_entropy_loss


def gradCE(target, prediction):

    yhat=prediction
    y=target
    N = (target.shape[0])

    #get the gradient of error with respect to s
    return (-1.0/N)*(np.transpose(y)/yhat)


def outergradCE(target,x2,x1,s2,K,C):
    N = (target.shape[0])

    deltaouter=(1/N)*(x2-np.transpose(target))
    #get the gradient with respect to outer weights
    gradWouter=np.dot(x1,np.transpose(deltaouter))

    gradBouter=np.sum(deltaouter,axis=1)

    return deltaouter,gradWouter,gradBouter


def innergradCE(x, deltaouter,w2,s1):

    #get the delta for inner layer
    deltainner=np.multiply(reluDerivative(s1),np.dot(w2,deltaouter))
    #get the gradient with respect to inner layer weight
    gradwinner=np.dot(x,np.transpose(deltainner))
    #get the gradient with respect to inner bias
    gradb=np.sum(deltainner,axis=1)


    return gradwinner,gradb

def forwardPropogation(x,w1,w2,b1,b2,y):
    #compute first layer prediction
    s1=computeLayer(x,w1,np.transpose(b1))

    #use the activation function
    x1=relu(s1)

    #compute the last layer prediction
    s2=computeLayer(x1,w2,np.transpose(b2))

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


def calculateAccuracyPart1(prediction,y):
    predictedclasses=np.argmax(prediction,axis=0)
    t=np.argmax(y,axis=1)

    N=len(predictedclasses)
    count=0.0
    for i in range(N):
        if t[i]==predictedclasses[i]:
            count=count+1.0

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

    #initialize V vectors
    V1=np.ones((x.shape[0],K))
    V2=np.ones((K,trainingLabels.shape[1]))
    V1=V1*1e-5
    V2=V2*1e-5

    # Initialize b the bias vector to zero (1 x 1 array)
    b1 = np.zeros((1,K))
    b2 = np.zeros((1,C))
    Vb1 = np.zeros((1,K))
    Vb2 = np.zeros((1,C))


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

        # append loss to train_loss for plotting
        train_loss.append(loss)
        valid_loss.append(loss2)
        test_loss.append(loss3)
        print(acc,"%")
        print(loss)

        #Back back Propogation
        gradWouter,gradBouter,gradwinner,gradbinner=backPropogation(x,x1,prediction,W1,W2,s1,s2,y,K,C)

        V1=(gamma*V1)+(alpha*gradwinner)

        W1=W1-(V1)

        V2=(gamma*V2)+((alpha)*gradWouter)
        W2=W2-(V2)

        b1=b1.flatten()
        b2=b2.flatten()
        Vb1=Vb1.flatten()
        Vb2=b2.flatten()
        Vb1=(gamma*Vb1)+(alpha*gradbinner)

        Vb2=(gamma*Vb2)+(alpha*gradBouter)
        b2=b2-(Vb2)


        b1 = b1.reshape((1, b1.shape[0]))

        b2 = b2.reshape((1, b2.shape[0]))
        Vb1 = Vb1.reshape((1, Vb1.shape[0]))

        Vb2 = Vb2.reshape((1, Vb2.shape[0]))


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

    print("Final train Loss: ", train_loss[len(train_loss)-1])
    print("Final Train Accuracy: ",train_accuracy[len(train_accuracy)-1],"%")

    print("Final Valid Loss: ", valid_loss[len(valid_loss)-1])
    print("Final Valid Accuracy: ",valid_accuracy[len(valid_accuracy)-1],"%")

    print("Final Test Loss: ", test_loss[len(test_loss)-1])
    print("Final Test Accuracy: ",test_accuracy[len(test_accuracy)-1],"%")
    return W1,W2,b1,b2,train_loss,train_accuracy,valid_loss,valid_accuracy,test_loss,test_accuracy

def part1Main():
    trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
    newtrain, newvalid, newtest= convertOneHot(trainTarget, validTarget, testTarget)
    iterations=200
    gamma=0.9
    K=1000
    alpha=0.1
    W1,W2,b1,b2,train_loss,train_accuracy,valid_loss,valid_accuracy,test_loss,test_accuracy=GD(trainData, newtrain,validData,newvalid,testData,newtest, alpha, iterations, gamma,K)

    index=valid_loss.index(min(valid_loss))
    print(" Validation Accuracy : ",valid_accuracy[index])
    print(" Validation Accuracy interation: ",index)
    print("Minimum Validation Loss: ",(min(valid_loss)))

    print("Test Accuracy : ",test_accuracy[index])
    print("Test Loss : ",test_loss[index])
    print("Train Accuracy : ",train_accuracy[index])
    print("Train Loss : ",train_loss[index])

    plt.close('all')

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plot the loss curve
    X_test = np.linspace(0, len(train_loss), len(train_loss))
    X_test2 = np.linspace(0, len(valid_loss), len(valid_loss))
    X_test3 = np.linspace(0, len(test_loss), len(test_loss))
    plt.title('Cross Entropy Loss for Training, Validation and Testing Data')
    plt.plot(X_test, train_loss, label='Training Data')
    plt.plot(X_test2, valid_loss, label='Validation Data')
    plt.plot(X_test3, test_loss, label='Testing Data')
    plt.legend()

    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    #plot accuracy curve
    X_test4 = np.linspace(0, len(train_accuracy), len(train_accuracy))
    X_test5 = np.linspace(0, len(valid_accuracy), len(valid_accuracy))
    X_test6 = np.linspace(0, len(test_accuracy), len(test_accuracy))
    plt.title('Accuracy curve for Training, Validation and Testing Data')
    plt.plot(X_test4, train_accuracy, label='Training Data')
    plt.plot(X_test5, valid_accuracy, label='Validation Data')
    plt.plot(X_test6, test_accuracy, label='Testing Data')
    plt.legend()


    ####################### Part 1.4###################
    W1,W2,b1,b2,train_loss,train_accuracy,valid_loss,valid_accuracy,test_loss,test_accuracy=GD(trainData, newtrain,validData,newvalid,testData,newtest, alpha, iterations, gamma,100)
    W1,W2,b1,b2,train_loss1,train_accuracy1,valid_loss,valid_accuracy,test_loss1,test_accuracy1=GD(trainData, newtrain,validData,newvalid,testData,newtest, alpha, iterations, gamma,500)
    W1,W2,b1,b2,train_loss2,train_accuracy2,valid_loss,valid_accuracy,test_loss2,test_accuracy2=GD(trainData, newtrain,validData,newvalid,testData,newtest, alpha, iterations, gamma,2000)

    print("Maximum Accuracy for Test K=100: ", max(test_accuracy), "%")
    print("Maximum Accuracy for Test K=500: ", max(test_accuracy1), "%")
    print("Maximum Accuracy for Test K=2000: ", max(test_accuracy2), "%")
    print("Minumum Loss for Test K=100: ", min(test_loss), "%")
    print("Minimum Loss for Test K=500: ", min(test_loss1), "%")
    print("Minimum Loss for Test K=2000: ", min(test_loss2), "%")

    plt.figure(3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plot the loss curve
    X_test11 = np.linspace(0, len(test_loss), len(test_loss))
    X_test12 = np.linspace(0, len(test_loss1), len(test_loss1))
    X_test13 = np.linspace(0, len(test_loss2), len(test_loss2))
    plt.title('Cross Entropy Loss for Testing Data using various units')
    plt.plot(X_test11, test_loss, label='K=100')
    plt.plot(X_test12, test_loss1, label='K=500')
    plt.plot(X_test13, test_loss2, label='K=2000')
    plt.legend()

    plt.figure(4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    #plot accuracy curve
    X_test8 = np.linspace(0, len(test_accuracy), len(test_accuracy))
    X_test9 = np.linspace(0, len(test_accuracy1), len(test_accuracy1))
    X_test10 = np.linspace(0, len(test_accuracy2), len(test_accuracy2))
    plt.title('Accuracy curve for Training Data using various units')
    plt.plot(X_test8, test_accuracy, label='K=100')
    plt.plot(X_test9, test_accuracy1, label='K=500')
    plt.plot(X_test10, test_accuracy2, label='K=2000')
    plt.legend()

    plt.show()


#part1Main()





def calculateAccuracy(prediction,y):

    p=np.argmax(prediction,axis=1)
    t=np.argmax(y,axis=1)

    count=0.0
    N=len(p)
    for i in range(N):
        if t[i] == p[i]:
            count+=1.0

    return count/N * 100.0



def model_implementation(dropout_rate, regularization, var_num):

    tf.set_random_seed(421)

    # create placeholders for x, y and alpha
    x = tf.placeholder(tf.float32, [None, 784], name = "data")
    y = tf.placeholder(tf.float32, [None, 10], name = "labels")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1], name = "reshaped_data")

    conv_output_size = 14 * 14 * 32

    if regularization:
        # 3x3 filter with 1 input channel and 32 filters
        conv_filter = tf.get_variable(shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(regularization), dtype=tf.float32, name="filter_{}".format(var_num))
        conv_bias = tf.get_variable(shape=[32], dtype=tf.float32, name="bias_{}".format(var_num))

        W1 = tf.get_variable(shape=[conv_output_size,784], regularizer=tf.contrib.layers.l2_regularizer(regularization), dtype=tf.float32, name="W1_{}".format(var_num))
        b1 = tf.get_variable(shape=[784], dtype=tf.float32, name="b1_{}".format(var_num))

        W2 = tf.get_variable(shape=[784,10], regularizer=tf.contrib.layers.l2_regularizer(regularization), dtype=tf.float32, name="W2_{}".format(var_num))
        b2 = tf.get_variable(shape=[10], dtype=tf.float32, name="b2_{}".format(var_num))
    else:
        # 3x3 filter with 1 input channel and 32 filters
        conv_filter = tf.get_variable(shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, name="filter_{}".format(var_num))
        conv_bias = tf.get_variable(shape=[32], dtype=tf.float32, name="bias_{}".format(var_num))

        W1 = tf.get_variable(shape=[conv_output_size,784], dtype=tf.float32, name="W1_{}".format(var_num))
        b1 = tf.get_variable(shape=[784], dtype=tf.float32, name="b1_{}".format(var_num))

        W2 = tf.get_variable(shape=[784,10], dtype=tf.float32, name="W2_{}".format(var_num))
        b2 = tf.get_variable(shape=[10], dtype=tf.float32, name="b2_{}".format(var_num))

    # 3x3 convolutional layer, with 32filters, using vertical and horizontal strides of 1
    conv = tf.nn.conv2d(x_reshaped, conv_filter, strides=[1,1,1,1], padding="SAME", name="conv_layer")
    relu_activation = tf.nn.relu(conv + conv_bias)

    # batch normalization layer
    mean, var = tf.nn.moments(relu_activation, axes=[0, 1, 2])
    batch_norm = tf.nn.batch_normalization(relu_activation, mean, var, None, None, 10**(-5))

    # 2x2 max pooling layer
    batch_norm_pool = tf.nn.max_pool(value=batch_norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = "max_pool_layer")

    # flatten layer
    flattened = tf.reshape(batch_norm_pool, [-1, conv_output_size])

    # fully connected layer with relu activiation
    fc_layer1 = tf.matmul(flattened, W1) + b1

    # dropout layer
    if dropout_rate:
        dropout_layer = tf.nn.dropout(fc_layer1, keep_prob = dropout_rate)
        fc_layer1 = dropout_layer

    fc_layer1_relu = tf.nn.relu(fc_layer1)

    # fully connected layer with softmax
    fc_layer2 = tf.matmul(fc_layer1_relu, W2) + b2
    predicted_y = tf.nn.softmax(fc_layer2)

    if regularization:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_y, labels=y))
        total_loss = loss + tf.losses.get_regularization_loss()
    else:
        total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_y, labels=y))

    adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name="Adam").minimize(total_loss)

    return predicted_y, x, y, total_loss, adam_optimizer, learning_rate

def stochastic_gradient_descent(minibatch_size, epochs, lamda, alpha, dropout_rate = None, regularization = None, var_num = 0):

    ''' Uses adam optimizer and the stochastic gradient descent algorithm
    to minimize losses. '''

    # build computaional graph and initialize variables
    predicted_y, x, y, total_loss, adam_optimizer, learning_rate = model_implementation(dropout_rate, regularization, var_num)
    init = tf.global_variables_initializer()

    # get data and labels
    trainData, validData, testData, trainTarget, validTarget, testTarget=loadData()
    newtrain, newvalid, newtest=convertOneHot(trainTarget, validTarget, testTarget)

    # reshape data and labels
    d = trainData.reshape(trainData.shape[0], trainData.shape[1] * trainData.shape[2])
    l = newtrain.reshape(newtrain.shape[0], newtrain.shape[1])
    valid_d = validData.reshape(validData.shape[0], validData.shape[1] * validData.shape[2])
    valid_l = newvalid.reshape(newvalid.shape[0], newvalid.shape[1])
    test_d = testData.reshape(testData.shape[0], testData.shape[1] * testData.shape[2])
    test_l = newtest.reshape(newtest.shape[0], newtest.shape[1])

    # get the number of batches
    num_batches_train = trainData.shape[0] / minibatch_size
    num_batches_valid = validData.shape[0] / minibatch_size
    num_batches_test = testData.shape[0] / minibatch_size

    trainLoss = []
    trainAcc = []
    validLoss = []
    validAcc = []
    testLoss = []
    testAcc = []

    # start tensorflow session
    with tf.Session() as sess:
        sess.run(init)
        # SGD algorithm
        # for each interation loop through all minibatches and run the session
        for i in range(epochs):

            for j in range(0, trainData.shape[0], minibatch_size):
                # get minibatch and run session
                X_batch = d[j:j + minibatch_size]
                Y_batch = l[j:j + minibatch_size]
                _, y_pred, tl = sess.run([adam_optimizer, predicted_y, total_loss],
                feed_dict={x:X_batch, y:Y_batch, learning_rate:alpha})

            # shuffle data and labels
            d, l = shuffle(d, l)

            # get loss and accuracy per iteration
            err = total_loss.eval(feed_dict={x: d, y: l})
            acc = calculateAccuracy(predicted_y.eval(feed_dict={x: d}), l)
            trainLoss.append(err)
            trainAcc.append(acc)
            err = total_loss.eval(feed_dict={x: valid_d, y: valid_l})
            acc = calculateAccuracy(predicted_y.eval(feed_dict={x: valid_d}), valid_l)
            validLoss.append(err)
            validAcc.append(acc)
            err = total_loss.eval(feed_dict={x: test_d, y: test_l})
            acc = calculateAccuracy(predicted_y.eval(feed_dict={x: test_d}), test_l)
            testLoss.append(err)
            testAcc.append(acc)
            print(i)

    # print final error and accuracy
    print("The final training error is ", trainLoss[len(trainLoss)-1])
    print("The final training accuracy is ", trainAcc[len(trainAcc)-1])
    print("The final validation error is ", validLoss[len(validLoss)-1])
    print("The final validation accuracy is ", validAcc[len(validAcc)-1])
    print("The final testing error is ", testLoss[len(testLoss)-1])
    print("The final testing accuracy is ", testAcc[len(testAcc)-1])

    return trainLoss, trainAcc, validLoss, validAcc, testLoss, testAcc

def part_2_1():
    iterations = 50
    batch_size = 32
    alpha = 10**(-4)
    trainLoss, trainAcc, validLoss, validAcc, testLoss, testAcc = stochastic_gradient_descent(batch_size, iterations, 0, alpha)

    plt.close('all')

    #plot the loss curves
    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    X = np.linspace(0, len(trainLoss), len(trainLoss))
    plt.title('Loss for Training, Validation and Testing Data')
    plt.plot(X, trainLoss, label='Training Data')
    plt.plot(X, validLoss, label='Validation Data')
    plt.plot(X, testLoss, label='Testing Data')
    plt.legend()

    #plot accuracy curves
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')

    X = np.linspace(0, len(trainAcc), len(trainAcc))
    plt.title('Accuracy Curve for Training, Validation and Testing Data')
    plt.plot(X, trainAcc, label='Training Data')
    plt.plot(X, validAcc, label='Validation Data')
    plt.plot(X, testAcc, label='Testing Data')
    plt.legend()

    plt.show()

def part_2_2():
    iterations = 50
    batch_size = 32
    alpha = 10**(-4)

    l2_regularization = [0.01, 0.1, 0.5]

    var_num = 0

    for reg in l2_regularization:
        print("REG: ", reg)
        trainLoss, trainAcc, validLoss, validAcc, testLoss, testAcc = stochastic_gradient_descent(batch_size, iterations, 0, alpha, None, reg, var_num)
        var_num += 1

def part_2_3(dropout_rate):
    iterations = 50
    batch_size = 32
    alpha = 10**(-4)

    print("RATE: ", dropout_rate)
    trainLoss, trainAcc, validLoss, validAcc, testLoss, testAcc = stochastic_gradient_descent(batch_size, iterations, 0, alpha, dropout_rate)

    #plot accuracy curves
    plt.close('all')

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')

    X = np.linspace(0, len(trainAcc), len(trainAcc))
    plt.title('Accuracy Curve for Training, Validation and Testing Data for p = {}'.format(dropout_rate))
    plt.plot(X, trainAcc, label='Training Data')
    plt.plot(X, validAcc, label='Validation Data')
    plt.plot(X, testAcc, label='Testing Data')

    plt.legend()
    plt.show()



#part_2_1()

#part_2_2()

#prob = [0.9, 0.75, 0.5]
#for p in prob:
#    part_2_3(p)
