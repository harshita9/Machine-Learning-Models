import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

'''Assignment 1 by
    Suyasha Acharya (1003083511)
    Sai Harshita Tupili (1002938556)'''

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
    # variable to store total loss (L = LD + LW)
    total_loss = 0

    N = len(x)
    # get yhat (predicted y)
    yhat = np.dot(W.flatten(), x)
    # calculate MSE loss (LD)
    MSEloss = yhat.flatten() - y.flatten() + b
    MSEloss = np.linalg.norm(MSEloss) ** 2
    # calculate weight decay loss (LW)
    weight_decay_loss = (reg / 2) * (np.linalg.norm(W) ** 2)
    # calculate total loss (L)
    total_loss = (MSEloss / (2 * N)) + weight_decay_loss

    return total_loss

def gradMSE(W, b, x, y, reg):
    # Gradient with Respect to Weights and Gradient with Respect to Biases
    gradMSE_weights = 0
    gradMSE_biases = 0

    N = len(x)
    # get yhat (predicted y)
    yhat = np.dot(W.flatten(), x)
    # calculate error
    MSE = (yhat.flatten() - y.flatten() + b)

    # calculate gradient with respect to weights
    gradMSE_weights = 2 * np.dot(MSE, np.transpose(x))
    grad_weight_decay_loss = reg * W
    gradMSE_weights = np.transpose(gradMSE_weights / N) + grad_weight_decay_loss

    # calculate gradient with respect to biases
    gradMSE_biases = np.sum(MSE / N)

    return gradMSE_weights, gradMSE_biases


def crossEntropyLoss(W, b, x, y, reg):
    # total cross entropy loss
    cross_entropy_loss = 0

    N = len(x)

    # get yhat (predicted y) using the sigmoid function
    z = np.dot(W.flatten(), x) + b
    yhat = 1 / (1 + np.exp(-1 * z))

    # get values of inner expressions
    firstexpression=np.dot((np.log(yhat)),(-1*y))
    te = (1-y)
    secondexpression=np.log(1-yhat)
    thirdexpression=np.dot(np.transpose(te),secondexpression)

    fourthexpression=firstexpression.flatten()-thirdexpression.flatten()
    # calculate cross entropy loss
    crossEntropyLoss = (1 / N) * np.sum(fourthexpression)
    # calculate weight decay loss
    weight_decay_loss = (reg / 2) * (np.linalg.norm(W) ** 2)
    # calculate total cross entropy loss
    cross_entropy_loss = crossEntropyLoss + weight_decay_loss

    return cross_entropy_loss


def gradCE(W, b, x, y, reg):
        # Gradient with Respect to Weights and Gradient with Respect to Biases
        gradCE_weight, gradCE_bias = 0, 0

        N = len(x)

        # get yhat (predicted y) using the sigmoid function
        z = np.dot(W.flatten(), x) + b
        yhat = 1 / (1 + np.exp(-1 * z))

        # calculate gradient with respect to weights
        grad_weight_decay_loss = reg * W

        gradCE_weight = 1/N*(np.dot(x,yhat.flatten()-y.flatten()))

        gradCE_weight = gradCE_weight.flatten()+grad_weight_decay_loss.flatten()

        # calculate gradient with respect to biases
        gradCE_bias = np.sum(yhat.flatten() - y.flatten()) / N

        return gradCE_weight, gradCE_bias

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType):
    # Initialize W the weight vector to zeros (784 x 1 array)
    W = np.zeros(trainingData.shape[1] * trainingData.shape[2])
    # Initialize b the bias vector to zero (1 x 1 array)
    b = np.zeros(1)
    # reshape x to be a 2D array (number of samples x 784)
    x = trainingData.reshape(trainingData.shape[0], (trainingData.shape[1]*trainingData.shape[2]))
    x = np.transpose(x)

    y = trainingLabels

    i = 0
    train_loss = []
    accuracy = []

    for i in range(iterations):
        # get total loss based on lossType (default is MSE)
        if lossType == "MSE":
            loss = MSE(W,b,x,y,reg)
        elif lossType == "CE":
            loss = crossEntropyLoss(W,b,x,y,reg)
            acc = calculateAccuracy(W,b,x,y)
            accuracy.append(acc)
        else:
            loss = MSE(W,b,x,y,reg)
        # append loss to train_loss for plotting
        train_loss.append(loss)

        # get gradient with respect to weight and bias
        if lossType == "MSE":
            weight_gradient, bias_gradient = gradMSE(W,b,x,y,reg)
        elif lossType == "CE":
            weight_gradient, bias_gradient = gradCE(W,b,x,y,reg)

        else:
            weight_gradient, bias_gradient = gradMSE(W,b,x,y,reg)

        # Calulate optimal weight an bias

        # Calculate the direction of the gradient of weight vector
        norm_weight_grad = np.linalg.norm(weight_gradient)
        weight_direction = -1 * weight_gradient / norm_weight_grad
        # Calculate the direction of the gradient of bias vector
        norm_bias_grad = np.linalg.norm(bias_gradient)
        bias_direction = -1 * bias_gradient / norm_bias_grad
        # Calculate the new weight and bias vector
        new_w = W + alpha * weight_direction
        new_b = b + alpha * bias_direction
        # weight error
        difference = np.linalg.norm(new_w - W)
        # checking if new_w (new weight) is minimum by checking the gradient
        if(difference < EPS):
            # minimum/final weight array found
            break
        else:
            W = new_w
            b = new_b

    return W,b,train_loss,accuracy

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    '''Build computational graph for SGD.'''
    #Initialize weight and bias tensors
    W = tf.Variable(tf.truncated_normal(shape = (1,784), stddev = 0.5, dtype = tf.float32, name="weights"))
    b = tf.Variable(tf.zeros(1), name="biases")

    # create placeholders for x, y, reg, and alpha
    x = tf.placeholder(tf.float32, [784, None], name="data")
    y = tf.placeholder(tf.float32, [None, 1], name="labels")
    reg = tf.placeholder(tf.float32, name="reg")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    tf.set_random_seed(421)

    # calculate predicted y
    predicted_y = tf.matmul((W), x) + b
    # calculate weight_decay_loss
    wd = (reg/2) * tf.reduce_sum(tf.square(W), name="weight_decay_loss")

    total_loss = 0
    loss = 0

    # calculate loss based on loss type
    if lossType == "MSE":
        loss = tf.losses.mean_squared_error(labels=y, predictions=predicted_y)
        loss = 1/2*(loss)

    elif lossType == "CE":
        yhat = tf.sigmoid(predicted_y)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predicted_y)

    # calculate total loss
    total_loss = loss + wd

    # use adam optimizer for the given hyperparameters
    if beta1:
        adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=beta1, name="Adam").minimize(total_loss)
    elif beta2:
        adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta2=beta2, name="Adam").minimize(total_loss)
    elif epsilon:
        adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon=epsilon, name="Adam").minimize(total_loss)
    else:
        adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name="Adam").minimize(total_loss)

    return W, b, predicted_y, x, y, total_loss, adam_optimizer, reg, learning_rate

def calculateAccuracy(W,b,x,y):
    '''Calculates accuracy of the predicted labels using given weights and biases.'''
    yhat = np.dot(W.flatten(),x) + b
    yhat = yhat.flatten()
    y = y.flatten()

    #number of accurate data classified
    correct = 0

    for i in range(0,len(y)):
        if((yhat[i]<0 and y[i]==0) or (yhat[i]>=0 and y[i]==1)):
            correct=correct+1;

    return float(correct/len(y))*100

def stochastic_gradient_descent(minibatch_size, epochs, lamda, data, labels, loss_type, alpha, b1 = None, b2 = None, e = None):

    ''' Uses adam optimizer and the stochastic gradient descent algorithm
    to compute optimal losses. '''

    # build computaional graph and initialize variables
    W, b, predicted_y, x, y, total_loss, adam_optimizer, reg, learning_rate = buildGraph(lossType = loss_type, beta1 = b1, beta2 = b2, epsilon = e)
    init = tf.global_variables_initializer()

    # reshape data
    d = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    l = labels.reshape(len(labels),1)

    # get the number of batches
    num_batches = data.shape[0] / minibatch_size

    losses = []
    acc = []
    # start tensor flow session
    with tf.Session() as sess:
        sess.run(init)
        # SGD algorithm
        # for each interation loop through all minibatches and run the session
        losses = []
        acc = []
        for i in range(epochs):
            a = []
            epochLoss = []
            # shuffle data and labels
            index_shuffle = [m for m in range(len(l))]
            shuffle(index_shuffle)
            d  = d[index_shuffle, :]
            l = l[index_shuffle,]
            for j in range(0,data.shape[0],minibatch_size):
                # get minibatch and run session
                X_batch = d[j:j + minibatch_size]
                Y_batch = l[j:j + minibatch_size]
                X_batch = np.transpose(X_batch)
                _, W_new, b_new, tl = sess.run([adam_optimizer, W, b, total_loss],
                feed_dict={x:X_batch,reg:lamda, y:Y_batch, learning_rate:alpha})
                # add loss and acuuracy for each batch
                epochLoss.append(tl)
                a.append(calculateAccuracy(W_new,b_new,X_batch,Y_batch))
            # get average loss per iteration
            losses.append(np.mean(epochLoss))
            acc.append(np.mean(a))

    #import pdb; pdb.set_trace() # for debugging

    # print final error and accuracy
    print("The final error is ", losses[len(losses)-1])
    print("The final accuracy is ", acc[len(acc)-1])

    return losses, acc

def mainSGD(parameter,n=1,n2=2):
    '''Plotting graphs for part 3'''
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    alpha = 0.001
    losses, acc = stochastic_gradient_descent(100, 700, 0, trainData, trainTarget, parameter, alpha)
    losses1, acc1 = stochastic_gradient_descent(700, 700, 0, trainData, trainTarget, parameter, alpha)
    losses2, acc2 = stochastic_gradient_descent(1750, 700, 0, trainData, trainTarget, parameter, alpha)

    X_test = np.linspace(0, len(losses), len(losses))
    X_test1 = np.linspace(0, len(losses1), len(losses1))
    X_test2 = np.linspace(0, len(losses2), len(losses2))

    Y_test = np.linspace(0, len(acc), len(acc))
    Y_test1 = np.linspace(0, len(acc1), len(acc1))
    Y_test2 = np.linspace(0, len(acc2), len(acc2))

    plt.figure(n)
    plt.plot(X_test, losses, label='Batch=100')
    plt.plot(X_test1, losses1, label='Batch=700')
    plt.plot(X_test2, losses2, label='Batch=1750')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SGD using Training Data for '+parameter)

    plt.legend()

    plt.figure(n2)
    plt.plot(Y_test, acc, label='Batch=100')
    plt.plot(Y_test1, acc1, label='Batch=700')
    plt.plot(Y_test2, acc2, label='Batch=1750')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('SGD using Training Data for '+parameter)
    plt.legend()

def hyperparameters(data, labels):
    '''Hyperparameters investigation for part 3'''
    alpha=0.001
    print ("MSE")
    print ("beta1 = 0.95")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "MSE",alpha=alpha, b1 = 0.95)
    print ("beta1 = 0.99")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "MSE",alpha=alpha, b1 = 0.99)
    print ("beta2 = 0.99")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "MSE",alpha=alpha, b2 = 0.99)
    print ("beta2 = 0.9999")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "MSE",alpha=alpha, b2 = 0.9999)
    print ("epsilon = 1e-09")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "MSE",alpha=alpha, e = 1e-09)
    print ("epsilon = 1e-4")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "MSE",alpha=alpha, e = 1e-4)

    print ("CE")
    print ("beta1 = 0.95")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "CE", alpha=alpha, b1 = 0.95)
    print ("beta1 = 0.99")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "CE", alpha=alpha, b1 = 0.99)
    print ("beta2 = 0.99")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "CE", alpha=alpha, b2 = 0.99)
    print ("beta2 = 0.9999")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "CE", alpha=alpha, b2 = 0.9999)
    print ("epsilon = 1e-09")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "CE", alpha=alpha, e = 1e-09)
    print ("epsilon = 1e-4")
    losses, acc = stochastic_gradient_descent(500, 700, 0, data, labels, "CE", alpha=alpha, e = 1e-4)

    return


def normalMSE(x,y,reg):
    '''Normal MSE equation for linear regression'''
    inversexx = np.linalg.inv(np.dot(np.transpose(x),x)+reg*np.identity(x.shape[1]))
    w = np.dot(np.dot(inversexx,np.transpose(x)).T,y)
    return w


def plotlinearRegression(Data, Target,alpha,alpha1,alpha2,iterations,reg1,reg2,reg3,EPS,parameter, lossType):
    '''Plots for linear regression'''
    #initialize W and bias
    W = np.zeros(Data.shape[1] * Data.shape[2])
    W1 = np.zeros(Data.shape[1] * Data.shape[2])
    W2 = np.zeros(Data.shape[1] * Data.shape[2])

    b=np.zeros(1)

    #Get the x and y
    x=Data.reshape(Data.shape[0],(Data.shape[1]*Data.shape[2]))
    x=np.transpose(x)
    y=Target

    #Get the optimized weight, bias and the loss
    W, b,trainloss,acc = grad_descent(W, b, Data, Target, alpha, iterations, reg1, EPS, "MSE")
    print('MSE loss 1: ', trainloss[len(trainloss)-1])

    #Get accuracy for each
    accuracy= calculateAccuracy(W,b,x,y)
    print('accuracy 1: ',accuracy, '%')

    W1, b,trainloss2,acc = grad_descent(W1, b, Data, Target, alpha1, iterations, reg2, EPS,"MSE")
    print('MSE loss 2: ', trainloss2[len(trainloss2)-1])
    #Get accuracy for each
    accuracy= calculateAccuracy(W1,b,x,y)
    print('accuracy 2: ',accuracy, '%')

    W2, b,trainloss3,acc = grad_descent(W2, b, Data, Target, alpha2, iterations, reg3, EPS,"MSE")
    print('MSE loss 3: ', trainloss3[len(trainloss3)-1])

    #Get accuracy for each
    accuracy= calculateAccuracy(W2,b,x,y)
    print('accuracy 3: ',accuracy,'%')


    #plotting
    X_test = np.linspace(0, len(trainloss), len(trainloss))
    X_test2 = np.linspace(0, len(trainloss2), len(trainloss2))
    X_test3 = np.linspace(0, len(trainloss3), len(trainloss3))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MSE with different '+ parameter)

    #plot the graph
    if(parameter=="alpha"):
        plt.plot(X_test, trainloss, label='alpha=0.005')
        plt.plot(X_test2, trainloss2, label='alpha=0.001')
        plt.plot(X_test3, trainloss3, label='alpha=0.0001')
    else:
        plt.plot(X_test, trainloss, label='reg=0.001')
        plt.plot(X_test2, trainloss2, label='reg=0.1')
        plt.plot(X_test3, trainloss3, label='reg=0.5')

    plt.legend()
    return

def Part1and2():
    '''Main function for part one and two (linear and logistic regression)'''
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    W = np.zeros(trainData.shape[1] * trainData.shape[2])

    b = np.zeros(1)


    iterations = 5000
    reg = 0

    EPS = 1 * 10 ** (-7)


    plt.close('all')

#############################LINEAR REGRESSION ###################################

    #Tuning the Learning Rate Plot, Plot losses
    alpha = 0.005
    alpha1 = 0.001
    alpha2 = 0.0001
    print('Learning Rate')
    plt.figure(1)
    #Training losses
    plotlinearRegression(trainData, trainTarget,alpha,alpha1,alpha2,iterations,reg,reg,reg,EPS,"alpha", "MSE")
    #Validation losses
    plt.figure(2)
    plotlinearRegression(validData, validTarget,alpha,alpha1,alpha2,iterations,reg,reg,reg,EPS, "alpha", "MSE")
    #Testing Losses
    plt.figure(3)
    plotlinearRegression(testData, testTarget,alpha,alpha1,alpha2,iterations,reg,reg,reg,EPS, "alpha", "MSE")

    #Generalization
    reg1=0.001
    reg2= 0.1
    reg3= 0.5
    print('Regularization')
    plt.figure(4)
    #Training losses
    plotlinearRegression(trainData, trainTarget,alpha,alpha,alpha,iterations,reg1,reg2,reg3,EPS, "regularization parameter", "MSE")
    #Validation losses
    plt.figure(5)
    plotlinearRegression(validData, validTarget,alpha,alpha,alpha,iterations,reg1,reg2,reg3,EPS, "regularization parameter", "MSE")
    #Testing Losses
    plt.figure(6)
    plotlinearRegression(testData, testTarget,alpha,alpha,alpha,iterations,reg1,reg2,reg3,EPS, "regularization parameter", "MSE")


    #Comparing Batch GD with normal equation
    startBatched=time.time()
    W, b,trainloss,acc = grad_descent(W, b, trainData, trainTarget, alpha1, iterations, reg1, EPS, "MSE")
    print('computation time batched GD: ',time.time()-startBatched)

    x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    x=np.transpose(x)
    y=trainTarget

    #Calcualate error for batch GD
    loss_batched=MSE(W,b,x,y,reg1)
    #Calcualate accuracy
    accuracy= accuracy= calculateAccuracy(W,b,x,y)
    print('accuracy batch: ',accuracy)

    #calculate normal equation
    startNormal=time.time()
    W2=normalMSE(x,y,reg1)
    print('computation time Normal: ',time.time()-startNormal)

    loss=MSE(W2,0,x,y,reg1)

    #Calcualate accuracy
    accuracy= accuracy= calculateAccuracy(W2,0,x,y)
    print('accuracy normal: ',accuracy)

    print('loss batched: ',loss_batched)

    print('loss normal: ',loss)

############################# LOGISTIC REGRESSION ###################################

    #Tuning the Learning Rate Plot, Plot losses (logistic regression)
    reg = 0.1
    alpha = 0.005
    alpha1 = 0.001
    alpha2 = 0.0001
    print('Learning Rate')



    #W3 = np.zeros(Data.shape[1] * Data.shape[2])
    x = validData.reshape(validData.shape[0],(validData.shape[1]*validData.shape[2]))
    x = np.transpose(x)
    y = validTarget
    W = np.zeros((validData.shape[1] * validData.shape[2]))

    b = np.zeros(1)


    #Get the optimized weight, bias and the loss
    W, b, trainloss, acc = grad_descent(W, b, validData, validTarget, alpha, iterations, reg, EPS, "CE")
    print('CE loss 1: ',crossEntropyLoss(W, b, x, y, reg))
    #Get accuracy for each
    accuracy = calculateAccuracy(W,b,x,y)
    print('accuracy 1: ',accuracy, '%')

    W, b, trainloss1, acc1 = grad_descent(W, b, validData, validTarget, alpha1, iterations, reg, EPS, "CE")
    print('CE loss 2: ', crossEntropyLoss(W, b, x, y, reg))
    #Get accuracy for each
    accuracy = calculateAccuracy(W,b,x,y)
    print('accuracy 2: ',accuracy, '%')

    W, b, trainloss2, acc2 = grad_descent(W, b, validData, validTarget, alpha2, iterations, reg, EPS, "CE")
    print('CE loss 3: ', crossEntropyLoss(W, b, x, y, reg))
    #Get accuracy for each
    accuracy = calculateAccuracy(W,b,x,y)
    print('accuracy 3: ',accuracy, '%')

    #plotting
    X_test = np.linspace(0, len(trainloss), len(trainloss))
    X_test2 = np.linspace(0, len(acc), len(acc))
    X_test3 = np.linspace(0, len(trainloss1), len(trainloss1))
    X_test4 = np.linspace(0, len(acc1), len(acc1))
    X_test5 = np.linspace(0, len(trainloss2), len(trainloss2))
    X_test6 = np.linspace(0, len(acc2), len(acc2))

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title('CE loss for Validation Data with different learning rate')
    plt.plot(X_test, trainloss, label='alpha=0.005')
    plt.plot(X_test3, trainloss1, label='alpha=0.001')
    plt.plot(X_test5, trainloss2, label='alpha=0.0001')
    plt.legend()


    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CE accuracy for Validation Data with different learning rate')
    plt.plot(X_test2, acc, label='alpha=0.005')
    plt.plot(X_test3, acc1, label='alpha=0.001')
    plt.plot(X_test6, acc2, label='alpha=0.0001')
    #plt.plot(X_test3, trainloss3, label='alpha=0.0001')

    plt.legend()



    #Comparison to Linear regression
    x = trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    x = np.transpose(x)
    y = trainTarget
    W = np.zeros((trainData.shape[1] * trainData.shape[2]))

    b = np.zeros(1)

    plt.figure(3)
    W, b, trainloss, acc = grad_descent(W, b, trainData, trainTarget, 0.005, iterations, 0, EPS, "MSE")
    print('MSE loss 1: ', trainloss[len(trainloss)-1])
    #Get accuracy for each
    accuracy = calculateAccuracy(W,b,x,y)
    print('accuracy 1: ',accuracy, '%')

    X_test = np.linspace(0, len(trainloss), len(trainloss))

    plt.plot(X_test, trainloss, label='Linear Regression')

    #Logistic Regression
    W, b, trainloss, acc = grad_descent(W, b, trainData, trainTarget, 0.005, iterations, 0, EPS, "CE")
    print('CE loss 1: ', trainloss[len(trainloss)-1])
    #Get accuracy for each
    accuracy = calculateAccuracy(W,b,x,y)
    print('accuracy 1: ',accuracy, '%')

    X_test = np.linspace(0, len(trainloss), len(trainloss))


    plt.plot(X_test, trainloss, label='Logistic Regression')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Linear Regression Vs. Logistic Regression losses')
#
    plt.legend()
    plt.show()

def softmaxDerivative(x2,C,s2,y):
    softmaxder=np.zeros(shape=x2.shape)
    #return x2*(1-x2)
    N=x2.shape[1]
    '''for n in range(x2.shape[1]):
        for i in range(C):
            for j in range(C):
                if i==j:
                    softmaxder[i,n]=x2[j,n] *(1.0-x2[j,n])
                else:
                    softmaxder[j,n]=-1.0*x2[i,n] * x2[j,n]'''

    for i in range(x2.shape[1]):
        for j in range(C):

            if j==np.argmax(y[i]):
                softmaxder[j,i]=x2[j,i]*(1-x2[j,i])
            else:
                softmaxder[j,i]= -1*x2[j,i] *x2[np.argmax(y[i]),i]

    return softmaxder
#trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

# For part one and two call commented function below
#Part1and2()

# For part 3 call commented functions below
#print("train")
#hyperparameters(trainData, trainTarget)
#print("valid")
#hyperparameters(validData, validTarget)
#print("test")
#hyperparameters(testData, testTarget)
#mainSGD("MSE")
#mainSGD("CE", 3, 4) # 3 and 4 are figure numbers for plotting
