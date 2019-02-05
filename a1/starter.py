import tensorflow as tf
import time
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
    gradMSE_weights = np.dot(MSE, np.transpose(x))
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
    firstexpression = -1 * np.dot(np.dot(y, np.log(yhat)),x)
    secondexpression = np.dot((1 - y), np.log(1 - np.dot(yhat,x)))

    # calculate cross entropy loss
    crossEntropyLoss = (1 / N) * np.sum((firstexpression - secondexpression))
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
        # get values of inner expressions
        firstexpression = -1 * np.dot(y, 1 - yhat)
        secondexpression = np.dot((1 - y), yhat)
        # calculate gradient with respect to weights
        grad_weight_decay_loss = reg * W
        gradCE_weight = np.dot(firstexpression + secondexpression, x) / N + grad_weight_decay_loss
        # calculate gradient with respect to biases
        gradCE_bias = (firstexpression + secondexpression) / N

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
    acc=[]

    for i in range(iterations):
        # get total loss based on lossType (default is MSE)
        if lossType == "MSE":
            loss = MSE(W,b,x,y,reg)
        elif lossType == "CE":
            loss = crossEntropyLoss(W,b,x,y,reg)
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
        difference = np.linalg.norm(new_w - W) ** 2
        # checking if new_w (new weight) is minimum
        if(difference < EPS):
            # minimum/final weight array found
            break
        else:
            W = new_w
            b = new_b



    return W,b,train_loss,acc

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    #Initialize weight and bias tensors
    #W = tf.tructated_normal([784,1],0.5)
    W = tf.Variable(tf.zeros[28 * 28, 1], name="weights")
    b = tf.Variable(tf.zeros[1], name="biases")

    x = tf.placeholder(tf.float32, [None, 784], name="data")
    #x = tf.reshape(x, [None, 28 * 28])
    y = tf.placeholder(tf.float32,[None,1], name="labels")
    predicted_y = tf.placeholder(tf.float32,[None,1], name="predicted_labels")
    reg = tf.placeholder(tf.float32, name="reg")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')


    tf.set_random_seed(421)


    total_loss = 0

    if loss == "MSE":
        predicted_y = tf.matmul(x, W) + b
        error= predicted_y-y
        mse = 1/2*tf.reduce_mean(tf.square(error),name="mse")
        wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)), name="weight_decay_loss")

        total_loss = mse + wd

    elif loss == "CE":
        predicted_y = tf.matmul(tf.transpose(W),x) + b
        yhat = tf.sigmoid(predicted_y)
        ce = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(labels=y, logits=predicted_y), name="cross_entropy_loss")
        wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)), name="weight_decay_loss")

        total_loss = ce + wd



    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    '''with tf.Session() as sess:
        sess.run(init)

        for epoch in range)'''

    return W,b,predicted_y,y,total_loss,reg


def normalMSE(x,y,reg):

    inversexx = np.linalg.inv(np.dot(np.transpose(x),x)+reg*np.identity(x.shape[1]))
    w=np.dot(np.dot(inversexx,np.transpose(x)).T,y)
    return w


def calculateAccuracy(W,b,x,y):

    yhat=np.dot(W.flatten(),x)+ b
    yhat=yhat.flatten()
    y=y.flatten()

    #number of accurate data classified
    correct=0

    for i in range(0,len(y)):
        if((yhat[i]<=0 and y[i]==0) or (yhat[i]>0 and y[i]==1)):
            correct=correct+1;

    return float(correct/len(y))*100



def plotlinearRegression(Data, Target,alpha,alpha1,alpha2,iterations,reg1,reg2,reg3,EPS,parameter):
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

#if __name__ == "__main__":
def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    W = np.zeros(trainData.shape[1] * trainData.shape[2])

    b = np.zeros(1)


    iterations = 5000
    reg = 0

    EPS = 1 * 10 ** (-7)


    plt.close('all')
    #Tuning the Learning Rate Plot, Plot losses
    alpha = 0.005
    alpha1 = 0.001
    alpha2 = 0.0001
    print('Learning Rate')
    plt.figure(1)
    #Training losses
    plotlinearRegression(trainData, trainTarget,alpha,alpha1,alpha2,iterations,reg,reg,reg,EPS,"alpha")
    #Validation losses
    plt.figure(2)
    plotlinearRegression(validData, validTarget,alpha,alpha1,alpha2,iterations,reg,reg,reg,EPS, "alpha")
    #Testing Losses
    plt.figure(3)
    plotlinearRegression(testData, testTarget,alpha,alpha1,alpha2,iterations,reg,reg,reg,EPS, "alpha")

    #Generalization
    reg1=0.001
    reg2= 0.1
    reg3= 0.5
    print('Regularization')
    plt.figure(4)
    #Training losses
    plotlinearRegression(trainData, trainTarget,alpha,alpha,alpha,iterations,reg1,reg2,reg3,EPS, "regularization parameter")
    #Validation losses
    plt.figure(5)
    plotlinearRegression(validData, validTarget,alpha,alpha,alpha,iterations,reg1,reg2,reg3,EPS, "regularization parameter")
    #Testing Losses
    plt.figure(6)
    plotlinearRegression(testData, testTarget,alpha,alpha,alpha,iterations,reg1,reg2,reg3,EPS, "regularization parameter")


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
    plt.show()

'''
    W3 = np.zeros(Data.shape[1] * Data.shape[2])

    b=np.zeros(1)
    #Get the optimized weight, bias and the loss

    W3, b, trainloss4 = grad_descent(W, b, Data, Target, alpha, 5000, 0.1, EPS, "CE")
    print('CE loss: ', trainloss4[len(trainloss4)-1])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing Loss with different '+ parameter)

        #plot the graph
        if(parameter=="alpha"):
            plt.plot(X_test, trainloss, label='alpha=0.005')


    #plotting
    X_test4 = np.linspace(0, len(trainloss4)-1, len(trainloss4)-1)'''





main()

'''W=np.array([[1],[2],[3]])
x=np.array([[3,4,3],[1,2,2],[3,4,1]])
y=np.array([[3],[4],[5]])
b=np.array([[1]])
reg=3
print(MSE(W,b,x,y,reg))
print(gradMSE(W,b,x,y,reg))
print(crossEntropyLoss(W, b, x, y, reg))'''
