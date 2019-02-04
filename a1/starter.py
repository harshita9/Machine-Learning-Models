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
    # variable to store total loss (L = L)
    total_loss = 0

    N = len(x)
    yhat=np.dot(W.flatten(),x)
    MSEloss = yhat.flatten()-y.flatten()+ b
    MSEloss=np.linalg.norm(MSEloss) **2

    weight_decay_loss = (reg / 2) * (np.linalg.norm(W) ** 2)
    total_loss = (MSEloss / (2 * N)) + weight_decay_loss
    return total_loss

def gradMSE(W, b, x, y, reg):
    gradMSE_weight = 0
    gradMSE_bias = 0

    N = len(x)
    yhat = np.dot(W.flatten(), x)
    grad_MSE = (yhat.flatten() - y.flatten() + b)
    gradMSE_weight = np.dot(grad_MSE, np.transpose(x) * 2)

    grad_weight_decay_loss = reg * W
    gradMSE_weight = (np.transpose(gradMSE_weight / (2 * N))) + grad_weight_decay_loss

    gradMSE_bias = (grad_MSE * 2)
    gradMSE_bias = np.sum(gradMSE_bias / (2 * N))

    return gradMSE_weight, gradMSE_bias


def crossEntropyLoss(W, b, x, y, reg):
    cross_entropy_loss = 0
    N = len(x)

    z = np.dot(W.flatten(), x) + b
    yhat=1/(1+np.exp(-1*z))


    ylogx=-1* np.dot(np.dot(y,np.log(yhat)),x)

    secondexpression=np.dot((1-y),np.log(1-np.dot(yhat,x)))


    crossloss=1/N * np.sum((ylogx-secondexpression))

    weight_decay_loss = (reg / 2) * (np.linalg.norm(W) ** 2)

    cross_entropy_loss = crossloss + weight_decay_loss


    return cross_entropy_loss


def gradCE(W, b, x, y, reg):
        gradCE_weight, gradCE_bias = 0, 0
        N = len(x)

        z = np.dot(W.flatten(), x) + b

        yhat=1/(1+np.exp(-1*z))

        yandyhat=-1* np.dot(y,1-yhat)

        secondexpression=np.dot((1-y), yhat)

        grad_weight_decay_loss = reg * W

        gradCE_weight = np.dot(yandyhat + secondexpression, x) / N + grad_weight_decay_loss

        gradCE_bias = (yandyhat + secondexpression) / N

        return gradCE_weight, gradCE_bias

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType):
    # Initialize W the weight vector to zeros (784 x 1 array)
    W = np.zeros(trainingData.shape[1] * trainingData.shape[2])
    # Initialize b the bias vector to zero (1 x 1 array)
    b = np.zeros(1)
    # reshape x to be a 2D array (number of samples x 784)
    x = trainingData.reshape(trainingData.shape[0],(trainingData.shape[1]*trainingData.shape[2]))
    x = np.transpose(x)

    y = trainingLabels

    i = 0
    train_loss = []

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

    #Get final loss
    if lossType == "MSE":
        loss = MSE(W,b,x,y,reg)
    elif lossType == "CE":
        loss = crossEntropyLoss(W,b,x,y,reg)
    else:
        loss = MSE(W,b,x,y,reg)

    train_loss.append(loss)
    
    return W,b,train_loss

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    #Initialize weight and bias tensors
    #W = tf.tructated_normal([784,1],0.5)
    W = tf.Variable(tf.zeros[28 * 28, 1], name="weights")
    b = tf.Variable(tf.zeros[1], name="biases")

    x = tf.placeholder(tf.float32, [None, 784], name="data")
    #x = tf.reshape(x, [None, 28 * 28])
    labels = tf.placeholder(tf.float32,[None,1], name="labels")
    reg = tf.placeholder(tf.float32, name="reg")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')


    tf.set_random_seed(421)


    total_loss = 0

    if loss == "MSE":
        predicted_y = tf.matmul(x, W) + b
        yhat = tf.sigmoid(tf.matmul(x, W) + b)
        ce = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(labels=y, logits=linear_pred_y), name="cross_entropy_loss")
        wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)), name="weight_decay_loss")

        total_loss = ce + wd

    elif loss == "CE":
        predicted_y = tf.matmul(x, W) + b
        yhat = tf.sigmoid(predicted_y)
        ce = tf.reduce_mean(tf.sigmoid_cross_entropy_with_logits(labels=y, logits=predicted_y), name="cross_entropy_loss")
        wd = tf.multiply(lamb / 2, tf.reduce_sum(tf.square(w)), name="weight_decay_loss")

        total_loss = ce + wd



    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

return


def normalMSE(x,y,reg):

    inversexx = np.linalg.inv(np.dot(np.transpose(x),x)+reg*np.identity(x.shape[1]))
    w=np.dot(np.dot(inversexx,np.transpose(x)).T,y)
    return w


def plotlinearRegression(Data, Target,alpha,alpha1,alpha2,iterations,reg1,reg2,reg3,EPS,parameter):
    #initialize W and bias
    W = np.zeros(Data.shape[1] * Data.shape[2])
    W1 = np.zeros(Data.shape[1] * Data.shape[2])
    W2 = np.zeros(Data.shape[1] * Data.shape[2])

    b=np.zeros(1)
    #Get the optimized weight, bias and the loss
    W, b,trainloss = grad_descent(W, b, Data, Target, alpha, iterations, reg1, EPS, "MSE")
    print('MSE loss 1: ', trainloss[len(trainloss)-1])
    W1, b,trainloss2 = grad_descent(W, b, Data, Target, alpha1, iterations, reg2, EPS,"MSE")
    print('MSE loss 2: ', trainloss2[len(trainloss2)-1])
    W2, b,trainloss3 = grad_descent(W, b, Data, Target, alpha2, iterations, reg3, EPS,"MSE")
    print('MSE loss 3: ', trainloss3[len(trainloss)-1])


    #plotting
    X_test = np.linspace(0, len(trainloss)-1, len(trainloss)-1)
    X_test2 = np.linspace(0, len(trainloss2)-1, len(trainloss2)-1)
    X_test3 = np.linspace(0, len(trainloss3)-1, len(trainloss3)-1)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing Loss with different '+ parameter)

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

    reg1=0.001
    reg2= 0.1
    reg3= 0.5
    EPS = 1 * 10 ** (-7)

    alpha = 0.005
    alpha1 = 0.001
    alpha2 = 0.0001
    plt.close('all')
    #Tuning the Learning Rate Plot, Plot losses
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
    W, b,trainloss = grad_descent(W, b, trainData, trainTarget, alpha1, iterations, reg1, EPS, "MSE")
    print('computation time batched GD: ',time.time()-startBatched)

    x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    x=np.transpose(x)
    y=trainTarget

    #Calcualate error for batch GD
    loss_batched=MSE(W,b,x,y,reg1)
    #calculate normal equation
    W2=normalMSE(x,y)
    loss=MSE(W2,x,y,0,0)

    print('loss batched: ',loss_batched)
    print('loss normal: ',loss)

    #plot with different reg value


    plt.show()


main()

'''W=np.array([[1],[2],[3]])
x=np.array([[3,4,3],[1,2,2],[3,4,1]])
y=np.array([[3],[4],[5]])
b=np.array([[1]])
reg=3
print(MSE(W,b,x,y,reg))
print(gradMSE(W,b,x,y,reg))
print(crossEntropyLoss(W, b, x, y, reg))'''
