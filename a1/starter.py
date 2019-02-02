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
    MSEloss = yhat.flatten()-y.flatten()+ b
    MSEloss=np.linalg.norm(MSEloss) **2

    weight_decay_loss = (reg / 2) * (np.linalg.norm(W) ** 2)
    total_loss = (MSEloss / (2 * N)) + weight_decay_loss
    return total_loss





def gradMSE(W, b, x, y, reg):
    gradMSE_weight = 0
    gradMSE_bias = 0

    N = len(x)
    yhat = np.dot(np.transpose(W), x)
    grad_MSE = (yhat.flatten() - y.flatten() + b)
    gradMSE_weight = np.dot(grad_MSE, x * 2)

    grad_weight_decay_loss = reg * W
    gradMSE_weight = (np.transpose(gradMSE_weight / (2 * N))) + grad_weight_decay_loss

    gradMSE_bias = (grad_MSE * 2)
    gradMSE_bias = np.sum(gradMSE_bias / (2 * N))

    return gradMSE_weight, gradMSE_bias


def crossEntropyLoss(W, b, x, y, reg):
    #
    pass


def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
<<<<<<< HEAD
    '''W = np.zeros(trainingData.shape[1] * trainingData.shape[2])
    x=trainingData()
    i=0
    while previous_step_size > precision and i < iterations:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",cur_x) #Print iterations
    
    print("The local minimum occurs at", cur_x)'''


    return W
=======

    pass
>>>>>>> 8f04afa06181ee71376d7253fd1a8adbc1acc2d2

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass

#if __name__ == "__main__":
def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    x, y = trainData, trainTarget
    W = np.zeros(x.shape[1] * x.shape[2])
<<<<<<< HEAD
=======
    x_shape1, x_shape2 = x.shape[0], x.shape[1] * x.shape[2]
    print (x.shape)
    print (x_shape1, x_shape2)
    x.reshape(x_shape1, x_shape2)
>>>>>>> 8f04afa06181ee71376d7253fd1a8adbc1acc2d2
    print (W.shape)
    print (x.shape)

main()

W=np.array([[1],[2],[3]])
x=np.array([[3,4,3],[1,2,2],[3,4,1]])
y=np.array([[3],[4],[5]])
b=np.array([[1]])
reg=3
print(MSE(W,b,x,y,reg))
print(gradMSE(W,b,x,y,reg))
