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

    yhat = np.dot(np.transpose(W), x).flatten() + b

    MSEloss = yhat.flatten()-y.flatten()+ b
    MSEloss=np.linalg.norm(MSEloss) **2

    weight_decay_loss = (reg / 2) * (np.linalg.norm(W) ** 2)
    total_loss = (MSEloss / N) + weight_decay_loss

    return cross_entropy_loss


def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    W = np.zeros(trainingData.shape[1] * trainingData.shape[2])
    #W=np.transpose(W)
    b=np.zeros(1)
    x=trainingData.reshape(trainingData.shape[0],(trainingData.shape[1]*trainingData.shape[2]))
    x=np.transpose(x)
    y=trainingLabels
    i=0
    weight_check=True
    bias_check=True
    train_loss=[]
    for i in range(iterations):

        #plot the losses
        loss=MSE(W,b,x,y,reg)
        train_loss.append(loss)

        weight_gradient, bias_gradient=gradMSE(W,b,x,y,reg)
        if (weight_check):
            #Calculate the direction of the gradient of weight vector
            norm_weight_grad= np.linalg.norm(weight_gradient)
            weight_direction=-1*weight_gradient /norm_weight_grad
            #Calculate the new weight vecton
            new_w=W+alpha*weight_direction
            #weight error
            difference_weight=np.linalg.norm(new_w-W)**2

            if(difference_weight<EPS):
                weight_check=False
            else:
                W=new_w

        #Calulate optimal bias
        if(bias_check):
            norm_bias_grad=np.linalg.norm(bias_gradient)
            bias_direction=-1*bias_gradient / norm_bias_grad

            new_b=b+alpha*bias_direction

            differece_bias=np.linalg.norm(new_b-b)**2

            if(differece_bias<EPS):
                bias_check=False
            else:
                b=new_b


        if (not bias_check and not weight_check):
            break




    return W,b,train_loss

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass


def normalMSE(x,y):
    inversexx=np.linalg.inv(np.dot(np.transpose(x),x))
    w=np.dot(np.dot(inversexx,np.transpose(x),y))

    return w

#if __name__ == "__main__":
def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    W = np.zeros(trainData.shape[1] * trainData.shape[2])

    b = np.zeros(1)


    iterations = 5000
    reg = 0

    reg1=0.001
    reg2= 0.1
    reg2= 0.5
    EPS = 1 * 10 ** (-7)

    alpha = 0.005
    alpha1 = 0.001
    alpha2 = 0.0001

    #different alpha value
    W, b,trainloss = grad_descent(W, b, trainData, trainTarget, alpha, iterations, reg, EPS)
    #W, b,trainloss2 = grad_descent(W, b, trainData, trainTarget, alpha1, iterations, reg, EPS)
    #W, b,trainloss3 = grad_descent(W, b, trainData, trainTarget, alpha2, iterations, reg, EPS)

    loss_batched=trainloss[len(trainloss)-1]
    #different reg value
    '''W, b,trainloss = grad_descent(W, b, trainData, trainTarget, alpha, iterations, reg1, EPS)
    W, b,trainloss2 = grad_descent(W, b, trainData, trainTarget, alpha, iterations, reg2, EPS)
    W, b,trainloss3 = grad_descent(W, b, trainData, trainTarget, alpha, iterations, reg3, EPS)
'''

'''    plt.close('all')
    #plt.scatter(iteration, train_target)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    X_test = np.linspace(0, len(trainloss), len(trainloss))
    X_test2 = np.linspace(0, len(trainloss2), len(trainloss2))
    X_test3 = np.linspace(0, len(trainloss3), len(trainloss3))

    plt.title('Tuning the learning_rate')'''
'''
    #plot with different alpha value
    plt.plot(X_test, trainloss, label='alpha=0.05')
    plt.plot(X_test2, trainloss2, label='alpha=0.001')
    plt.plot(X_test3, trainloss3, label='alpha=0.0001')'''

    x=trainData.reshape(trainData.shape[0],(trainData.shape[1]*trainData.shape[2]))
    x=np.transpose(x)
    y=trainingLabels
    #calculate normal equation
    W2=normalMSE(x,y)
    loss=MSE(W2,x,y,0,0)

    print('loss batched: ',loss_batched)
    print('loss normal: ',loss)

    #plot with different reg value
    '''plt.plot(X_test, trainloss, label='reg=0.001')
    plt.plot(X_test2, trainloss2, label='reg=0.1')
    plt.plot(X_test3, trainloss3, label='alpha=0.5')'''

    W, b = grad_descent(W, b, trainData, trainTarget, alpha, iterations, reg, EPS)
    W, b = grad_descent(W, b, validData, validTarget, alpha, iterations, reg, EPS)
    W, b = grad_descent(W, b, testData, testTarget, alpha, iterations, reg, EPS)

    '''plt.legend()
    plt.show()'''

main()
'''
W=np.array([[1],[2],[3]])
x=np.array([[3,4,3],[1,2,2],[3,4,1]])
y=np.array([[3],[4],[5]])
b=np.array([[1]])
reg=3
print(MSE(W,b,x,y,reg))
print(gradMSE(W,b,x,y,reg))'''
