import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

def getVal(data):
# For Validation set

  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

  return val_data, data


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    #X(n)-MU(k)
    diff=tf.expand_dims(X,1)-tf.expand_dims(MU,0)
    #get the square of the normal
    sq=(tf.linalg.norm(diff,axis=2))**2

    return sq

def loss(X,MU):
    #Get the distance for K means
    distance=distanceFunc(X,MU)
    #Get the loss with minimum for K and sum from n=1 to N
    total_loss=tf.reduce_sum(tf.reduce_min(distance,1))

    predicted=tf.argmin(distance,1)

    return total_loss, predicted



def model_implementation(K,alpha,D):
    '''Build computational graph for SGD.'''

    tf.set_random_seed(421)

    # create placeholders for x, MU and alpha
    x = tf.placeholder(dtype=tf.float32, shape=[None, D])
    MU = tf.Variable(tf.random.normal(dtype=tf.float32, shape=[K, D], name = "MU"))
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    #get the loss and prediction
    total_loss,predicted=loss(x,MU)



    adam_optimizer = tf.train.AdamOptimizer(learning_rate = alpha,beta1=0.9,beta2=0.99,epsilon=1e05, name="Adam").minimize(total_loss)

    return predicted, x, MU, total_loss, adam_optimizer,learning_rate

def calculateAccuracy(predic,K):
    '''Calculate the percent of data in each K'''

    result=[]
    pr=list(predic)

    N=len(pr)
    #for each K count the number of data in prediction that says it belongs to that
    #particular K
    for i in range (0,K):

        num=pr.count(i)
        print(num)
        result.append(100*num/(float(N)))

    return result

def printKNum(result):
    '''Print the percent of data in each K'''

    for i in range(len(result)):
        print("K: ",i+1," ",result[i],"%")

    return



def stochastic_gradient_descent(epochs, alpha,K,D,trainData,ValidData=None,isValid=False):

    ''' Uses adam optimizer and the stochastic gradient descent algorithm
    to minimize losses. '''

    # build computaional graph and initialize variables
    predicted, x, MU, total_loss, adam_optimizer,learning_rate = model_implementation(K,alpha,D)
    init = tf.global_variables_initializer()


    trainLoss = []
    trainAcc = []
    validLoss = []
    validAcc = []
    predic=[]
    N=(trainData.shape)[0]


    # start tensorflow session
    with tf.Session() as sess:
        sess.run(init)
        # SGD algorithm
        # for each interation loop through all data and run the session
        for i in range(epochs):

            #run for train data
            _,MU_new,pred, tl = sess.run([adam_optimizer,MU, predicted, total_loss],
            feed_dict={x:trainData, learning_rate:alpha})



            # get loss per iteration
            err = total_loss.eval(feed_dict={x: trainData})
            err=err/N


            trainLoss.append(err)
            #for valid data get the loss
            if isValid != False:
                N2=(ValidData.shape)[0]
                #get the loss
                err = total_loss.eval(feed_dict={x: ValidData})
                err=err/N2

                validLoss.append(err)
        #Get the prediction for training and validation data
        predic=predicted.eval(feed_dict={x: trainData})
        if isValid != False:
            predic2=predicted.eval(feed_dict={x: ValidData})



    #calculate percente of data in each clusters
    acc = calculateAccuracy(predic, K)
    # print final error and accuracy
    print("The final training Loss is ", trainLoss[len(trainLoss)-1])
    print("The percentage of the training data belonging to each of K clusters is: ")
    printKNum(acc)



    if isValid != False:
        print("The final validation error is ", validLoss[len(validLoss)-1])
        acc2 = calculateAccuracy(predic2, K)
        print("The percentage of the validation data belonging to each of K clusters is: ")
        printKNum(acc2)



    return trainLoss, validLoss, predic


def plotDataClusters(datapt,K,predic):
    '''Plots the data into the predicted clusters'''

     colour = plt.cm.rainbow(np.linspace(0,1,K))


     for i in range(K):
         lbl="K = " + str(i+1)
         idx=np.where(predic==i)
         print(idx)
         plt.scatter(datapt[idx,0],datapt[idx,1],c=colour[i],label=lbl)

     plt.xlabel('X1')
     plt.ylabel('X2')
     plt.legend()

     return



#Learning K-Means
def mainPart1():
'''Get the plots for clusters and losses'''
#################  PART 1.1  ############################
    K=3
    learning_rate=0.1
    epochs=3000
    D=dim
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K,D,data,ValidData=None)
    plt.close('all')
    #plot the loss vs epochs
    X_test = np.linspace(0, len(trainLoss), len(trainLoss))

    plt.figure(1)
    plt.plot(X_test, trainLoss)

    plt.xlabel('Number of Updates')
    plt.ylabel('Loss')
    plt.title('Loss vs number of updates')

    #plt.legend()
    plt.show()



    #################  PART 1.2   ############################
    plt.close('all')
    #plot the data without labels(K=1)



    K1=1
    K2=2
    K3=3
    K4=4
    K5=5
    learning_rate=0.1
    epochs=6000
    D=dim

    print("Total K=1")
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K1,D,data,ValidData=None)
    plt.figure(2)
    plt.scatter(data[:,0],data[0:,1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data classified to K=1')

    print("Total K=2")
    plt.figure(3)
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K2,D,data,ValidData=None)
    plotDataClusters(data,K2,predic)
    plt.title('Data classified to K=2')


    print("Total K=3")
    plt.figure(4)
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K3,D,data,ValidData=None)
    plotDataClusters(data,K3,predic)
    plt.title('Data classified to K=3')

    print("Total K=4")
    plt.figure(5)
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K4,D,data,ValidData=None)
    plotDataClusters(data,K4,predic)
    plt.title('Data classified to K=4')

    plt.figure(6)
    print("Total K=5")
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K5,D,data,ValidData=None)
    plotDataClusters(data,K5,predic)
    plt.title('Data classified to K=5')
    plt.show()

    #################  PART 1.3   ############################
    plt.close('all')

    vali_data, train_data=getVal(data)


    K1=1
    K2=2
    K3=3
    K4=4
    K5=5
    learning_rate=0.1
    epochs=3500
    D=dim

    print("Total K=1")
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K1,D,train_data,ValidData=vali_data,isValid= True)
    plt.figure(7)
    plt.scatter(train_data[:,0],train_data[0:,1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Training Data classified')

    print("Total K=2")
    #plt.figure(3)
    trainLoss, validLoss1,predic=stochastic_gradient_descent(epochs, learning_rate,K2,D,train_data,ValidData=vali_data,isValid= True)
    #plotDataClusters(data,K2,predic)
    #plt.title('Data classified to K=2')


    print("Total K=3")
    #plt.figure(4)
    trainLoss, validLoss2,predic=stochastic_gradient_descent(epochs, learning_rate,K3,D,train_data,ValidData=vali_data,isValid= True)
    #plotDataClusters(data,K3,predic)
    #plt.title('Data classified to K=3')

    print("Total K=4")
    #plt.figure(5)
    trainLoss, validLoss3,predic=stochastic_gradient_descent(epochs, learning_rate,K4,D,train_data,ValidData=vali_data,isValid= True)
    #plotDataClusters(data,K4,predic)
    #plt.title('Data classified to K=4')

    #plt.figure(6)
    print("Total K=5")
    trainLoss, validLoss4,predic=stochastic_gradient_descent(epochs, learning_rate,K5,D,train_data,ValidData=vali_data,isValid= True)
    #plotDataClusters(data,K5,predic)
    #plt.title('Data classified to K=5')


    X_test1 = np.linspace(0, len(validLoss), len(validLoss))
    X_test2 = np.linspace(0, len(validLoss1), len(validLoss1))
    X_test3 = np.linspace(0, len(validLoss2), len(validLoss2))
    X_test4 = np.linspace(0, len(validLoss3), len(validLoss3))
    X_test5 = np.linspace(0, len(validLoss4), len(validLoss4))


    plt.figure(8)
    plt.plot(X_test1, validLoss, label="K=1")
    plt.plot(X_test2, validLoss1, label="K=2")
    plt.plot(X_test3, validLoss2, label="K=3")
    plt.plot(X_test4, validLoss3, label="K=4")
    plt.plot(X_test5, validLoss4, label="K=5")

    plt.xlabel('Number of Updates')
    plt.ylabel('Valid Loss')
    plt.title('Validation Loss vs number of updates for different K')
    plt.legend()
    plt.show()





mainPart1()


def mainPart2():

    ''' Running 100D data for part 2'''

    plt.close('all')

    vali_data, train_data=getVal(data)


    K1=5
    K2=10
    K3=15
    K4=20
    K5=30
    learning_rate=0.1
    epochs=3500
    D=dim

    print("Total K=5")
    trainLoss, validLoss,predic=stochastic_gradient_descent(epochs, learning_rate,K1,D,train_data,ValidData=vali_data,isValid= True)


    print("Total K=10")

    trainLoss1, validLoss1,predic=stochastic_gradient_descent(epochs, learning_rate,K2,D,train_data,ValidData=vali_data,isValid= True)



    print("Total K=15")

    trainLoss2, validLoss2,predic=stochastic_gradient_descent(epochs, learning_rate,K3,D,train_data,ValidData=vali_data,isValid= True)


    print("Total K=20")

    trainLoss3, validLoss3,predic=stochastic_gradient_descent(epochs, learning_rate,K4,D,train_data,ValidData=vali_data,isValid= True)


    print("Total K=30")
    trainLoss4, validLoss4,predic=stochastic_gradient_descent(epochs, learning_rate,K5,D,train_data,ValidData=vali_data,isValid= True)



    X_test1 = np.linspace(0, len(validLoss), len(validLoss))
    X_test2 = np.linspace(0, len(validLoss1), len(validLoss1))
    X_test3 = np.linspace(0, len(validLoss2), len(validLoss2))
    X_test4 = np.linspace(0, len(validLoss3), len(validLoss3))
    X_test5 = np.linspace(0, len(validLoss4), len(validLoss4))


    plt.figure(8)
    plt.plot(X_test1, validLoss, label="K=5")
    plt.plot(X_test2, validLoss1, label="K=10")
    plt.plot(X_test3, validLoss2, label="K=15")
    plt.plot(X_test4, validLoss3, label="K=20")
    plt.plot(X_test5, validLoss4, label="K=30")

    plt.xlabel('Number of Updates')
    plt.ylabel('Valid Loss')
    plt.title('Validation Loss vs number of updates for different K')
    plt.legend()




    X_test1 = np.linspace(0, len(trainLoss), len(trainLoss))
    X_test2 = np.linspace(0, len(trainLoss1), len(trainLoss1))
    X_test3 = np.linspace(0, len(trainLoss2), len(trainLoss2))
    X_test4 = np.linspace(0, len(trainLoss3), len(trainLoss3))
    X_test5 = np.linspace(0, len(trainLoss4), len(trainLoss4))


    plt.figure(9)
    plt.plot(X_test1, trainLoss, label="K=5")
    plt.plot(X_test2, trainLoss1, label="K=10")
    plt.plot(X_test3, trainLoss2, label="K=15")
    plt.plot(X_test4, trainLoss3, label="K=20")
    plt.plot(X_test5, trainLoss4, label="K=30")

    plt.xlabel('Number of Updates')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs number of updates for different K')
    plt.legend()

    plt.show()




mainPart2()
