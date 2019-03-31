import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True
#is_valid = False

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

    diff=tf.expand_dims(X,1)-tf.expand_dims(MU,0)
    sq=(tf.linalg.norm(diff,axis=2))**2

    return sq

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    sigma = tf.exp(sigma)
    distances = distanceFunc(X, mu)
    first_term = (-1/2) * (tf.log(2 * np.pi) + 2 * tf.log(sigma))
    second_term = -1 * (distances / (2 * tf.square(tf.squeeze(sigma))))

    gaussPDF = tf.squeeze(first_term) + second_term

    return gaussPDF

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    input_tensor = log_PDF + tf.squeeze(log_pi)
    log_post = hlp.logsoftmax(input_tensor) # uses logsumexp

    return log_post

def calculate_loss (log_PDF, log_pi):

    P = log_PDF + tf.squeeze(log_pi)
    P = hlp.reduce_logsumexp(P, 1, True)
    loss = -1 * tf.reduce_mean(P)

    return loss


def build_graph(K, D, alpha):

    tf.set_random_seed(421)

    # create placeholders for x, y and alpha
    X = tf.placeholder(dtype=tf.float32, shape=[None, D], name = "data")
    MU = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[K, D], name = "MU"))
    phi = tf.Variable(tf.truncated_normal([1, K], mean=0.0, stddev=1.0, dtype=tf.float32))
    #MU = tf.get_variable(tf.float32, shape=(K, D), name="MU")
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
    sigma = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[K, 1], name="sigma"))
    log_pi = hlp.logsoftmax(phi)

    # compute the P(xn | zn = K)
    log_PDF = log_GaussPDF(X, MU, sigma)

    # compute the P(z = k)
    log_post = log_posterior(log_PDF, log_pi)
    pred = tf.argmax(log_post, axis=1)

    total_loss = calculate_loss(log_PDF, log_post)

    gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha, name="Adam").minimize(total_loss)

    return total_loss, gd_optimizer, X, MU, pred, learning_rate

def MoG(epochs, alpha, K):

    D = data.shape[1]
    N = data.shape[0]
    total_loss, gd_optimizer, X, MU, pred, learning_rate = build_graph(K, D, alpha)
    init = tf.global_variables_initializer()

    traindata = data
    if is_valid != False:
        validdata = val_data

    trainLoss = []
    validLoss = []

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            _, tl, prediction = sess.run([gd_optimizer, total_loss, pred],
                feed_dict={X:data, learning_rate:alpha})

            # get loss per iteration
            err = total_loss.eval(feed_dict={X: traindata})
            err = err / N
            print(err)
            trainLoss.append(err)

            if is_valid != False:
                N2 = validdata.shape[0]
                err = total_loss.eval(feed_dict={X: validdata})
                err = err / N2
                validLoss.append(err)

        predic = pred.eval(feed_dict={X: traindata})
        centers = MU.eval()
        predic2 = 0
        centers2 = 0

        if is_valid != False:
            predic2 = pred.eval(feed_dict={X: validdata})
            centers2 = MU.eval()


    print("The final training Loss is ", trainLoss[len(trainLoss)-1])

    if is_valid != False:
        print("The final validation error is ", validLoss[len(validLoss)-1])

    #if is_valid != False:
     #   return trainLoss, validLoss, predic, predic2, centers, centers2

    return trainLoss, validLoss, predic, centers


#Learning K-Means
def mainPart1():

    K=3
    alpha=0.1
    epochs=1000
    D=dim
    trainLoss, validLoss, predic, centers = MoG(epochs, alpha, K)
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

def mainPart2():
    plt.close('all')

    K1=1
    K2=2
    K3=3
    K4=4
    K5=5

    alpha=0.1
    epochs=1000
    D=dim

    print("Total K=1")
    trainLoss, validLoss, predic, centers = MoG(epochs, alpha, K1)
    plt.figure(2)
    plt.scatter(data[:,0],data[0:,1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data classified to K=1')

    print("Total K=2")
    plt.figure(3)
    trainLoss, validLoss, predic, centers = MoG(epochs, alpha, K2)
    plotDataClusters(data,K2,predic)
    plt.title('Data classified to K=2')


    print("Total K=3")
    plt.figure(4)
    trainLoss, validLoss, predic, centers = MoG(epochs, alpha, K3)
    plotDataClusters(data,K3,predic)
    plt.title('Data classified to K=3')

    print("Total K=4")
    plt.figure(5)
    trainLoss, validLoss, predic, centers = MoG(epochs, alpha, K4)
    plotDataClusters(data,K4,predic)
    plt.title('Data classified to K=4')

    plt.figure(6)
    print("Total K=5")
    trainLoss, validLoss, predic, centers = MoG(epochs, alpha, K5)
    plotDataClusters(data,K5,predic)
    plt.title('Data classified to K=5')
    plt.show()

    #################  PART 1.3   ############################
    '''plt.close('all')

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
    plt.show()'''


mainPart2()
