# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    uniqueMat, indices = np.unique(y, return_inverse= True)    
    categorySize = uniqueMat.size
    
    mylist = []    
    for i in range(categorySize):
        mylist.append(np.zeros(shape = X[0].shape))                    
    
    for count in range(len(y)):
       arr = mylist[indices[count]]
       mylist[indices[count]] = np.vstack((arr,X[count]))

    for i in range(categorySize):
        arr = mylist[i]
        mylist[i] = np.delete(arr,(0),axis=0)        
    
    mylist_mean = []
    for i in range(categorySize):
        mylist_mean.append(np.mean(mylist[i],axis=0))
       
    
    for i in range(len(mylist_mean)):
        if i == 0:
            means = np.vstack((mylist_mean[i],mylist_mean[i+1]))
        elif  i > 1:
            means = np.vstack((means,mylist_mean[i]))

    means = means.transpose()            
    
    covmat = np.cov(X,rowvar=0);    
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    uniqueMat, indices = np.unique(y, return_inverse= True)    
    categorySize = uniqueMat.size
    
    mylist = []    
    for i in range(categorySize):
        mylist.append(np.zeros(shape = X[0].shape))                  
    
    for count in range(len(y)):
       arr = mylist[indices[count]]
       mylist[indices[count]] = np.vstack((arr,X[count]))

    # delete the first row
    for i in range(categorySize):
        arr = mylist[i]
        mylist[i] = np.delete(arr,(0),axis=0)   
        
            
    #print "starts mylist"
    #print mylist[1]
    #print "ends"     
    
    mylist_mean = []
    for i in range(categorySize):
        mylist_mean.append(np.mean(mylist[i],axis=0))
       
    
    for i in range(len(mylist_mean)):
        if i == 0:
            means = np.vstack((mylist_mean[i],mylist_mean[i+1]))
        elif  i > 1:
            means = np.vstack((means,mylist_mean[i]))

    means = means.transpose()        
    
    #print "starts mylist length"
    #print len(mylist)
    #print "ends"
    
    #print "starts means"
    #print means.transpose()
    #print "ends"
    
    covmats = None
    
    for countOfClass in range(len(mylist)):
        if covmats is None:
            covmats = [np.array(np.cov(mylist[countOfClass],rowvar=0))]
        else:
            covmats.append(np.array(np.cov(mylist[countOfClass],rowvar=0)))
    
    #print "starts covmats"
    #print covmats
    #print "ends"
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD    
    D = len(means)    
    
    py = float(1 / 1)
    
    pdfList = []
    yPredicted = None
    
    #means to be transposed to get the mean of each class
    means = means.transpose()
    
    for row in Xtest:
        for mean in means:
            g = row        
            significant = 1 / (np.power((2 * 3.142), D/2) * np.power(np.linalg.det(covmat),0.5))
            xMinusMean = np.subtract(row,mean)
            invSigma = np.linalg.inv(covmat)
            
            ans = np.dot(np.dot(xMinusMean, invSigma), np.transpose(xMinusMean))
            ans = - ans / 2
            
            numeratorOfExp = ans
            
            #numeratorOfExp = - (np.transpose(xMinusMean) * invSigma * xMinusMean) / 2
            pdf = py * significant * np.exp(numeratorOfExp)
            pdfList.append(pdf)
        maxPdf = max(pdfList)
        
        if yPredicted is None:
            yPredicted = np.array(float(pdfList.index(maxPdf) + 1))
        else:
            yPredicted = np.vstack((yPredicted, float(pdfList.index(maxPdf) + 1)))
            
        maxPdf = []
        pdfList = []
        
    #print "yPred starts"
    #print yPredicted
    #print "yPred ends"
    
    #print('\n Validation set Accuracy:' + str(100*np.mean((yPredicted == ytest).astype(float))) + '%')
    
    result = str(100*np.mean((yPredicted == ytest).astype(float)))
    
    return result

def ldaPlot(xTest,yPred,means,covmat,yTest):                
    minLimit = np.amin(xTest)
    maxLimit = np.amax(xTest)
    
    x1 = np.linspace(minLimit,maxLimit,num=100)
    x2 = np.linspace(minLimit,maxLimit,num=100)
    
    grid=np.meshgrid(x1,x2)
    grid[0] = grid[0].reshape(10000,1)
    grid[1] = grid[1].reshape(10000,1)
    
    data = np.concatenate((grid[0],grid[1]),axis = 1)    
        
    result,yPredicted = ldaTest(means,covmat, data,data)

    yPred = yPredicted        

    for i in range(len(yPred)):
        colors = ['b', 'c', 'y', 'g', 'r']
        index_color = int(yPred[i]-1)
        plt.scatter(data[i][0], data[i][1], c=str(colors[index_color]))#colors[index_color])
        plt.title("LDA")
    plt.show()
        
    return    
    
def qdaPlot(means,covmats,xTest):  
     minLimit = np.amin(xTest)
     maxLimit = np.amax(xTest)
    
     x1 = np.linspace(minLimit,maxLimit,num=100)
     x2 = np.linspace(minLimit,maxLimit,num=100)
    
     grid=np.meshgrid(x1,x2)
     grid[0] = grid[0].reshape(10000,1)
     grid[1] = grid[1].reshape(10000,1)
    
     data = np.concatenate((grid[0],grid[1]),axis = 1)    
        
     result, yPredicted = qdaTest(means,covmats,data,data)

     yPred = yPredicted        

     for i in range(len(yPred)):
        colors = ['b', 'c', 'y', 'g', 'r']
        index_color = int(yPred[i]-1)
        plt.scatter(data[i][0], data[i][1], c=str(colors[index_color]))#colors[index_color])
        plt.title("QDA")
     plt.show()
     
     return
   

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
   # IMPLEMENT THIS METHOD    
    D = len(means)    
    
    py = float(1 / 1)
    
    pdfList = []
    yPredicted = None
    
    #means to be transposed to get the mean of each class
    means = means.transpose()
    
    for row in Xtest:
        for index in range(len(means)):            
            mean = means[index]   
            i = 0
            covmat = covmats[index]
            i = i+1
                    
            significant = 1 / (np.power((2 * 3.142), D/2) * np.power(np.linalg.det(covmat),0.5))
            xMinusMean = np.subtract(row,mean)
            invSigma = np.linalg.inv(covmat)
            
            ans = np.dot(np.dot(xMinusMean, invSigma), np.transpose(xMinusMean))
            ans = - ans / 2
            
            numeratorOfExp = ans
            
            #numeratorOfExp = - (np.transpose(xMinusMean) * invSigma * xMinusMean) / 2
            pdf = py * significant * np.exp(numeratorOfExp)
            pdfList.append(pdf)
        maxPdf = max(pdfList)
        
        if yPredicted is None:
            yPredicted = np.array(float(pdfList.index(maxPdf) + 1))
        else:
            yPredicted = np.vstack((yPredicted, float(pdfList.index(maxPdf) + 1)))
            
        maxPdf = []
        pdfList = []
        
    result = 100*np.mean((yPredicted == ytest).astype(float))
    
    return result

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD     
    # w = inverse(X.T * X) (X.T*Y)                                              
    #opts = {'maxiter' : 50}    # Preferred value.
    #initialWeights = np.zeros((X.shape[1],1))
    #args = (X,y)
    #w=OLEmin(args);
    b = np.dot(X.T,X)
    c = np.dot(X.T,y)
    wR = np.linalg.inv(b)
    w = np.dot(wR, c)  
    return w;
    #w = minimize(OLEmin, initialWeights, jac=True, args=args,method='CG', options=opts)    
    #return w.x

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    # w = inverse(lambda . N . I + X.T * X) (X.T*Y)
    # w = inverse(a + b)(c)
    # w = inverse(wR)(c)
    
    N = X.shape[0];
    a = lambd*N*np.identity(X.shape[1])
    b = np.dot(X.T,X)
    c = np.dot(X.T,y)
    wR = np.linalg.inv(a + b)
    w = np.dot(wR, c)
    
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:   
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    # jw = 1/N root((y-wx)^2)
    # jw = 1/N root((y-b)^2)
    # jw = 1/N root(a^2)
    N = Xtest.shape[0];    
   
   
    b= np.dot(Xtest , w)    

    a = (ytest - b)
    a = np.dot(a.T,a)
    
    jw =  a.sum(axis=0)

    jw = np.sqrt(jw)

    rmse = jw/N;
    
    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda 
    # error_grad = dJ/dw=1/N(−y.T * X + w.T*(X.T * X))+λw
    # error_grad = dJ/dw=1/N(−a + w.T*(b))+λw
    # error=1/2N (y - w.T * x)^2 + 1/2 lambd * w.T *w

    # IMPLEMENT THIS METHOD                                             
    N = X.shape[0];    
   
    b= np.dot(X , w)    
    b=np.matrix(b)
    b=b.T
    a = (y - b)

    a = np.dot(a.T,a)

    jw1 =  a.sum(axis=0)

    jw1=jw1/(2*N);
    
    w_scalar=np.dot(w.T , w)
    jw2 = lambd * w_scalar/2
    error=jw1+jw2

    a = np.dot(y.T,X)
    
    b = np.dot(X.T,X)

    c = np.dot(w.T,b)
    c=np.matrix(c)

    jw1=c-a
    jw1=jw1/N

    jw2 = np.dot(lambd , w)
    jw2=np.matrix(jw2)

    error_grad=jw1+jw2

    error_grad = np.squeeze(np.asarray(error_grad))
    #print error
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    N=x.shape[0]
    Xd=np.ones([N, ])
    Xd=np.matrix(Xd)
    
    #for i in range(N):
    for j in range(p+1):
        if j!=0:
            pow=np.power(x,j)
            pow=np.matrix(pow)
            Xd=np.concatenate((Xd,pow),axis=0)
    Xd=Xd.T          
   # print 'Xd s is ' +str(Xd.shape)
    return Xd


# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('C:\Users\SAnanda\Dropbox\Spring2015\CSE574ML\pa2\sample.pickle','rb'))            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('C:\Users\SAnanda\Dropbox\Spring2015\CSE574ML\pa2\diabetes.pickle','rb'))  
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
