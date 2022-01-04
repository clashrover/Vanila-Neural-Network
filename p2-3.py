import numpy as np
import time
import math
import sys

def error(y,out):
    e = np.subtract(y,out)
    e = np.square(e)
    sum = np.sum(e)
    m,n = np.shape(y)
    sum = sum/(2*m)
    return sum

def one_hot_encode(y):
    b = np.zeros((y.size,10))
    b[np.arange(y.size),y] = 1
    
    return b

def findAccuracy(layers,x,y, out_dir):
    out = np.matmul(x,layers[0])
    out = 1/(1+np.exp(-out))                # taking sigmoid
    for i in range(1,len(layers)):
        out = np.matmul(out,layers[i])
        out = 1/(1+np.exp(-out))
  
    m,n = np.shape(x)
    score =0
    f = open(out_dir,"w")
    for i in range(m):
        l = np.argmax(out[i])
        print(l,file = f)
        if l == y[i]:
            score+=1
    f.close()
    return score/m

def neuralNet(trainx,trainy,testx,testy,out_dir,batch_size,no_features,architecture,no_targets,eta):
    x1 = np.load(trainx)
    y1 = np.load(trainy)
    start = time.time()

    max_epoch = 100
    con_limit = 0.0001
    if architecture[0]==500:
        con_limit = 0.001
        max_epoch = 150
        


    layers = []
    architecture.insert(0,no_features)
    for i in range(1,len(architecture)-1):
        theta = 0.01*np.random.randn(architecture[i-1]+1,architecture[i]+1)
        layers.append(theta)
    
    theta_last = 0.01*np.random.randn(architecture[len(architecture)-2]+1,architecture[len(architecture)-1])
    layers.append(theta_last)

    m,n,k = np.shape(x1) 
    x = np.reshape(x1,(m,n*k))
    m,n = np.shape(x)

    x = np.hstack((x,np.ones((m,1))))
    
    y = one_hot_encode(y1)
    eta=eta/batch_size
    eta1 = eta
    j_prev = 0
    for epoch in range(max_epoch):
        j=0
        if epoch>0:
            eta=eta1/math.sqrt(epoch)
        for batch in range(0,int(m/batch_size)):
            
            x_batch = x[batch*batch_size:(batch+1)*batch_size,:]
            y_batch = y[batch*batch_size:(batch+1)*batch_size,:]
            # lets do forward propogation
            outputs = []
            out = np.matmul(x_batch,layers[0])
            out = 1/(1+np.exp(-out))                # taking sigmoid
            c=np.copy(out)
            outputs.append(c)
            for i in range(1,len(layers)):
                out = np.matmul(out,layers[i])
                out = 1/(1+np.exp(-out))
                c= np.copy(out)
                outputs.append(c)
            
            j = error(y_batch,out)
            # print(j)

            
            deltas = [None]*len(architecture)
            layersTemp = [None]*len(layers)
            for t in reversed(range(len(layers))):
                if t == len(layers)-1:
                    # case of output layer
                    # delJ(0)/delnetj = (yj-oj)(-1)oj(1-oj)        
                    out = outputs[t]    # current output. each jth column contains output for jth output unit for each example
                    delta = (y_batch-out)*out*(1-out)
                    deltas[t]=delta
                    input = outputs[t-1] # output of upstream layer is input of current layer
                    theta = layers[t]    
                    thetaT = theta + eta* np.matmul(np.transpose(input),delta)  # theta = theta + eta* (input.T * delta)
                    layersTemp[t] = thetaT
                else:
                    # case of hidden layer, change is only in delta
                    downNbrs_theta = layers[t+1]
                    # the jth row in theta_downNbrs contains thetalj
                    downNbr_delta = deltas[t+1]

                    delta = np.matmul(downNbr_delta,np.transpose(downNbrs_theta))
                    
                    out = outputs[t]
                    delta = delta * out * (1-out)
                    deltas[t]= delta
                    input = None
                    if t>0:
                        input = outputs[t-1]
                    else:
                        input = x_batch
                    theta = layers[t]
                    thetaT = theta + eta* np.matmul(np.transpose(input),delta) 
                    layersTemp[t] = thetaT
            
            layers = layersTemp
        if abs(j-j_prev) <= con_limit:
            break
        else:
            j_prev = j    


    acc1 = 100*findAccuracy(layers,x,y1,"out_train.txt")
  
    x2 = np.load(testx)
    m,n,k = np.shape(x2) 
    x2 = np.reshape(x2,(m,n*k))
    y2 = np.load(testy)
    x2 = np.hstack((x2,np.ones((m,1))))
    
    acc2 = 100*findAccuracy(layers,x2,y2,"out_test.txt")
    print(acc1,acc2)
    duration = (time.time()-start)/60
    return acc1,acc2,duration


acc_test = []
acc_train = []
times = []
sizes =  [1,10,50,100,500]

for s in sizes:
    acc1,acc2,dur = neuralNet(sys.argv[1],\
        sys.argv[2],\
        sys.argv[3],\
        sys.argv[4],\
        sys.argv[5],\
        100,784,[s,10],10,0.5)
    acc_train.append(acc1)
    acc_test.append(acc2)
    times.append(dur)

import matplotlib.pyplot as plt
plt.plot(sizes,acc_train,"-b",label = "ACC-Train")
plt.plot(sizes,acc_test,"-g",label = "ACC-Train")
plt.legend(loc="lower right")
plt.savefig('full.png')


# import matplotlib.pyplot as plt1
# plt1.plot(sizes,times,"-r", label="Time")
# plt1.legend(loc="lower right")
# plt1.savefig('full_time.png')


# Mini-Batch Size (M)
# • Number of features/attributes (n)
# • Hidden layer architecture: List of numbers denoting the number of perceptrons in the corresponding hidden layer. Eg. a list [100 50] specifies two hidden layers; first one with 100 units and second
# one with 50 units.
# • Number of target classes (r)

# 10.0 10.0
# 83.885 73.32
# 97.74000000000001 90.31
# 96.81 90.22
# 97.65666666666667 91.33