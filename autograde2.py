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
    b = np.zeros((y.size,y.max()+1))
    b[np.arange(y.size),y] = 1
    
    return b

def printPrediction(layers,x,out_dir,act_fn):
    out = np.matmul(x,layers[0])
    if act_fn == "softmax":
        out = 1/(1+np.exp(-out))                # taking softmax
    else:
        out[out<=0] = 0
    for i in range(1,len(layers)):
        out = np.matmul(out,layers[i])
        if act_fn == "softmax" or i == len(layers)-1 :
            out = 1/(1+np.exp(-out))
        else:
            out[out<=0]=0
            # out[:][:] = max(0,out[:][:])
  
    m,n = np.shape(x)
    
    f = open(out_dir,"w")
    for i in range(m):
        l = np.argmax(out[i])
        print(l,file = f)
    f.close()
    
def neuralNet(trainx,trainy,testx,out_file,batch_size,architecture,act_fn):
    architecture = [int(i) for i in architecture.split()] # convert architectures into list
    x1 = np.load(trainx)
    m,n,k = np.shape(x1)
    no_features= n*k

    y1 = np.load(trainy)

    architecture.append(y1.max()+1)     # appending no of targets
    

    eta=0.5
    if act_fn == "relu":
        eta = 0.01
    

    max_epoch = 100
    con_limit = 0.0001
        
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
            if act_fn == "softmax":
                out = 1/(1+np.exp(-out))                # taking softmax
            else:
                out[out<=0] = 0
            c=np.copy(out)
            outputs.append(c)
            for i in range(1,len(layers)):
                out = np.matmul(out,layers[i])
                if act_fn == "softmax" or i == len(layers)-1:
                    out = 1/(1+np.exp(-out))
                else:
                    out[out<=0] = 0
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
                    if act_fn == "softmax":
                        delta = delta * out * (1-out)
                    else:
                        temp = np.copy(out)
                        temp[temp>0] = 1
                        temp[temp<=0] = 0
                        delta = delta * temp

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


    x2 = np.load(testx)
    m,n,k = np.shape(x2) 
    x2 = np.reshape(x2,(m,n*k))
    # y2 = np.load(testy)
    x2 = np.hstack((x2,np.ones((m,1))))
    
    printPrediction(layers,x2,out_file,act_fn)
 
 
neuralNet(sys.argv[1],\
    sys.argv[2],\
    sys.argv[3],\
    sys.argv[4],\
    sys.argv[5],sys.argv[6],sys.argv[7])

