import numpy as np
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
        f.write(str(l))
        if l == y[i]:
            score+=1
    f.close()
    return score/m

def main(trainx,trainy,testx,testy,out_dir,batch_size,no_features,architecture,no_targets):
    x1 = np.load(trainx)
    y1 = np.load(trainy)

    layers = []
    architecture.insert(0,no_features)
    for i in range(1,len(architecture)):
        theta = 0.01*np.random.randn(architecture[i-1],architecture[i])
        layers.append(theta)
    
    m,n,k = np.shape(x1) 
    x = np.reshape(x1,(m,n*k))
    m,n = np.shape(x)
    y = one_hot_encode(y1)
    eta=0.1/m

    err = -1
    iter=0
    bb = False
    sum=0
    for epoch in range(150):
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
            sum+= (j/1000)
            if iter%1000 ==0:
                print(sum)
                if abs(sum-err)<0.0001:
                    bb = True
                    break
                else:
                    err = j
                    sum=0

            
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
            iter+=1
        if bb:
            break    

    x2 = np.load(testx)
    m,n,k = np.shape(x2) 
    x = np.reshape(x2,(m,n*k))
    y2 = np.load(testy)
    print("trainAcc:",findAccuracy(layers,x,y1,out_dir))



main(sys.argv[1],\
     sys.argv[2],\
     sys.argv[3],\
     sys.argv[4],\
     sys.argv[5],\
     100,784,[100,10],10)

# Mini-Batch Size (M)
# • Number of features/attributes (n)
# • Hidden layer architecture: List of numbers denoting the number of perceptrons in the corresponding hidden layer. Eg. a list [100 50] specifies two hidden layers; first one with 100 units and second
# one with 50 units.
# • Number of target classes (r)