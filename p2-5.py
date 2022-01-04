from sklearn.neural_network import MLPClassifier
import numpy as np
import sys

def neuralNet(trainx,trainy,testx,testy,output):
    x = np.load(trainx)
    y = np.load(trainy)
    m,n,k = np.shape(x) 
    x = np.reshape(x,(m,n*k))
    m,n = np.shape(x)
    clf = MLPClassifier(hidden_layer_sizes=[100,100], activation='relu').fit(x, y)
    
    x_test = np.load(testx)
    m1,n1,k1 = np.shape(x_test) 
    x_test = np.reshape(x_test,(m1,n1*k1))
    m1,n1 = np.shape(x_test)
    y_test = np.load(testy)
    
    y_pred = clf.predict(x)
    # print(y_pred)
    score=0
    for i in range(m):
        if y_pred[i] == y[i]:
            score+=1
    print("train acc",score/m)
    
    
    y_pred = clf.predict(x_test)
    # print(y_pred)
    score=0
    for i in range(m1):
        if y_pred[i] == y_test[i]:
            score+=1
    print("test acc",score/m1)


def main(): 
    neuralNet(sys.argv[1],\
        sys.argv[2],\
        sys.argv[3],\
        sys.argv[4],\
        sys.argv[5])

main()

# train acc 99.94333333333333
# test acc 94.18