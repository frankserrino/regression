
"""
Created on Fri Nov 27 21:49:39 2020

@author: Frank Serrino

Much of this code is a modified from the material in Amit Yadav's excellent Coursera course 'Linear Regression with Python' 
"""
import numpy as np
from numpy import savetxt
np.random.seed(1)
np.set_printoptions(precision=8)

# set all the variables here:

number_of_data_points = 1000
parameters = 4  # not including intercept as a parameter here
intercept = 0.5
learn_rate = 3e-3
number_model_iterations = 30001
print_every_nth = 100

def generate_examples(num=number_of_data_points):
    W_list = [1.0]
    for i in range(parameters-1):
        W_list.append(W_list[i]*-1)
    W = np.asarray(W_list)
    print(W)
    b = intercept
    W = np.reshape(W, (W.shape[0], 1))
    X = np.random.randn(num,parameters)
    y_no_noise = b + np.dot(X, W)
    rand_y = np.random.randn(num,1)
    y_with_noise = np.add(y_no_noise, rand_y)
    y_with_noise= np.reshape(y_with_noise, (num, 1))
    return X, y_with_noise, y_no_noise

class Model:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.randn(num_features, 1)
        self.b = np.random.randn()
 
    def forward_pass(self, X):
        y = self.b + np.dot(X, self.W)
        return y

    def backward_pass(self, X, y_true, y_hat):
        m = y_hat.shape[0]
        db = np.sum(y_hat - y_true)/m
        dW = np.sum(np.dot(np.transpose(y_hat - y_true), X), axis=0)/m
        return dW, db  

    def update_params(self, dW, db, lr):
        self.W = self.W - lr * np.reshape(dW, (self.num_features, 1))
        self.b = self.b - lr * db
        
    def compute_loss(self, y, y_true):
        loss = np.sum(np.square(y - y_true))
        return loss/(2*y.shape[0])             
 
    def train(self, x_train, y_train, iterations, lr):
        losses = []
        for i in range(iterations):
            y_hat = self.forward_pass(x_train)
            dW, db = self.backward_pass(x_train, y_train, y_hat)
            self.update_params(dW, db, lr)
            loss = self.compute_loss(y_hat, y_train)
            losses.append(loss)
            if i % print_every_nth == 0:
                print('kth-iter: {}, loss: {:.8f}'.format(i/1000, loss), 'b: {:.8f}'.format(model.b), ' W:', np.reshape(model.W,(1,self.num_features)))
     #       print('kth-iter: {}, loss: {:.6f}'.format(i/1000, loss), 'b: {:.6f}'.format(model.b), ' W:', np.reshape(model.W,(1,self.num_features)))

        return losses   

        
X, y_noisy,y_without_noise = generate_examples()
model = Model(parameters)
losses=model.train(X, y_noisy, number_model_iterations, learn_rate) 


test_regress_file = X
test_regress_label = y_noisy
savetxt('/Users/fs/daata/test2_regress_file.csv', test_regress_file, delimiter=',')
savetxt('/Users/fs/daata/test2_regress_label.csv', test_regress_label, delimiter=',')

    
