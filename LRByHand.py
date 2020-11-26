import numpy as np


class LinearRegressionByHand():
    def __init__(self, learning_rate = 0.001, iterations = 100000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def compute_loss(self, y, y_true):
        loss = np.sum( (y - y_true)**2  )
        return loss / ( 2 * self.n )
    
    def predict(self, X):  #returns y = b + p1x1 + p2x2 + ... + pnxn
        y = self.b + X @ self.W
        return y
        
    def gd(self, X, y_true, y_hat):   #Compute the derivatives from the loss function
        db = np.sum(y_hat - y_true) / self.m
        dW = np.sum(  (y_hat - y_true).T @ X, axis=0)/ self.m
        return dW, db
    
    def update_params(self, dW, db):
        self.W = self.W - self.learning_rate * np.reshape(dW, (self.n, 1))
        self.b = self.b - self.learning_rate * db

    def fit(self, X, y):
        y = y.values.reshape(-1,1)
        self.m, self.n = X.shape # m is # of rows. n is # of features
        self.W = np.random.randn(self.n, 1) # init random params
        self.b = np.random.randn() # bias
        self.losses = []
        for _ in range(self.iterations):
            y_hat = self.predict(X)
            loss = self.compute_loss(y_hat, y)
            self.losses.append(loss)
            dW, db = self.gd(X, y, y_hat) 
            self.update_params(dW, db)
        return self.losses
