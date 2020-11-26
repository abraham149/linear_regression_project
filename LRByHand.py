

class LinearRegressionByHand():
    def __init__(self, learning_rate = 0.001, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def compute_loss(self, y, y_true):
        loss = np.sum( (y - y_true)**2  )
        return loss/(2*y.shape[0])
    
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

    def fit(self, x_train, y_train):
        y_train = y_train.values.reshape(-1,1)
        self.m, self.n = x_train.shape # m is # of rows. n is # of features
        self.W = np.random.randn(self.n, 1) # params
        self.b = np.random.randn() # bias
        losses = []
        for i in range(self.iterations):
            y_hat = self.predict(x_train)
            loss = self.compute_loss(y_hat, y_train)
            
            dW, db = self.gd(x_train, y_train, y_hat) 
            
            self.update_params(dW, db)
            
            losses.append(loss)
            if i % int(self.iterations/10) == 0:
                print('Iter: {}, Current loss: {:.4f}'.format(i, loss))
        return self
