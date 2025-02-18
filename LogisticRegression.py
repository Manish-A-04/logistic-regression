import numpy as np


class LogisticRegression():

    def __init__(self , lr=0.0001 , iters = 2000):

        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self , X , Y):
        n_samples  , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):

            linear_grad = np.dot(X , self.weights) + self. bias

            predictions = self._sigmoid(linear_grad)

            dw = (1/n_samples) * np.dot(X.T , (predictions - Y) )
            db = (1/n_samples) * np.sum(predictions - Y)

            self.weights = self.weights -self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, x):
        linear_pred = np.dot(x , self.weights)+self.bias
        y_pred = self._sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred 

    def _sigmoid(self ,x):
        value = 1/(1+np.exp(-x))
        return value

from sklearn.datasets import load_breast_cancer

if __name__=="__main__":
    model = LogisticRegression()
    x , y = load_breast_cancer(return_X_y=True)
    model.fit(x , y)
    pred = model.predict(x)

    print(np.sum(pred == y)/y.shape)