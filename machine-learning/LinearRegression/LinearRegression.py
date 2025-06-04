
class LinearRegression:
    def __init__(self, features, target, epochs, lr, weight=0.01, bias=0):
        self.features = features
        self.target = target
        self.weight = [weight] * len(features[0])
        self.bias = bias
        self.epochs = epochs
        self.lr = lr
        
    
    def loss_function(self, Y_pred):
        m = len(self.features)
        loss = 1 / m
        summation = 0
        for i in range(m):
            summation += (Y_pred[i] - self.target[i]) ** 2
            
        return loss * summation
                
    def train(self):
        for i in range(self.epochs):
            print("epoch: ", i)
            Y_pred = []
            for j in self.features:
                y_pred = self.dot_product(j, self.weight) + self.bias
                Y_pred.append(y_pred)
            
            loss = self.loss_function(Y_pred)
            print("The loss is: ", loss)
            
            print("old weights: ", self.weight)
            print("old bias: ", self.bias)
            
            self.gradient_descent(Y_pred)
             
            print("new weights: ", self.weight)
            print("new bias: ", self.bias)
        
    def gradient_descent(self, Y_pred):
        m = len(self.features)
        loss_w = 2 / m
        loss_b = 2 / m
        
        n = len(self.weight)
        sum_w = [0] * n
        sum_b = 0
        for i in range(m):
            for j in range(n):
                sum_w[j] += (Y_pred[i] - self.target[i]) * self.features[i][j]
            sum_b += (Y_pred[i] - self.target[i])
            
        for i in range(n):
            self.weight[i] -= (self.lr * (loss_w * sum_w[i]))
        
        self.bias -= (self.lr * (loss_b * sum_b))
        
        return
    
    def dot_product(self, x, w):
        dot_prod = 0
        m = len(x)
        for i in range(m):
            dot_prod += x[i] * w[i]
        return dot_prod

    def predict(self, features):
        return self.dot_product(features, self.weight) + self.bias 
        
if __name__ == "__main__":
    data = [
        ([1.0, 2.0], 5.0),
        ([2.0, 1.0], 6.0),
        ([3.0, 3.0], 12.0),
        ([4.0, 5.0], 18.0),
        ([5.0, 4.0], 17.0),
        ([6.0, 7.0], 26.0),
        ([7.0, 6.0], 25.0),
        ([8.0, 9.0], 35.0),
        ([9.0, 8.0], 34.0),
        ([10.0, 10.0], 40.0),
    ]
    
    X = [x[0] for x in data]
    Y = [x[1] for x in data]
    model = LinearRegression(X, Y, 10, 0.1)
    model.train()
