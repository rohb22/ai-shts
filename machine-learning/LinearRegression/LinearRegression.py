
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
        summation = 0
        for i in range(m):
            summation += (Y_pred[i] - self.target[i]) ** 2
        return summation / m
                
    def train(self):
        for epoch in range(self.epochs):
            print("epoch:", epoch)
            Y_pred = []
            for x in self.features:
                y_pred = self.dot_product(x, self.weight) + self.bias
                Y_pred.append(y_pred)
            
            loss = self.loss_function(Y_pred)
            print("Loss:", loss)
            
            print("Old weights:", self.weight)
            print("Old bias:", self.bias)
            
            self.gradient_descent(Y_pred)
             
            print("New weights:", self.weight)
            print("New bias:", self.bias)
        
    def gradient_descent(self, Y_pred):
        m = len(self.features)
        n = len(self.weight)
        sum_w = [0] * n
        sum_b = 0
        
        for i in range(m):
            error = Y_pred[i] - self.target[i]
            for j in range(n):
                sum_w[j] += error * self.features[i][j]
            sum_b += error
            
        for i in range(n):
            self.weight[i] -= self.lr * (2 / m) * sum_w[i]
        
        self.bias -= self.lr * (2 / m) * sum_b
        
    def dot_product(self, x, w):
        dot_prod = 0
        for i in range(len(x)):
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
    model = LinearRegression(X, Y, 100, 0.001)
    model.train()
    print(model.predict([8.0,9.0]))
