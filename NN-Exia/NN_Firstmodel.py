import numpy as np

class NN:

    def __init__(self):
        #with param
        self.X = np.array([[2, 1],
                           [-1, 1],
                           [-1, -1],
                           [1, -1]])
        self.t = np.array([0,1,2,3])
        np.random.seed(1)
        self.input_Dim = self.X.shape[1]
        self.num_classes = self.t.shape[0]
        self.hidden_layers = 50
        self.reg = 0.001
        self.epsilon = 0.001
        self.generations = 10000
        self.W1 = np.random.randn(self.input_Dim, self.hidden_layers)
        self.W2 = np.random.randn(self.hidden_layers, self.num_classes)
        self.b1 = np.zeros((1, self.hidden_layers))
        self.b2 = np.zeros((1, self.num_classes))

    def affine_forward(self, x, w, b):
        out = None
        #N为输入变量的维度
        N = x.shape[0]
        #将输入变量变成（维度，数据）的形式
        x_row = x.reshape(N, -1)
        #输入变量通过隐含层
        out = np.dot(x_row, w) + b
        #储存这一次的变量
        cache = (x, w, b)
        return out, cache

    def affine_backward(self, dout, cache):
        x, w, b = cache
        dx, dw, db = None, None, None
        #下一轮的x,对out求x的导数为w
        dx = np.dot(dout, w.T)
        dx = np.reshape(dx, x.shape)
        x_row = x.reshape(x.shape[0], -1)
        dw = np.dot(x_row.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        return dx, dw, db

    def fit(self):
        for i in range(self.generations):
            #前向传播
            H, fc_cache = self.affine_forward(self.X, self.W1, self.b1)
            H = np.maximum(0, H)
            relu_cache = H
            Y, cachey = self.affine_forward(H, self.W2, self.b2)
            #Softmax
            probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            #Loss
            N = Y.shape[0]
            print(probs[np.arange(N), self.t])
            loss = -np.sum(np.log(probs[np.arange(N), self.t])) / N
            print(loss)
            #反向传播
            dx = probs.copy()
            dx[np.arange(N), self.t] -= 1
            dx /= N
            dh1, dW2, db2 = self.affine_backward(dx, cachey)
            dh1[relu_cache <= 0] = 0
            dX, dW1, db1 = self.affine_backward(dh1, fc_cache)
            #参数更新
            dW2 += self.reg * self.W2
            dW1 += self.reg * self.W1
            self.W2 += -self.epsilon * dW2
            self.b2 += -self.epsilon * db2
            self.W1 += -self.epsilon * dW1
            self.b1 += -self.epsilon * db1

    def predict(self):
        test = np.array([[2, 2],
                        [-2, 2],
                        [-2, -2],
                        [2, -2]])
        H, fc_cache = self.affine_forward(test, self.W1, self.b1)
        H = np.maximum(0, H)
        relu_cache = H
        Y, cachey = self.affine_forward(H, self.W2, self.b2)
        probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        print(probs)
        for k in range(4):
            print(test[k,:], '所在象限为', np.argmax(probs[k,:])+1)


if __name__ == "__main__":
    NN_model = NN()
    NN_model.fit()
    NN_model.predict()