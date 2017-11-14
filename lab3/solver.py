import numpy as np


class Solver:

    def __init__(self, model, data, learning_rate=1e-2):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.batch_size = 1000
        self.learning_rate = learning_rate

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)

        for p, w in self.model.params.items():
            dw = grads[p]
            self.model.params[p] -= self.learning_rate * dw

    def train(self, max_iters=1000, silent=False):
        val_acc_hist = []
        train_acc_hist = []
        for i in range(max_iters):
            if i % (max_iters//10) == 0:
                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                val_acc_hist.append(val_acc)
                train_acc_hist.append(train_acc)
                if not silent:
                    print("%d, train_acc: %f, val_acc : %f" % (i, train_acc, val_acc))
            self._step()
        return (train_acc_hist, val_acc_hist)

    def check_accuracy(self, X, y):
        y_pred = self.model.predict(X)
        acc = np.mean(y_pred == y)
        return acc
