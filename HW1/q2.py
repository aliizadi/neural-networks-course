import numpy as np
import matplotlib.pyplot as plt

n1 = 100
cat1 = np.vstack((np.random.normal(2, 0.5, n1), np.random.normal(0, 0.2, n1), np.ones(n1))).T

n2 = 30
cat2 = np.vstack((np.random.normal(0, 0.1, n2), np.random.normal(1, 0.7, n2), np.full((n2,), -1))).T

plt.scatter(cat1[:, 0], cat1[:, 1], label='cat1')
plt.scatter(cat2[:, 0], cat2[:, 1], label='cat2')

data = np.vstack((cat1, cat2))
np.random.shuffle(data)

X = data[:, 0:2]
y = data[:, 2]

# two for x and y and one for biaas
W = np.random.normal(0, 0.1, 2)
b = np.random.normal(0, 0.1)

alpha = 0.01


def mse(X1, y1, W1):
    def f(net):
        return 1 if net >= 0 else -1

    f = np.vectorize(f)
    net = np.dot(X1, W1.T)
    out = f(net)
    return 0.5 * np.mean((np.square(y1 - out)))


number_of_epochs = 20
mses = []
squared_errors = []
for epoch in range(number_of_epochs):
    for s, t in zip(X, y):
        net = np.dot(s, W.T)
        error = t - net
        W = W + alpha * error * s
        b = b + alpha * error

        squared_errors.append(0.5 * (error ** 2))
        cost = mse(X, y, W)
        mses.append(cost)

        if cost < 0.001:
            break

    if cost < 0.001:
        print('*** exit condition *** with cost: ', cost)
        break

plt.figure()
plt.plot(mses)
plt.title('mse')

plt.figure()
plt.plot(squared_errors)
plt.title('squared_errors')

plt.figure()
plt.scatter(cat1[:, 0], cat1[:, 1], label='cat1')
plt.scatter(cat2[:, 0], cat2[:, 1], label='cat2')

x = np.linspace(-1, 4)
y = -(W[0] / W[1]) * x - b / W[1]
plt.plot(x, y, color='red')
plt.show()

plt.legend()
