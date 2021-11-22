import numpy as np
import matplotlib.pyplot as plt

std = 0.3
n1 = 100
cat1 = np.random.multivariate_normal([3, 0], [[std, 0], [0, std]], n1)
cat1 = np.hstack((cat1, np.full((n1, 1), 1)))

n2 = 100
cat2 = np.random.multivariate_normal([0, 0], [[std, 0], [0, std]], n2)
cat2 = np.hstack((cat2, np.full((n1, 1), 2)))

n3 = 150
linspace_out = np.linspace(0, 2 * np.pi, n3)
outer_circ_x = np.cos(linspace_out) * 5 + 1.5
outer_circ_y = np.sin(linspace_out) * 5

cat3 = np.vstack(
    [outer_circ_x, outer_circ_y]
).T

cat3 += np.random.uniform(-1, 1, size=cat3.shape)
cat3 = np.hstack((cat3, np.full((n3, 1), 3)))

data = np.vstack((cat1, cat2, cat3))
np.random.shuffle(data)

X = data[:, 0:2]
labels = data[:, 2]


# plt.scatter(cat1[:, 0], cat1[:, 1], label='cat1')
# plt.scatter(cat2[:, 0], cat2[:, 1], label='cat2')
# plt.scatter(cat3[:, 0], cat3[:, 1], label='cat3')
#
# plt.legend()


def f1_active(z_in):
    return (z_in > 0) * 1 + (z_in <= 0) * -1


def f2_active(y_in):
    return (y_in > 0) * 1 + (y_in <= 0) * -1


W1 = np.random.normal(0, 0.1, (8, 2))
b1 = np.random.normal(0, 0.1, (8, 1))

print(W1)

W2 = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 1]])

b2 = np.array([[4], [4]])

alpha = 0.1
number_of_epochs = 20
mses = []
squared_errors = []


def mse(X1, y1, W1, b1):
    z_in = np.dot(X1, W1.T) + b1.T
    Zs = []
    for z in z_in:
        Zs.append(f1_active(z))

    Zs = np.array(Zs)

    y_in = np.dot(Zs, W2.T) + b2.T
    ys = []
    for y in y_in:
        out = f2_active(y)
        label = 3
        if out[0] == 1 and out[1] == -1:
            label = 1
        elif out[0] == -1 and out[1] == -1:
            label = 2
        ys.append(label)
    return np.mean(ys == y1)


costs = []

for epoch in range(number_of_epochs):
    for s, t in zip(X, labels):

        z_in = np.dot(s, W1.T) + b1.T
        z1 = f1_active(z_in[0])

        y_in = np.dot(z1, W2.T) + b2.T
        y = f2_active(y_in[0])

        print(y[0], '-- ', y[1])
        if t == 1 and y[0] == -1:  # cat1
            z_min_arg = np.argmin(abs(z_in[0, 0:4]))
            W1[z_min_arg, :] = W1[z_min_arg, :] + alpha * (1 - z_in[0, z_min_arg]) * s
            b1[z_min_arg] = b1[z_min_arg] + alpha * (1 - z_in[0, z_min_arg])

        elif t == 1 and y[1] == 1:
            # print('111')
            z_positive_arg = np.hstack((np.array([False, False, False, False]), z_in[0, 4:8] > 0))
            W1[z_positive_arg, :] = W1[z_positive_arg, :] + alpha * np.dot(
                (-1 - z_in[0, z_positive_arg]).reshape((sum(z_positive_arg), 1)), s.reshape((1, len(s))))
            b1[z_positive_arg] = b1[z_positive_arg] + alpha * (-1 - z_in[0, z_positive_arg]).reshape(
                (sum(z_positive_arg), 1))

        elif t == 2 and y[1] == -1:  # cat 2
            print('cat2')
            z_min_arg = 4 + np.argmin(abs(z_in[0, 4:8]))
            W1[z_min_arg, :] = W1[z_min_arg, :] + alpha * (1 - z_in[0, z_min_arg]) * s
            b1[z_min_arg] = b1[z_min_arg] + alpha * (1 - z_in[0, z_min_arg])

        elif t == 2 and y[0] == 1:
            # print('222')
            z_positive_arg = np.hstack((z_in[0, 0:4] > 0, np.array([False, False, False, False])))
            W1[z_positive_arg, :] = W1[z_positive_arg, :] + alpha * np.dot(
                (-1 - z_in[0, z_positive_arg]).reshape((sum(z_positive_arg), 1)), s.reshape((1, len(s))))
            b1[z_positive_arg] = b1[z_positive_arg] + alpha * (-1 - z_in[0, z_positive_arg]).reshape(
                (sum(z_positive_arg), 1))

        elif t == 3 and (y[0] == 1 or y[1] == 1):  # cat3
            if y[0] == 1:
                z_positive_arg = np.hstack((z_in[0, 0:4] > 0, np.array([False, False, False, False])))
                W1[z_positive_arg, :] = W1[z_positive_arg, :] + alpha * np.dot(
                    (-1 - z_in[0, z_positive_arg]).reshape((sum(z_positive_arg), 1)), s.reshape((1, len(s))))
                b1[z_positive_arg] = b1[z_positive_arg] + alpha * (-1 - z_in[0, z_positive_arg]).reshape(
                    (sum(z_positive_arg), 1))

            if y[1] == 1:
                z_positive_arg = np.hstack((np.array([False, False, False, False]), z_in[0, 4:8] > 0))
                W1[z_positive_arg, :] = W1[z_positive_arg, :] + alpha * np.dot(
                    (-1 - z_in[0, z_positive_arg]).reshape((sum(z_positive_arg), 1)), s.reshape((1, len(s))))
                b1[z_positive_arg] = b1[z_positive_arg] + alpha * (-1 - z_in[0, z_positive_arg]).reshape(
                    (sum(z_positive_arg), 1))

    #     cost = mse(X, labels, W1, b1)
    #     costs.append(cost)
    #     if cost < 0.01:
    #         break
    #
    # if cost < 0.1:
    #     break

# plt.figure()
# plt.plot(costs)

plt.figure()
plt.scatter(cat1[:, 0], cat1[:, 1], label='cat1')
plt.scatter(cat2[:, 0], cat2[:, 1], label='cat2')
plt.scatter(cat3[:, 0], cat3[:, 1], label='cat3')
plt.legend()

x = np.linspace(-5, 7)

print(W1)

for i in range(4):
    y = -(W1[i, 0] / W1[i, 1]) * x - b1[i] / W1[i, 1]
    plt.plot(x, y, color='blue')

for i in range(4, 8):
    y = -(W1[i, 0] / W1[i, 1]) * x - b1[i] / W1[i, 1]
    plt.plot(x, y, color='red')

plt.show()
