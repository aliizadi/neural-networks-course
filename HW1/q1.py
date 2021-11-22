import matplotlib.pyplot as plt
import numpy as np

# plot lines
x = np.linspace(-1, 4)
y = 0 * x + 3
plt.plot(x, y, color='red')

x = np.linspace(2, 6)
y = (-5.0 / 2.0) * x + 21.0 / 2.0
plt.plot(x, y, color='red')

x = np.linspace(-2, 6)
y = 0 * x - 2
plt.plot(x, y, color='red')

x = np.linspace(-2, 1, 100)
y = 5 * x + 3

plt.plot(x, y, color='red')

plt.scatter([0, 3], [3, 3], color='blue')
plt.scatter([0, -1], [3, -2], color='blue')
plt.scatter([-1, 5], [-2, -2], color='blue')
plt.scatter([3, 5], [3, -2], color='blue')

# Mcculloch pitts

input = (-5, -5)


def active(x):
    if x >= 0:
        return 1
    else:
        return -1


active = np.vectorize(active)

X = np.array([input[0], input[1], 1])
W1 = np.array([[0, 1, -3],
               [5, 2, -21],
               [0, 1, 2],
               [5, -1, 3]])

h1 = np.dot(X, W1.T)
h1 = np.hstack((h1, np.array([1])))
h1 = active(h1)
print(h1)

W2 = np.array([[1, 1, 0, 0, -1],
               [0, 0, 1, 1, -1]])

h2 = np.dot(h1, W2.T)
h2 = active(h2)
print(h2)
h2 = np.hstack((h2, np.array([1])))

W3 = np.array([[-1, 1, -1]])

output = np.dot(h2, W3.T)

print(output)

plt.show()
