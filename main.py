import scipy.optimize as opt
import numpy, random, math
import matplotlib.pyplot as plt


def objective(alpha):
    result = numpy.dot(alpha, Pmatrix)
    return 0.5 * numpy.dot(result, alpha) - numpy.sum(alpha)


sigma = 1


def kernel_function(x1, x2):
    dot = numpy.dot(x1, x2)
    res = pow((dot + 1), 2)
    exp = math.exp(- pow(numpy.linalg.norm(x1-x2), 2)/(2 * pow(sigma, 2)))
    return res


def zerofun(vector):
    return numpy.dot(vector, targets)


# generating test data
numpy.random.seed(100)
classA = numpy.concatenate((numpy.random.randn(100, 2)*3.2 + [1.5, 0.5],
                            numpy.random.randn(10, 2)*3.2 + [-1.5, 0.5]))
classB = numpy.random.randn(200, 2) * 2.2 + [0.0, -0.5]
inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]
Pmatrix = numpy.zeros((N, N))
for i in range(N):
    for j in range(N):
        Pmatrix[i][j] = targets[i] * targets[j] * kernel_function(inputs[i], inputs[j])

start = numpy.zeros(N)
C = 10
B = [(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}


def indicator_function(x, y, nonzero_alphas, b_value):
    indicator = 0
    S = [x, y]
    for i in range(len(nonzero_alphas)):
        indicator += nonzero_alphas[i][1] * nonzero_alphas[i][2] * kernel_function(S, nonzero_alphas[i][3])
    indicator -= b_value
    return indicator

def main():
    ret = opt.minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']
    nonzero_alpha = []
    for i in range(N):
        if alpha[i] > 1e-5:
            nonzero_alpha.append([i, alpha[i], targets[i], inputs[i]])
    b_value = 0
    for i in range(N):
        b_value += alpha[i] * targets[i] * kernel_function(inputs[i], nonzero_alpha[0][3])
    b_value -= targets[nonzero_alpha[0][0]]
    xgrid = numpy.linspace(-15, 15)
    ygrid = numpy.linspace(-15, 15)
    grid = numpy.array([[indicator_function(x, y, nonzero_alpha, b_value) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')
    plt.savefig('svmplot.pdf')
    plt.show()


if __name__ == '__main__':
    main()

