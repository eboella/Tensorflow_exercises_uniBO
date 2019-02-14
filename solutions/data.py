import numpy as np
import matplotlib.pyplot as plt

#plt.set_cmap(plt.cm.bwr)

#import matplotlib as mpl
#mpl.rcParams['image.cmap'] = 'bwr'

num_data = 300
std_dev = 0.13
center = 0.5

class_t = np.intp

def get_spiral(num_data):

    num_class = 3

    X = np.zeros((num_data * num_class, 2))
    y = np.zeros(num_data * num_class, dtype=class_t)

    for k in range(num_class):
        ix = range(num_data * k, num_data * (k + 1))
        radius = np.linspace(0, 1, num_data)
        theta = np.linspace(k * 4, (k + 1) * 4, num_data) + np.random.randn(num_data) * 0.2
        X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[ix] = k

    y = y.reshape(num_data*num_class, 1)

    return X, y


def get_circles(num_data, factor=0.5, noise=0.05):

    linspace = np.linspace(0, 2 * np.pi, num_data, endpoint=False)

    outer_circ_x = np.cos(linspace)
    outer_circ_y = np.sin(linspace)
    inner_circ_x = np.cos(linspace) * factor
    inner_circ_y = np.sin(linspace) * factor

    X = np.vstack((np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(num_data, dtype=class_t), np.ones(num_data, dtype=class_t)]).reshape(2*num_data, 1)

    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

def get_data(center, std_dev, num_data, dim=1, offset=None, stress=None):
    d = []
    for i in range(dim):
        off = 1.0
        sts = 1.0
        if offset is not None:
            off = offset[i]
        if stress is not None:
            sts = stress[i]
        d.append(np.random.normal(center * off, std_dev * sts, num_data))
    return d

def get_data2(center, std_dev, num_data, dim=1, offset=None, stress=None):
    x = np.zeros((num_data, dim))
    for i in range(dim):
        off = 1.0
        sts = 1.0
        if offset is not None:
            off = offset[i]
        if stress is not None:
            sts = stress[i]
        x[:,i] = np.random.normal(center * off, std_dev * sts, num_data)
    return x


data1D = {}

data1D['x'] = get_data(center, std_dev, num_data, dim=1)
data1D['x_oblate'] = get_data(center, std_dev, num_data, dim=1, stress=[0.1])

data = {}

data['bl'] = get_data(center, std_dev, num_data, dim=2, offset=[-1, -1])
data['tr'] = get_data(center, std_dev, num_data, dim=2, offset=[+1, +1])
data['tl'] = get_data(center, std_dev, num_data, dim=2, offset=[-1, +1])
data['br'] = get_data(center, std_dev, num_data, dim=2, offset=[+1, -1])

data['br_sx'] = get_data(center, std_dev, num_data, dim=2, offset=[+1, -1], stress=[4.0, 1.0])
data['br_sy'] = get_data(center, std_dev, num_data, dim=2, offset=[+1, -1], stress=[1.0, 4.0])


def get_2bubbles(num_data):

    dim = 2

    a = get_data2(center, std_dev, num_data, dim=dim, offset=[-1, -1])
    b = get_data2(center, std_dev, num_data, dim=dim, offset=[+1, +1])

    X = np.append(a, b).reshape(2*num_data, dim)
    y = np.hstack([np.zeros(num_data, dtype=class_t), np.ones(num_data, dtype=class_t)]).reshape(2*num_data, 1)

    return X, y


def get_2bubbles_oblate(num_data):

    dim = 2

    a = get_data2(center, std_dev, num_data, dim=dim, offset=[-0.1, -1], stress=[2.5, 1.0])
    b = get_data2(center, std_dev, num_data, dim=dim, offset=[+1, +1])

    X = np.append(a, b).reshape(2*num_data, dim)
    y = np.hstack([np.zeros(num_data, dtype=class_t), np.ones(num_data, dtype=class_t)]).reshape(2*num_data, 1)

    return X, y


def get_2bubbles_prolate(num_data):

    dim = 2

    a = get_data2(center, std_dev, num_data, dim=dim, offset=[-1, -0.1], stress=[1.0, 2.5])
    b = get_data2(center, std_dev, num_data, dim=dim, offset=[+1, +1])

    X = np.append(a, b).reshape(2*num_data, dim)
    y = np.hstack([np.zeros(num_data, dtype=class_t), np.ones(num_data, dtype=class_t)]).reshape(2*num_data, 1)

    return X, y


def get_4bubbles(num_data):

    dim = 2

    a = get_data2(center, std_dev, num_data, dim=2, offset=[-1, -1])
    b = get_data2(center, std_dev, num_data, dim=2, offset=[+1, +1])
    c = get_data2(center, std_dev, num_data, dim=2, offset=[-1, +1])
    d = get_data2(center, std_dev, num_data, dim=2, offset=[+1, -1])

    aa = np.append(a, b)
    bb = np.append(c, d)

    X = np.append(aa, bb).reshape(4*num_data, dim)
    y = np.hstack([np.zeros(2*num_data, dtype=class_t), np.ones(2*num_data, dtype=class_t)]).reshape(4*num_data, 1)

    return X, y


if __name__ == '__main__':

    # 1D gaussian
    #graph = ['x', 'x_oblate']
    #for k in graph:
    #    plt.scatter(np.zeros(num_data), data1D[k])
    #plt.show()

    # 2D gaussian
    #graph = ['bl', 'tr', 'tl', 'br']
    #graph = ['br_sx', 'br_sy']
    #for i, k in enumerate(graph):
    #    d = data[k]
    #    plt.scatter(d[0][:], d[1][:])

    # circle
    #x, y = get_circles(num_data)
    #plt.scatter(x[:,0], x[:,1], c=y)

    # spiral
    #x, y = get_spiral(num_data)
    #plt.scatter(x[:,0], x[:,1], c=y)

    # 2 bubbles
    x, y = get_2bubbles(num_data)
    plt.scatter(x[:,0], x[:,1], c=y)

    # 2 bubbles oblate
    #x, y = get_2bubbles_oblate(num_data)
    #plt.scatter(x[:,0], x[:,1], c=y)

    # 2 bubbles prolate
    #x, y = get_2bubbles_prolate(num_data)
    #plt.scatter(x[:,0], x[:,1], c=y)

    # 4 bubbles
    #x, y = get_4bubbles(num_data)
    #plt.scatter(x[:,0], x[:,1], c=y)

    plt.show()

