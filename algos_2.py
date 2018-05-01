import numpy as np
import scipy.optimize

from evaluate_f_gradf import *
from generate_testproblem import *


def phi(x, dim):
    ''' Takes the vector x with entries of A and b or A and c and gives back
    the corresponding matrix A and the vector b/c. Needs as second parameter
    the dimension.
    '''
    matrix = np.zeros([dim, dim], dtype=np.double)
    vector = np.zeros(dim)
    start = 0
    k = dim - 1
    j = dim - 1

    # find the right entries
    for i in range(dim):

        if i == dim - 1:
            matrix[i, i] = x[start]
        else:
            # add 1 for considering also (i+dim-1-i)-th entry
            matrix[i, i:] = x[start: k + 1]
            start = k + 1
            k = k + j
            j -= 1
    # use symmetry of A to construct it
    diagonal = matrix[np.diag_indices(dim)]
    matrix = matrix + matrix.T
    matrix[np.diag_indices(dim)] -= diagonal
    vector = x[int((dim * (dim + 1)) / 2):]
    return matrix, vector


def phi_inv(matrix, vector):
    '''Takes a matrix and a vector and constructs the vector x with their
    entries under consideration of the symmetry of the matrix.
    '''
    n = len(vector)
    x_vector = np.zeros(int(0.5 * (n * (n + 1)) + n))
    start = 0
    k = n - 1
    j = n - 1
    # fill the vector x with entries
    for i in range(n):

        if i == n - 1:
            x_vector[start] = matrix[i, i]
        else:
            # add 1 for considering also (i+dim-1-i)-th entry
            x_vector[start: k + 1] = matrix[i, i:]
            start = k + 1
            k = k + j
            j -= 1

    x_vector[int((n * (n + 1)) / 2):] = vector
    return x_vector


def steepest_descent(bound, x, my):

    while np.linalg.norm(grad_P(x, my), 2) > bound:

        direction = -grad_P(x, my)

        alpha = armijo_stepsize_modded(x, my, direction)

        x += alpha * direction

        print(alpha, np.linalg.norm(grad_P(x, my), 2),func_f(x),x)

    return x


def armijo_stepsize_modded(x_old, my,  d, delta=.75, gamma=1, beta=.5):

    value_old = func_P(x_old, my)

    sigma = -gamma*np.dot(grad_P(x_old,my),d/(np.linalg.norm(d,2)**2))

    x_new = x_old + sigma*d

    for c_i in constraints():
        while c_i(x_new) <= 0:
            sigma *= .5
            x_new = x_old + sigma*d

    while func_P(x_new, my) > value_old + delta * sigma * np.dot(grad_P(x_old, my),d):
        sigma *= beta
        x_new = x_old + sigma * d

    #print(sigma)

    return sigma


def tau(n): return 1/2**n


def barrier_method(my, x_init, k=1):

    print(x_init)
    print(np.linalg.norm(grad_f(x_init), 2) )
    while np.linalg.norm(grad_f(x_init), 2) > 10**-1: # does this work?????
        x_init = steepest_descent(tau(k), x_init, my)
        my *= .5; k += 1
        print(x_init)
        print('\n ---------------------------------------------')


# np.log = nan exception?
# phi dim?

def c1(x): return x[0] - l_min


def c2(x): return l_max - x[0]


def c3(x): return x[2] - l_min


def c4(x): return l_max - x[2]


def c5(x): return (x[0]*x[2])**.5 - (l_min**2+x[1]**2)**.5


def constraints(): return [c1, c2, c3, c4, c5]


def gradc1(x): return [1, 0, 0, 0, 0]


def gradc2(x): return [-1, 0, 0, 0, 0]


def gradc3(x): return [0, 0, 1, 0, 0]


def gradc4(x): return [0, 0, -1, 0, 0]


def gradc5(x): return [.5*((x[0]*x[2])**(-.5))*x[2], x[1]*(l_min**2+x[1]**2)**(-.5), .5*((x[0]*x[2])**(-.5))*x[0], 0, 0]


def gradconstraints(): return [gradc1, gradc2, gradc3, gradc4, gradc5]


def func_f(x):
    A, b = phi(x,2)
    return evaluate_f_m2(z, w, A, b)


def grad_f(x):
    A, b = phi(x,2)
    return evaluate_grad_f_m2(z, w, A, b)


def func_P(v_1, v_2):
    return func_f(v_1) - v_2 * sum([np.log((c_i(v_1))) for c_i in constraints()])


def grad_P(v_1, v_2):
    return grad_f(v_1) - v_2 * np.array([sum([gc_i(v_1)[q]*(1/c_i(v_1)) for gc_i, c_i in zip(gradconstraints(), constraints())]) for q in range(5)])


def create_rd_x_initial():
    x = np.random.rand(5)
    x[0] = x[0] * (l_max - l_min) + l_min
    x[1] += -1
    x[2] = x[2] * (l_max - l_min) + l_min
    return x


def call_for_help(z,w):
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 0.1},
            {'type': 'ineq', 'fun': lambda x: 10 - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[2] - 0.1},
            {'type': 'ineq', 'fun': lambda x: 10 - x[0]},
            {'type': 'ineq', 'fun': lambda x: (x[0]*x[2])**.5 - (.1**2+x[1]**2)**.5})

    func_mod = lambda x, z, w: evaluate_f_m2(z, w, phi(x,2)[0], phi(x,2)[1])

    result = scipy.optimize.minimize(func_mod, create_rd_x_initial(), (z,w), constraints=cons, tol=10**-6)

    print(result)

    print(func_f(result.x))
    print(result.x)


if __name__ == "__main__":

    temp = 0

    global z
    global w
    global l_min
    global l_max

    l_min = 0.1
    l_max = 10

    my_x = create_rd_x_initial()

    A, b = phi(my_x,2)
    A = generate_rnd_ND_mx(2)
    (z, w) = generate_rnd_points_m2(A, b, 100)
    plot_ellipsoid_m2(A,b,z,w)
    plt.show()
'''
    while sum(w) == -200:
        (z, w) = generate_rnd_points_m2(A, b, 200)
        temp += 1; print(temp)


    barrier_method(.001, create_rd_x_initial())

    #call_for_help(z,w)

    print(my_x)
'''
