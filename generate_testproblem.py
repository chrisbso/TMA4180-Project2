from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from evaluate_f_gradf import *
import numpy as np


def main():
    #plot_test()
    n = 2; #dimension
    m = 50; #points
    evaluate_test_m2(n,m)

## function to test, from Project 1
def evaluate_test_m2(n,m):
    # generate A,b to make points
    A = generate_rnd_PD_mx(n)
    b = generate_rnd_b_c(n)

    # generate points with w_i's
    (z,w) = generate_rnd_points_m2(A,b,m)

    # plot ellipsoid (A,b) and points z,w (if in 2D)
    if n == 2:
        plot_ellipsoid_m2(A, b, z, w)
        plt.title(r'Model 2: $\hat{S}(A,b)$ with points and labels')
        plt.show()

    # generate another (A,b)
    A = generate_rnd_mx(n)
    b = generate_rnd_b_c(n)

    # plot new ellipsoid with the "old" points z,w (if in 2D)
    if n == 2:
        plot_ellipsoid_m2(A,b,z,w)
        plt.title(r'Model 2: New $\hat{S}(A,b)$ with previous points and labels')
        plt.show()

    # generate random direction
    p = np.random.randn(int(n * (n + 1) / 2 + n))
    f0 = evaluate_f_m2(z, w, A, b)
    g = evaluate_grad_f_m2(z, w, A, b).dot(p)
    (A_p, b_p) = convert_x_To_A_and_c(p)
    # compare directional derivative with finite differences
    eps = 10.0 ** np.arange(-1, -13, -1)
    error_vec = np.zeros(len(eps))
    print('Model 2: compare directional derivative with finite differences')
    i = 0
    for ep in eps:
        g_app = (evaluate_f_m2(z, w, A + ep * A_p, b + ep * b_p) - f0) / ep
        # print(g_app)
        error_vec[i] = abs(g_app - g) / abs(g)

        print('ep = %e, error = %e' % (ep, error_vec[i]))
        i += 1
    plt.figure()
    plt.loglog(eps, error_vec,'ro-')
    plt.xlabel('Steplength',fontsize = 12)
    plt.ylabel('Relative error',fontsize = 12)
    plt.title('Model 2: Relative error between directional derivative and finite difference approximation',fontsize = 10)
    plt.figtext(0.65,0.15,'n = ' + str(n) + ', m = ' + str(m),fontsize = 12)
    plt.show()
    #print(evaluate_grad_f_m2(z, w, A, b))

#to generate random parameter b for the model
def generate_rnd_b_c(n):
    return n*np.random.rand(n)-n/2

#same functionality as Phi
def convert_x_To_A_and_c(x):

    dim = x.shape[0];
    n = int(-3/2+np.sqrt(9/4+2*dim))
    A = np.zeros([n,n])
    l=0;
    for i in range(n):
        for j in range(i,n):
            A[i][j] = x[l]
            A[j][i] = x[l]
            l += 1;
    c = x[int(dim-n):]
    return A,c

##Model 2
##################################################################################################
##plots the 0-level contour
def plot_ellipsoid_m2(A,b,z,w):
    assert(A.shape[0] == 2) #make sure we're in 2d
    assert(b.shape[0] == 2)
    assert(z.shape[0] == 2)

    maxPlotLimit = max(np.max(z),np.abs(np.min(z)));
    maxPlotLimit *= 1.5

    delta = 0.01
    z1 = z2 = np.arange(-maxPlotLimit, maxPlotLimit, delta)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.ones_like(Z1)
    for i in range(len(z1)):
        for j in range(len(z1)):
            z_i = [z1[i], z2[j]]
            Z[j][i] = (np.dot(z_i, np.dot(A, z_i)) + np.dot(b, z_i)) - 1
    plt.figure()
    plt.plot(np.take(z[0, :], np.where(w == 1)[0]), np.take(z[1, :], np.where(w == 1)[0]), 'ro')
    plt.plot(np.take(z[0, :], np.where(w == -1)[0]), np.take(z[1, :], np.where(w == -1)[0]), 'bo')

    plt.contour((Z1),(Z2),Z,0)
    plt.xlim(-maxPlotLimit, maxPlotLimit)
    plt.ylim(-maxPlotLimit, maxPlotLimit)
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')

#generate random points given A,b
def generate_rnd_points_m2(A, b, m):
    n = A.shape[0]
    z = np.zeros([n, m])
    w = np.ones(m)
    for i in range(m):
        z[:, i] = ((np.max(np.sqrt(np.abs(np.linalg.eigvals(A))))*np.random.rand(n))-0.5*np.max(np.sqrt(np.abs(np.linalg.eigvals(A)))))/1
        z_i = z[:, i]
        if ((np.dot(z_i, np.dot(A, z_i))+ np.dot(z_i,b))-1) >= 0:
            w[i] = -1.0
    return z, w

#generate a random matrix given condition in a string. See algos_2.py for usage.
def generate_rnd_mx(*args):
    if type(args[1]) is str and type(args[0]) is int:
        if args[1] == 'PD':
            A = generate_rnd_PD_mx(args[0]) #generate random positive definite mx
        elif args[1] == 'indef':
            A = generate_rnd_indef_mx(args[0]) #generate random indefinite mx
        elif (args[1] == 'own' or args[1] == 'symPts') and len(args) == 3:
            A = args[2]
        elif args[1] == "limit" and len(args) == 3:
            A = generate_matrix_limit(args[2]) #generate feasible mx with very small EVs.
        else:
            raise ValueError('generate_rnd_mx(): "str" is either "PD", "indef", "own" or "limit. "limit and "own" require a third arguement. Try again.')
    else:
         raise ValueError('"str" must be a string. Try again.')
    return A

#Generates a random PD mx
def generate_rnd_PD_mx(n):
    alpha = 0.2 # to guarantee our matrix is PD and not PSD.
    A = np.random.rand(n, n) # A is now random n x n matrix
    A = np.matmul(A,A.transpose())/0.3# A is now PSD
    A = A+alpha*np.identity(n)
    #  A is now PD
    return A

#generate an indefinite matrix
def generate_rnd_indef_mx(n):
    A = np.random.rand(n,n)
    A = (A+A.transpose())/0.3
    if np.linalg.det(A) > 0:
        #if it's not already indefinite, make it diagonally dominant.
        A[0,0] = (-abs(A[0,0]) - abs(A[0,1]))
        A[1,1] = (abs(A[1,1]) + abs(A[1,0]))
    print(np.linalg.det(A))
    return A

def generate_symmetric_points(a):
    '''
    generate a test problem in which points are aligned in a 5-point star.
    like this: (* <=> w_i == 1, X <=> w_i == -1)
        *
    *   X   *
        *
    '''
    w = np.ones(5)
    w[0] = -1 #we don't want the middle point in.
    z = np.zeros([2,5]);
    z[:,0] = [0,0] #middle
    z[:,1] = [a,0] #east
    z[:,2] = [-a,0]#west
    z[:,3] = [0,a] #north
    z[:,4] = [0,-a]#south
    return z,w

#generate a matrix with very small eigenvalues.
def generate_matrix_limit(lambdas):
    deltaMax = lambdas[1] - lambdas[0];
    z = np.random.rand(3);
    A11 = z[0]*deltaMax
    A22 = z[1]*deltaMax
    A12 = (2*z[2]-1)*np.sqrt(2*lambdas[0]*deltaMax+deltaMax**2)
    A = np.array([[A11,A12],[A12,A22]])
    return A


if __name__ == "__main__":
    main()




