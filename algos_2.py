import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

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

        #print(np.linalg.norm(grad_P(x, my), 2), my)

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

    return sigma


def tau(n): return .5**n


def barrier_method(my, x_init, bound, k=1):

    x_good_sol = call_for_help(bound, x_init)

    conv_in_f = [abs(func_f(x_good_sol)-func_f(x_init))]

    #x_storage = np.zeros((1,len(x_init)))
    #x_storage=first.reshape((1, len(x_init)))
    x_storage=[np.array(x_init)]    

    t_storage = [1]

    while np.linalg.norm(grad_f(x_init), 2) > bound: 
        # feasible starting point, that is updated
      
        x_init = steepest_descent(tau(k), x_init, my)
        
        my *= .5
        k += 1
        #error development array
        conv_in_f.append(abs(func_f(x_good_sol)-func_f(x_init)))
        # development of x array
        
        x_storage.append(np.array(x_init))
        #x_storage=np.concatenate((x_storage, x_init.reshape((1, len(x_init)))), axis=0)
         # development of tau array
        t_storage.append(tau(k))
        #print(x_init)
        #print('\n ---------------------------------------------')
    
    return x_storage, conv_in_f, t_storage


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


def call_for_help(bound, x_input):
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 0.1},
            {'type': 'ineq', 'fun': lambda x: 10 - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[2] - 0.1},
            {'type': 'ineq', 'fun': lambda x: 10 - x[0]},
            {'type': 'ineq', 'fun': lambda x: (x[0]*x[2])**.5 - (.1**2+x[1]**2)**.5})

    func_mod = lambda x, z, w: evaluate_f_m2(z, w, phi(x,2)[0], phi(x,2)[1])

    result = scipy.optimize.minimize(func_mod, x_input, (z,w), constraints=cons, tol=bound)

    return result.x


def plot_convergence(conv_1, color_1, label1, y_axis, kind_of_plotting):
    '''Creates convergence plots over iterations.
    '''
    n=len(conv_1)
    grid = np.arange(0,n,1)
                   
    if(kind_of_plotting=='loglog'):  
        # plot using loglog scale
        plt.loglog(grid, conv_1,'-',label=label1, color=color_1)
        
    elif(kind_of_plotting=='plot'):
        # using normal plotting scale, without first entry
        plt.plot(grid[1:], conv_1[1:],'-',label=label1, color=color_1)
        
    else:
        print('undefined kind of ploting scale! Enter \'loglog\' or \'plot\'.')
    
    plt.xlabel("iterations")
    plt.ylabel(y_axis)
    #plot legend
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2)
    # show also the legend and title
    plt.tight_layout(pad=7)
    
def plot_ellipses(creation, array,z):
    ''' Plots the ellipses of model 2. Takes an array with the stored x-vectors
    during the iterations and the set of points z and the corresponding labels.
    '''
    maxPlotLimit = max(np.max(z),np.abs(np.min(z)));
    maxPlotLimit *= 1.2

    delta = 0.01
    z1 = z2 = np.arange(-maxPlotLimit, maxPlotLimit, delta)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.ones_like(Z1)
    
    
    for r in range(array.shape[0]):
        matrix, vector=phi(array[r,:],2)
        
        for i in range(len(z1)):
            for j in range(len(z1)):
                z_i = [z1[i], z2[j]]
                Z[j][i] = (np.dot(z_i, np.dot(matrix, z_i)) + np.dot(vector, z_i)) - 1
        
        #plot level sets
        if(r==0):
            ellipse_num=plt.contour(Z1,Z2,Z,0, colors=('k'), linewidths=0.5)
            path_num=ellipse_num.collections[0].get_paths()[0] 
            xy_num = path_num.vertices
        else:
            plt.contour(Z1,Z2,Z,0, colors=('g'), linewidths=0.5)

    m,v=phi(array[r,:],2)
        
    for i in range(len(z1)):
        for j in range(len(z1)):
            z_i = [z1[i], z2[j]]
            Z[j][i] = (np.dot(z_i, np.dot(m, z_i)) + np.dot(v, z_i)) - 1
    
    #plot ellipse used for data set creation
    ellipse=plt.contour(Z1,Z2,Z,0, colors=('r'), linestyles='dashed')    
    path=ellipse.collections[0].get_paths()[0] 
    xy = path.vertices
    
    
    
    x_lim=max(max(abs(xy[:,0])), max(abs(xy_num[:,0])))+0.1
    y_lim=max(max(abs(xy[:,1])), max(abs(xy_num[:,1])))+0.1
    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)
    
    #plt.xlim(-maxPlotLimit, maxPlotLimit)
    #plt.ylim(-maxPlotLimit, maxPlotLimit)
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')


if __name__ == "__main__":
    
    '''set boolean True if you want to see the plots. For saving computational
    time, use False.'''
    boolean= True

    global z
    global w
    global l_min
    global l_max

    l_min = 0.1
    l_max = 10

    termination_crit = 10**-3

    my_x = create_rd_x_initial()

    A, b = phi(my_x, 2)

    #A, b = generate_rnd_mx(2, 'own', phi(create_rd_x_initial(),2)[0]), np.random.rand(2)
    #A, b = generate_rnd_mx(2, 'ND'), np.random.rand(2)
    #A, b = generate_rnd_mx(2, 'PD'), np.random.rand(2)

    (z, w) = generate_rnd_points_m2(A, b, 200)

    while abs(sum(w)) > 160:
        my_x = create_rd_x_initial()

        A, b = phi(my_x, 2)
        (z, w) = generate_rnd_points_m2(A, b, 200)


    x_sol, conv, t = barrier_method(.001, create_rd_x_initial(), termination_crit)
    

    '''
    A2, b2 = phi(x_sol,2)
    plot_ellipsoid_m2(A, b, z, w)
    plot_ellipsoid_m2(A2, b2, z, w)
    plt.show()
    '''


    if(boolean==True):
            #first figure
            
            plt.figure(1)
            plot_convergence(conv, 'r','error function','numerical error','loglog')
            #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
            plt.show()
            
            plt.figure(2)
            plot_convergence(conv, 'r','error function','numerical error','plot')
            #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
            plt.show()
            
            plt.figure(3)
            stor_f=np.zeros(len(x_sol))
            for i in range(len(x_sol)):
                stor_f[i]=func_f(x_sol[i])
                
            plot_convergence(stor_f, 'g','blabla', 'objective function','loglog')
            #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
            plt.show()
            
            
            plt.figure(4)
            stor_f=np.zeros(len(x_sol))
            for i in range(len(x_sol)):
                stor_f[i]=func_f(x_sol[i])
                
            plot_convergence(stor_f, 'g','blabla', 'objective function','plot')
            #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
            plt.show()
            
            

            plt.figure(5)
            plot_convergence(np.linalg.norm((np.array(x_sol)-my_x), axis=1), 'b','blabla', 'error','loglog')
            #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
            plt.show()
            
            plt.figure(6)
            plot_convergence(np.linalg.norm((np.array(x_sol)-my_x), axis=1), 'b','blabla', 'error','plot')
            #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
            plt.show()
            
            
            
            plt.figure(7)
            plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro', alpha=0.8, ms=3)
            plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo',alpha=0.5, ms=3)
            plot_ellipses(my_x,np.array(x_sol),z)
            plt.title('exact solution and convergence to it')
            #plt.savefig('Model2_SD_b.png')
            plt.show()
            
           


# Abbruch mit lagrangian
# Q -> Anton (maybe)
# I) reasonable selection of my?
# II) does it depend on tau?
# when we have GN then
# -> new stepsize control? switch to sd
# -> start with sd and switch to gn?
# ->
