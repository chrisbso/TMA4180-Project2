import numpy as np
##MODEL 1
#################################################################################################
# z is n x m matrix
# w is 1 x m matrix (labels of z_i-s)
# A is n x n matrix
# c is 1 x n matrix

#Calculate r_i according to model 1
def evaluate_r_i_m1(z_i,w_i,A,c):
    if w_i == 1:
        r_i = max(np.dot((z_i - c), np.dot(A, (z_i - c)))-1, 0)
    elif w_i == -1:
        r_i = max(1 - np.dot((z_i - c), np.dot(A, (z_i - c))), 0)
    return r_i

#Calculate f according to model 1
def evaluate_f_m1(z, w, A, c):
    m = w.shape[0]
    f = 0
    for i in range(m):
        z_i = z[:, i]  # extract column i of matrix z, i.e. element z_i, i=0,...,m-1
        r_i = evaluate_r_i_m1(z_i,w[i],A,c)
        f += r_i ** 2
    return f

#Calculate grad{r_i} according to model 1
def evaluate_grad_r_i_m1(z_i,A,c):
    n = A.shape[0]
    l = 0;
    grad_r = np.zeros(int(n * (n + 1) / 2 + n))
    for j in range(n):
        for k in range(j, n):
            if j == k:
                grad_r[l] = (z_i[j] - c[j]) ** 2
            else:
                grad_r[l] = 2 * (z_i[j] - c[j]) * (z_i[k] - c[k])
            l += 1
    tail = 2 * np.matmul(c - z_i, A)
    grad_r[int(n * (n + 1) / 2):] = tail
    return grad_r

#Calculate grad{f} according to model 1
def evaluate_grad_f_m1(z,w,A,c):
    n = A.shape[0]
    m = z.shape[1]
    grad_f = np.zeros(int(n*(n+1)/2+n))
    for i in range(m):
        z_i = z[:,i]
        w_i = w[i]
        if (isInRightSet_m1(z_i,w_i,A,c)):
            continue
        else:
            grad_r = w_i*evaluate_grad_r_i_m1(z_i,A,c)
            r = evaluate_r_i_m1(z_i, w_i, A, c)
            grad_f += r * grad_r
    grad_f *= 2
    return grad_f

#Check if the points are in S_{A,c} or not, and give indication whether or not they should
def isInRightSet_m1(z_i,w_i,A,c):
    isIn = False
    if np.dot((z_i - c), np.dot(A, (z_i - c)))-1 < 0 and w_i == 1:
        isIn = True
    elif np.dot((z_i - c), np.dot(A, (z_i - c)))-1 > 0 and w_i == -1:
        isIn = True
    return isIn

##MODEL 2
#############################################################################################
    # z is n x m matrix
    # w is 1 x m matrix (labels of z_i-s)
    # A is n x n matrix
    # b is 1 x n matrix

#Calculate r_i according to model 2
def evaluate_r_i_m2(z_i,w_i,A,b):
    if w_i == 1:
        r_i = max(np.dot(z_i, np.dot(A, z_i))+ np.dot(b,z_i)-1, 0)
    elif w_i == -1:
        r_i = max(1 -(np.dot(z_i, np.dot(A,z_i))+ np.dot(b,z_i)), 0)
    return r_i


#Calculate f according to model 2
def evaluate_f_m2(z, w, A, b):
    m = w.shape[0]
    f = 0
    for i in range(m):
        z_i = z[:, i]  # extract column i of matrix z, i.e. element z_i, i=0,...,m-1
        r_i = evaluate_r_i_m2(z_i,w[i],A,b)
        f += r_i ** 2
    return f


#Calculate grad{r_i} according to model 2
def evaluate_grad_r_i_m2(z_i,A):
    n = A.shape[0]
    l = 0;
    grad_r = np.zeros(int(n * (n + 1) / 2 + n))
    for j in range(n):
        for k in range(j, n):
            if j == k:
                grad_r[l] = z_i[j] ** 2
            else:
                grad_r[l] = 2 * z_i[j] * z_i[k]
            l += 1
    grad_r[int(n * (n + 1) / 2):] = z_i
    return grad_r

#Calculate grad{f} according to model 2
def evaluate_grad_f_m2(z,w,A,b):
    n = A.shape[0]
    m = z.shape[1]
    grad_f = np.zeros(int(n*(n+1)/2+n))
    for i in range(m):
        z_i = z[:,i]
        w_i = w[i]
        if (isInRightSet_m2(z_i,w_i,A,b)):
            continue
        else:
            grad_r = w_i*evaluate_grad_r_i_m2(z_i,A)
            r = evaluate_r_i_m2(z_i, w_i, A, b)
            grad_f += r * grad_r
    grad_f *= 2
    return grad_f

#Check if the points are in S_{A,b} or not, and give indication whether or not they should
def isInRightSet_m2(z_i,w_i,A,b):
    isIn = False
    if (np.dot(z_i, np.dot(A, z_i))+ np.dot(z_i,b))-1 < 0 and w_i == 1:
        isIn = True
    elif (np.dot(z_i, np.dot(A, z_i))+ np.dot(z_i,b))-1 > 0 and w_i == -1:
        isIn = True
    return isIn
