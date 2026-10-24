from A2helpers import *
import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import minimize

solvers.options['show_progress'] = False

# Question 1 Binary Classification (Primal Form)


# Question 2 Binary Classification (Adjoint Form)
def adjSqHinge(X, y, lamb, kernel_func):
    '''
    This function computes the best model for the adjoint squared hinge loss optimization problem

    Args:
        X: input data (n x n)
        y: labels (n x 1)
        lamb: regularization parameter
        kernel_func: kernel function to compute the kernel matrix
    
    Returns:
        (a, a0): tuple of optimal model parameters
            a: dual coefficients, shape (n x 1)
            a0: scalar scalar
    '''
    K = kernel_func(X, X)   # compute the kernel matrix
    y = y.reshape(-1)    # reshape y to be a 1D array
    n = len(y)

    def obj_func(params):
        '''
        This function computes the objective function value for the adjoint squared hinge loss optimization problem
        '''
        a = params[:n]  # parameters
        a0 = params[n]  # intercept

        m = y * (K @ a + a0)
        loss = np.sum(np.maximum(0, 1 - m) ** 2)
        reg = 0.5 * lamb * float(a.T @ K @ a)

        return loss + reg

    def obj_gradient(params):
        a = params[:n]
        a0 = params[n]
        margins = y * (K @ a + a0)
        residual = np.maximum(0, 1 - margins)
        grad_a = -2 * K.T @ (y * residual) + lamb * (K @ a)
        grad_a0 = -2 * np.sum(y * residual)
        return np.concatenate([grad_a, [grad_a0]])

    # minimize the total loss and
    res = minimize(obj_func, np.zeros(n + 1), jac=obj_gradient)
    a, a0 = res.x[:n].reshape(-1, 1), res.x[n]
    return a, a0

def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    ''' 
    This function computes the adjoint hinge loss optimization problem
    '''
    K = kernel_func(X, X)
    y = y.reshape(-1)
    n = len(y)

    # objective quadratic term
    P = np.block([
        [lamb * K, np.zeros((n, n+1))],
        [np.zeros((1, 2*n+1))],
        [np.zeros((n, 2*n+1))]
    ])
    P = P + stabilizer * np.eye(2*n + 1)  # add stabilizer to the diagonal

    # objective linear term
    q = np.concatenate([np.zeros(n + 1), np.ones(n)])

    # constraints
    G = np.block([
        [np.zeros((n, n)), np.zeros((n, 1)), -np.eye(n)],
        [-np.diag(y) @ K, -y.reshape(-1, 1), -np.eye(n)]
    ])
    h = np.concatenate([np.zeros(n), -np.ones(n)])

    # minimize the total loss with constraints using cvxopt
    res = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    solution = np.asarray(res['x']).reshape(-1)
    a, a0 = solution[:n].reshape(-1, 1), solution[n]
    return a, a0

def adjClassify(Xtest, a, a0, X, kernel_func):
    '''
    This function classifies the test data using the adjoint model parameters

    Args:
        Xtest: test data (m x n)
        a: parameters (n x 1)
        a0: scalar intercept
        X: training data (n x n)
        kernel_func: kernel function to compute the kernel matrix
    
    Returns:
        yhat: predicted labels for the test data (m x 1)
    '''
    K_test = kernel_func(Xtest, X)  # compute the kernel matrix between test and training data
    yhat = np.sign(K_test @ a + a0)  # compute the predicted labels
    return yhat

def synExperimentsKernel():
    
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
                    lambda X1, X2: polyKernel(X1, X2, 2),
                    lambda X1, X2: polyKernel(X1, X2, 3),
                    lambda X1, X2: gaussKernel(X1, X2, 1.0),
                    lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    
    np.random.seed(176)

    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                
                a, a0 = adjSqHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_sq_hinge[i, j, r] = np.mean(
                    adjClassify(Xtrain, a, a0, Xtrain, kernel).reshape(-1) == ytrain.reshape(-1)
                )
                #plotAdjModel(Xtrain, ytrain, a, a0, kernel, adjClassify)
                test_acc_sq_hinge[i, j, r] = np.mean(
                    adjClassify(Xtest, a, a0, Xtrain, kernel).reshape(-1) == ytest.reshape(-1)
                )
                #plotAdjModel(Xtest, ytest, a, a0, kernel, adjClassify)

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(
                    adjClassify(Xtrain, a, a0, Xtrain, kernel).reshape(-1) == ytrain.reshape(-1)
                )
                test_acc_hinge[i, j, r] = np.mean(
                    adjClassify(Xtest, a, a0, Xtrain, kernel).reshape(-1) == ytest.reshape(-1)
                )

    # compute the average accuracies over runs
    avg_train_acc_sq_hinge = np.mean(train_acc_sq_hinge, axis=2)
    avg_test_acc_sq_hinge = np.mean(test_acc_sq_hinge, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    # combine accuracies (sq_hinge and hinge)
    combined_train_acc = np.concatenate(
        [avg_train_acc_sq_hinge, avg_train_acc_hinge], axis=1
    )
    combined_test_acc = np.concatenate(
        [avg_test_acc_sq_hinge, avg_test_acc_hinge], axis=1
    )

    # return 5-by-6 train accuracy and 5-by-6 test accuracy
    return combined_train_acc, combined_test_acc

# Question 3 Binary Classification (SVM Dual Form)
