import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers

def minSqHinge(X, y, lamb):
    """
        This function computes the best model with squared hinge loss and L2 regularization.
    
        Args:
            X: An n-by-d matrix of input vectors
            y: An n-by-1 vector of output labels
    
        Returns:
            A tuple of optimal model parameters (w, w0) where w is d-by-1 and
            w0 is a scalar
    """

    d = X.shape[1]

    def obj_func(u): 
        w0 = u[-1]  
        w = u[:-1] 
        w = w[:, None]
    
        loss = np.sum(np.maximum(0, 1 - y * (X @ w + w0)) ** 2)
        reg = 0.5 * lamb * (w.T @ w).item()
    
        return loss + reg
    
    u0 = np.ones(d + 1) 

    sol = minimize(obj_func, u0)

    w = sol['x'][:-1][:, None]
    w0 = sol['x'][-1]

    return w, w0

    
def minHinge(X, y, lamb, stabilizer=1e-5):
    """
        This function computes the best model with hinge loss and L2 regularization.
    
        Args:
            X: An n-by-d matrix of input vectors
            y: An n-by-1 vector of output labels
    
        Returns:
            A tuple of optimal model parameters (w, w0) where w is d-by-1 and
            w0 is a scalar
    """
    n, d = X.shape

    P = np.zeros((n + d + 1, n + d + 1))
    P[:d, :d] = lamb * np.eye(d)
    P = P + stabilizer * np.eye(n + d + 1)

    q = np.zeros(n + d + 1)
    q[d+1:] = 1.0

    y = y.reshape(-1, 1)

    G = np.concatenate([np.concatenate([np.zeros((n, d)), np.zeros((n, 1)), -np.eye(n)], axis=1),
                        np.concatenate([np.diag(y.flatten()) @ X, -y, -np.eye(n)], axis=1)
                        ])
    h = np.concatenate([np.zeros((n, 1)),-np.ones((n,1))], axis=0)

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    sol = solvers.qp(P, q, G, h)
    
    u = np.array(sol['x'])
    
    w = u[:d].reshape(-1, 1)
    w0 = u[d]
    
    return w, w0

def  classify(Xtest, w, w0):
    """
        This function classifies the test data using the learned model parameters.
    
        Args:
            Xtest: An n-by-d matrix of input vectors
            w: A d-by-1 vector of model weights
            w0: A scalar bias term
    
        Returns:
            An n-by-1 vector of predicted labels
    """
    scores = Xtest @ w + w0
    yhat = np.where(scores >= 0, 1.0, -1.0)
    return yhat

def synExperimentsRegularize():

    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    np.random.seed(176)

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                w, w0 = minSqHinge(Xtrain, ytrain, lamb)
                train_acc_sq_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)
                test_acc_sq_hinge[i, j, r] = np.mean(classify(Xtest, w, w0) == ytest)
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)
                test_acc_hinge[i, j, r] = np.mean(classify(Xtest, w, w0) == ytest)

    avg_train_acc_sq_hinge = np.mean(train_acc_sq_hinge, axis=2)
    avg_test_acc_sq_hinge = np.mean(test_acc_sq_hinge, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    avg_train_acc = np.concatenate([avg_train_acc_sq_hinge, avg_train_acc_hinge], axis=1)
    avg_test_acc = np.concatenate([avg_test_acc_sq_hinge, avg_test_acc_hinge], axis=1)

    return avg_train_acc, avg_test_acc
