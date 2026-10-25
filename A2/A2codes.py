from A2helpers import *
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from scipy.optimize import minimize

solvers.options['show_progress'] = False

# Question 1 Binary Classification (Primal Form)
def minSqHinge(X, y, lamb):
    '''
    Compute the best model with squared hinge loss and L2 regularization.
    '''
    d = X.shape[1]

    def obj_func(params):
        w = params[:d].reshape(-1, 1)
        w0 = params[d]
        loss = np.sum(np.maximum(0, 1 - y * (X @ w + w0)) ** 2)
        reg = 0.5 * lamb * float(w.T @ w)
        return loss + reg

    res = minimize(obj_func, np.ones(d + 1))
    w = res.x[:d].reshape(-1, 1)
    w0 = res.x[d]
    return w, w0


def minHinge(X, y, lamb, stabilizer=1e-5):
    '''
    Compute the best model with hinge loss and L2 regularization.
    '''
    n, d = X.shape
    y = y.reshape(-1)

    P = np.zeros((n + d + 1, n + d + 1))
    P[:d, :d] = lamb * np.eye(d)
    P += stabilizer * np.eye(n + d + 1)

    q = np.zeros(n + d + 1)
    q[d + 1:] = 1.0

    G = np.vstack([
        np.hstack([np.zeros((n, d)), np.zeros((n, 1)), -np.eye(n)]),
        np.hstack([-(y[:, None] * X), -y[:, None], -np.eye(n)])
    ])
    h = np.concatenate([np.zeros(n), -np.ones(n)])

    res = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    solution = np.asarray(res['x']).reshape(-1)
    w = solution[:d].reshape(-1, 1)
    w0 = solution[d]
    return w, w0


def classify(Xtest, w, w0):
    '''Classify test inputs, predicting +1 for scores exactly equal to zero.'''
    scores = Xtest @ w + w0
    return np.where(scores >= 0, 1.0, -1.0)


def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.0]
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
                train_acc_sq_hinge[i, j, r] = np.mean(
                    classify(Xtrain, w, w0) == ytrain
                )
                test_acc_sq_hinge[i, j, r] = np.mean(
                    classify(Xtest, w, w0) == ytest
                )
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(
                    classify(Xtrain, w, w0) == ytrain
                )
                test_acc_hinge[i, j, r] = np.mean(
                    classify(Xtest, w, w0) == ytest
                )

    avg_train_acc = np.concatenate([
        np.mean(train_acc_sq_hinge, axis=2),
        np.mean(train_acc_hinge, axis=2)
    ], axis=1)
    avg_test_acc = np.concatenate([
        np.mean(test_acc_sq_hinge, axis=2),
        np.mean(test_acc_hinge, axis=2)
    ], axis=1)
    return avg_train_acc, avg_test_acc


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
    kernel_list = [
        linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5)
    ]
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
                test_acc_sq_hinge[i, j, r] = np.mean(
                    adjClassify(Xtest, a, a0, Xtrain, kernel).reshape(-1) == ytest.reshape(-1)
                )

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(
                    adjClassify(Xtrain, a, a0, Xtrain, kernel).reshape(-1) == ytrain.reshape(-1)
                )
                test_acc_hinge[i, j, r] = np.mean(
                    adjClassify(Xtest, a, a0, Xtrain, kernel).reshape(-1) == ytest.reshape(-1)
                )

    avg_train_acc_sq_hinge = np.mean(train_acc_sq_hinge, axis=2)
    avg_test_acc_sq_hinge = np.mean(test_acc_sq_hinge, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    combined_train_acc = np.concatenate(
        [avg_train_acc_sq_hinge, avg_train_acc_hinge], axis=1
    )
    combined_test_acc = np.concatenate(
        [avg_test_acc_sq_hinge, avg_test_acc_hinge], axis=1
    )
    return combined_train_acc, combined_test_acc


# Question 3 Binary Classification (SVM Dual Form)
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    '''
    Solve the dual SVM hinge-loss problem and recover its intercept.
    '''
    K = kernel_func(X, X)
    y = y.reshape(-1)
    n = len(y)

    # cvxopt minimizes 1/2 * alpha.T @ P @ alpha + q.T @ alpha.
    P = np.outer(y, y) * K / lamb
    P = P + stabilizer * np.eye(n)
    q = -np.ones(n)

    # 0 <= alpha <= 1, with y.T @ alpha = 0.
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.concatenate([np.zeros(n), np.ones(n)])
    A = y.reshape(1, -1)
    b_eq = np.zeros(1)

    res = solvers.qp(
        matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b_eq)
    )
    alpha = np.asarray(res['x']).reshape(-1)

    # The coefficient closest to 0.5 is the most reliable for recovering b.
    support_idx = np.argmin(np.abs(alpha - 0.5))
    b = y[support_idx] - K[support_idx, :] @ (y * alpha) / lamb
    return alpha.reshape(-1, 1), float(b)


def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    '''Classify inputs using the dual SVM model, predicting +1 at score zero.'''
    alpha = np.asarray(a).reshape(-1)
    labels = np.asarray(y).reshape(-1)
    scores = (
        kernel_func(Xtest, X) @ (labels * alpha) / lamb
        + np.asarray(b).reshape(-1)[0]
    )
    return np.where(scores >= 0, 1.0, -1.0).reshape(-1, 1)


def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(
        f"{dataset_folder}/A2train.csv", header=None
    ).to_numpy()

    X = train_data[:, 1:] / 255.0
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1

    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])

    np.random.seed(176)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for fold_idx, val_idx in enumerate(folds):
                train_idx = np.concatenate(
                    [fold for idx, fold in enumerate(folds) if idx != fold_idx]
                )
                Xtrain, ytrain = X[train_idx], y[train_idx]
                Xval, yval = X[val_idx], y[val_idx]
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(
                    Xval, a, b, Xtrain, ytrain, lamb, kernel_func
                )
                cv_acc[fold_idx, i, j] = np.mean(yhat == yval)

    cv_acc = np.mean(cv_acc, axis=0)
    best_lamb_idx, best_kernel_idx = np.unravel_index(
        np.argmax(cv_acc), cv_acc.shape
    )
    return cv_acc, lamb_list[best_lamb_idx], kernel_list[best_kernel_idx]
