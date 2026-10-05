import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import os

def minimizeL2(X, y):  
    ''' 
    This function computes the L2 regression solution using the closed form solution
    '''
    X_transpose = X.T
    w = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    return w

def minimizeL1(X, y):
    ''' 
    This function computes the L1 regression solution using linear programming
    '''
    n, d = X.shape
    c = np.concatenate([np.zeros(d), np.ones(n)])
    G = np.concatenate([np.concatenate([np.zeros_like(X),-np.eye(n)], axis=1),
                        np.concatenate([X,-np.eye(n)], axis=1),
                        np.concatenate([-X,-np.eye(n)], axis=1)
                        ])
    h = np.concatenate([np.zeros_like(y), y,-y])
    solvers.options["show_progress"] = False
    res = solvers.lp(matrix(c), matrix(G), matrix(h))
    w = res['x'][:d]
    return np.array(w)

def minimizeLinf(X, y):
    ''' 
    This function computes the Linf regression solution using linear programming
    '''
    n, d = X.shape
    c = np.concatenate([np.zeros(d), [1.0]])
    G = np.concatenate([np.concatenate([np.zeros((1,d)),-np.ones((1,1))], axis=1),
                        np.concatenate([X,-np.ones((n,1))], axis=1),
                        np.concatenate([-X,-np.ones((n,1))], axis=1)
                        ])
    h = np.concatenate([np.zeros((1,1)), y,-y])
    solvers.options["show_progress"] = False
    res = solvers.lp(matrix(c), matrix(G), matrix(h))
    w = res['x'][:d]
    return np.array(w)

def synRegExperiments():
    ''' 
    This function runs synthetic regression experiments and returns the average train and test loss
    '''

    def genData(n_points, is_training=False):
        '''
        This function generates synthetic data
        '''
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        if is_training:
            y[0] *= -0.1
        return X, y
    
    n_runs = 50
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    np.random.seed(101303431)

    for r in range(n_runs):

        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)

        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # Compute the training and test loss for each model and each metric
        train_loss[r, 0, 0] = (np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=2) ** 2) / (2 * n_train)
        train_loss[r, 0, 1] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=1)/n_train
        train_loss[r, 0, 2] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=np.inf)
        train_loss[r, 1, 0] = (np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=2) ** 2) / (2 * n_train)
        train_loss[r, 1, 1] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=1)/n_train
        train_loss[r, 1, 2] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=np.inf)
        train_loss[r, 2, 0] = (np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=2) ** 2) / (2 * n_train)
        train_loss[r, 2, 1] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=1)/n_train
        train_loss[r, 2, 2] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=np.inf)

        test_loss[r, 0, 0] = (np.linalg.norm(Xtest @ w_L2 - ytest, ord=2) ** 2) / (2 * n_test)
        test_loss[r, 0, 1] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=1)/n_test
        test_loss[r, 0, 2] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=np.inf)
        test_loss[r, 1, 0] = (np.linalg.norm(Xtest @ w_L1 - ytest, ord=2) ** 2) / (2 * n_test)
        test_loss[r, 1, 1] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=1)/n_test
        test_loss[r, 1, 2] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=np.inf)
        test_loss[r, 2, 0] = (np.linalg.norm(Xtest @ w_Linf - ytest, ord=2) ** 2) / (2 * n_test)
        test_loss[r, 2, 1] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=1)/n_test
        test_loss[r, 2, 2] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=np.inf)

    avg_train_loss = np.mean(train_loss, axis=0)  # average over runs
    avg_test_loss = np.mean(test_loss, axis=0)  # average over runs

    return avg_train_loss, avg_test_loss

def preprocessCCS(dataset_folder):
    ''' 
    This function preprocesses the Concrete Compressive Strength dataset
    '''

    X = []
    y = []

    content = pd.read_excel(dataset_folder + '/Concrete_Data.xls', sheet_name='Sheet1')
    for index, row in content.iterrows():
        X.append(row.values[:-1]) # append all columns except the last one as features
        y.append(row.values[-1]) # append the last column as the target variable

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y

def runCCS(dataset_folder):
    ''' 
    This function runs the Concrete Compressive Strength dataset experiments
    and returns the average train and test loss
    '''

    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    n_runs = 50

    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    np.random.seed(101303246)

    for r in range(n_runs):
        indices = np.random.permutation(n)
        split_index  = n//2

        train_index = indices[:split_index]
        test_index = indices[split_index:]

        Xtrain, ytrain = X[train_index], y[train_index]
        Xtest, ytest = X[test_index], y[test_index]

        n_train = len(Xtrain)
        n_test = len(Xtest)

        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # Compute the training and test loss for each model and each metric
        train_loss[r, 0, 0] = (np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=2) ** 2) / (2 * n_train)
        train_loss[r, 0, 1] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=1)/n_train
        train_loss[r, 0, 2] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=np.inf)
        train_loss[r, 1, 0] = (np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=2) ** 2) / (2 * n_train)
        train_loss[r, 1, 1] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=1)/n_train
        train_loss[r, 1, 2] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=np.inf)
        train_loss[r, 2, 0] = (np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=2) ** 2) / (2 * n_train)
        train_loss[r, 2, 1] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=1)/n_train
        train_loss[r, 2, 2] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=np.inf)

        test_loss[r, 0, 0] = (np.linalg.norm(Xtest @ w_L2 - ytest, ord=2) ** 2) / (2 * n_test)
        test_loss[r, 0, 1] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=1)/n_test
        test_loss[r, 0, 2] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=np.inf)
        test_loss[r, 1, 0] = (np.linalg.norm(Xtest @ w_L1 - ytest, ord=2) ** 2) / (2 * n_test)
        test_loss[r, 1, 1] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=1)/n_test
        test_loss[r, 1, 2] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=np.inf)
        test_loss[r, 2, 0] = (np.linalg.norm(Xtest @ w_Linf - ytest, ord=2) ** 2) / (2 * n_test)
        test_loss[r, 2, 1] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=1)/n_test
        test_loss[r, 2, 2] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=np.inf)

    avg_train_loss = np.mean(train_loss, axis=0)  # average over runs
    avg_test_loss = np.mean(test_loss, axis=0)  # average over runs

    return avg_train_loss, avg_test_loss

def linearRegL2Obj(w, X, y):
    ''' 
    This function computes the L2 regression objective value
    '''
    n, d = X.shape
    obj_val = (1 / (2 * n)) *np.linalg.norm(X @ w - y, ord=2) ** 2
    return obj_val

def linearRegL2Grad(w, X, y):
    ''' 
    This function computes the L2 regression gradient
    '''
    n, d = X.shape
    gradient = (1 / n) * X.T @ (X @ w - y)
    return gradient 

def find_opt(obj_func, grad_func, X, y):
    ''' 
    This function finds the optimal solution for a given objective and gradient function
    '''
    d = X.shape[1]
    w_0 = np.random.rand(d)

    def func(w):
        ''' 
        This function computes the objective value for a given w
        '''
        return obj_func(w[:, None], X, y)
    def gd(w):
        ''' 
        This function computes the gradient for a given w
        '''
        return np.asarray(grad_func(w[:, None], X, y)).ravel()

    return minimize(func, w_0, jac=gd)['x'][:, None]

def logisticRegObj(w, X, y):
    ''' 
    This function computes the logistic regression objective value
    '''
    n, d = X.shape
    y = y.ravel()
    z = (X @ w).ravel()
    obj_val = 1/n * np.sum((y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(z, 0)))
    return obj_val

def logisticRegGrad(w, X, y):
    ''' 
    This function computes the logistic regression gradient
    '''
    n, d = X.shape
    y = y.ravel()
    z = (X @ w).ravel()
    gradient = 1/n * X.T @ (expit(z) - y)
    return gradient[:, None]

def synClsExperiments():
    ''' 
    This function runs synthetic classification experiments and returns the average train and test accuracy
    '''

    def genData(n_points, dim1, dim2):
        '''
        This function generates synthetic data
        '''
        c0 = np.ones([1, dim1]) # class 0 center
        c1 = -np.ones([1, dim1]) # class 1 center
        X0 = np.random.randn(n_points, dim1 + dim2) # class 0 input
        X0[:, :dim1] += c0
        X1 = np.random.randn(n_points, dim1 + dim2) # class 1 input
        X1[:, :dim1] += c1
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y
    
    def runClsExp(m=100, dim1=2, dim2=2):
        '''
        Run classification experiment with the specified arguments
        '''

        n_test = 1000
        Xtrain, ytrain = genData(m, dim1, dim2)
        Xtest, ytest = genData(n_test, dim1, dim2)

        w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)
        ytrain_hat = (Xtrain @ w_logit >= 0).astype(int) # Compute predicted labels for training points
        train_acc = np.mean(ytrain_hat == ytrain) # Compute training set accuracy
        
        ytest_hat = (Xtest @ w_logit >= 0).astype(int) # Compute predicted labels for test points
        test_acc = np.mean(ytest_hat == ytest) # Compute test set accuracy
        
        return train_acc, test_acc
    
    n_runs = 50
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])

    np.random.seed(101303431)

    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, dim1 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
        for i, dim2 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)
    
    # The average accuracies over runs
    avg_train_acc = np.mean(train_acc, axis=0)
    avg_test_acc = np.mean(test_acc, axis=0)

    return avg_train_acc, avg_test_acc

def preprocessBCW(dataset_folder):
    '''
    This function preprocesses the Breast Cancer Wisconsin dataset
    '''
    X = []
    y = []

    content = pd.read_csv(dataset_folder + '/wdbc.data', header=None)
    for _, row in content.iterrows():
        X.append(row.values[2:]) # append all columns except the first two as features
        y.append(1 if row.values[1] == 'M' else 0) # append 1 for malignant and 0 for benign as the target variable

    X = np.array(X).astype(float)
    y = np.array(y).reshape(-1, 1)

    return X, y

def runBCW(dataset_folder):
    '''
    This function runs the Breast Cancer Wisconsin dataset experiments
    and returns the average train and test accuracy
    '''
    X, y = preprocessBCW(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    
    n_runs = 50
    train_acc = np.zeros([n_runs])
    test_acc = np.zeros([n_runs])

    np.random.seed(101303431)

    for r in range(n_runs):

        # Random partition of the dataset (50% training, 50% testing)
        indices = np.random.permutation(n)
        split_index  = n//2

        train_index = indices[:split_index]
        test_index = indices[split_index:]

        Xtrain, ytrain = X[train_index], y[train_index]
        Xtest, ytest = X[test_index], y[test_index]

        w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # The model's accuracy on the training data
        train_acc[r] = np.mean((Xtrain @ w >= 0) == ytrain)

        # The model's accuracy on the test data
        test_acc[r] = np.mean((Xtest @ w >= 0) == ytest)

    # The average accuracies over runs
    avg_train_acc = np.mean(train_acc)
    avg_test_acc = np.mean(test_acc)

    return avg_train_acc, avg_test_acc
