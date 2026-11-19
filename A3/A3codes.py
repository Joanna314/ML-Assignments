# COMP 3105 Assignment 3
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from scipy.linalg import eigh
import A3helpers as A3helpers

# Q1: Linear Multi-Class Classifier

def minMulDev(X, Y):
    """
    Solves the multinomial deviance minimization loss.
    Inputs:
        X: n x d matrix of training data (augmented)
        Y: n x k matrix of one-hot encoded labels
    Outputs:
        W: d x k matrix of weights
    """
    d = X.shape[1]
    k = Y.shape[1]

    def obj_func(W):
        W = W.reshape(d, k)
        z_hat = X @ W
        loss = np.mean(logsumexp(z_hat, axis=1) - np.sum(Y * z_hat, axis=1))
        return loss

    res = minimize(obj_func, np.zeros(d * k), method='BFGS')
    return res.x.reshape(d, k)

def classify(Xtest, W):
    """
    Inputs:
        Xtest: m x d matrix of test data (augmented)
        W: d x k matrix of weights
    Outputs:
        Yhat: m x k matrix of one-hot encoded predicted labels
    """
    k = W.shape[1]

    z_hat = Xtest @ W
    Yhat = np.eye(k)[np.argmax(z_hat, axis=1)]
    return Yhat

def calculateAcc(Yhat, Y):
    """
    Inputs:
        Yhat: m x k matrix of one-hot encoded predicted labels
        Y: m x k matrix of one-hot encoded true labels
    Outputs:
        acc: accuracy (between 0 and 1)
    """
    acc = np.mean(np.argmax(Yhat, axis=1) == np.argmax(Y, axis=1))
    return acc


# Q2: Principal Component Analysis

def PCA(X, k):

    X = np.array(X)
    n, d = X.shape

    mu = np.mean(X, axis=0, keepdims=True).T
    centered_data = X - mu.T

    covariance_matrix = centered_data.T @ centered_data

    eigenvalues, eigenvectors = eigh(covariance_matrix, subset_by_index=[d - k, d - 1])

    U = eigenvectors[:, ::-1].T

    return U, mu

def projPCA(Xtest, mu, U):

    Xproj = (Xtest - mu.T) @ U.T

    return Xproj

def kernelPCA(X, k, kernel_func):

    X = np.array(X, dtype=float)
    n, d = X.shape

    K = kernel_func(X, X)

    ones_vector = np.ones((n, n))/n
    K_tilde = K - ones_vector @ K - K @ ones_vector + ones_vector @ K @ ones_vector

    eigenvalues, eigenvectors = eigh(K_tilde, subset_by_index=[n - k, n - 1])
    eigenvalues = eigenvalues[::-1]
    alphas = eigenvectors[:, ::-1]

    A = alphas / np.sqrt(eigenvalues)

    return A 

def projKernelPCA(Xtest, Xtrain, kernel_func, A):

    Xtest = np.array(Xtest, dtype=float)
    Xtrain = np.array(Xtrain, dtype=float)
    A = np.array(A, dtype=float)

    m, dtest = Xtest.shape
    n, dtrain = Xtrain.shape

    K_te_tr = kernel_func(Xtest, Xtrain)
    K_tr_tr = kernel_func(Xtrain, Xtrain)

    ones_vector_dim_mn = np.ones((m, n))/n
    ones_vector_dim_nn = np.ones((n, n))/n

    K_tilde = K_te_tr - ones_vector_dim_mn @ K_tr_tr - K_te_tr @ ones_vector_dim_nn + ones_vector_dim_mn @ K_tr_tr @ ones_vector_dim_nn

    Xproj = K_tilde @ A.T

    return Xproj


def synClsExperimentsPCA():
    print("Running synthetic classification experiments with PCA...")
    n_runs = 100
    n_train = 128
    n_test = 1000

    dim_list = [1, 2]
    gen_model_list = [1, 2]

    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])

    np.random.seed(75)

    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):

                Xtrain, Ytrain = A3helpers.generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = A3helpers.generateData(n=n_test, gen_model=gen_model)

                U, mu = PCA(Xtrain, k)
                
                Xtrain_proj = projPCA(Xtrain, mu, U)
                Xtest_proj = projPCA(Xtest, mu, U)

                Xtrain_proj = A3helpers.augmentX(Xtrain_proj) # add augmentation
                Xtest_proj = A3helpers.augmentX(Xtest_proj)

                W = minMulDev(Xtrain_proj, Ytrain) # from Q1
                Yhat = classify(Xtrain_proj, W) # from Q1

                train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1
                Yhat = classify(Xtest_proj, W)
                test_acc[i, j, r] = calculateAcc(Yhat, Ytest)
                
    avg_train_acc = np.mean(train_acc, axis=2)
    avg_test_acc = np.mean(test_acc, axis=2)

    return avg_train_acc, avg_test_acc

# Q3: K-means Clustering
