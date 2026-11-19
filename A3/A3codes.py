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

    X = np.array(X)
    n, d = X.shape

    K = kernel_func(X, X)

    ones_vector = np.ones((n, n))/n
    K_tilde = K - ones_vector @ K - K @ ones_vector + ones_vector @ K @ ones_vector

    eigenvalues, eigenvectors = eigh(K_tilde, subset_by_index=[n - k, n - 1])
    eigenvalues = eigenvalues[::-1]
    alphas = eigenvectors[:, ::-1]

    A = alphas / np.sqrt(eigenvalues)

    return A 


# Q3: K-means Clustering