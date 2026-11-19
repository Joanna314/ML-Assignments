# COMP 3105 Assignment 3
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from scipy.spatial.distance import cdist
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
    Classifies the test data using the learned weights.
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
    Computes the accuracy of the predictions.
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
def kmeans(X, k, max_iter=1000):
    """
    Computes the k-means clustering of the data points in X.
    Inputs:
        X: n x d matrix of data points
        k: number of clusters
        max_iter: maximum number of iterations
    Outputs:
        Y: n x k matrix of one-hot encoded cluster assignments
        U: k x d matrix of cluster centers
        obj_val: final objective value
    """
    n, d = X.shape
    assert max_iter > 0 and 1 < k < n
    U = X[np.random.choice(n, k, replace=False)] # Select k random points as initial cluster centers
    for i in range(max_iter):
        D = cdist(X, U, metric='sqeuclidean') # Compute pairwise distance between X and U
        Y = np.eye(k)[np.argmin(D, axis=1)] # Find the new cluster assignments
        old_U = U
        U = np.linalg.pinv(Y.T @ Y) @ Y.T @ X # Update cluster centers
        if np.allclose(old_U, U):
            break
    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, U, obj_val

def repeatKmeans(X, k, n_runs=100):
    """
    Repeats the k-means clustering n_runs times and returns the best result.
    """
    best_obj_val = float('inf')
    for r in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        if obj_val < best_obj_val:
            best_Y, best_U, best_obj_val = Y, U, obj_val
    return best_Y, best_U, best_obj_val

def chooseK(X, k_candidates=[2,3,4,5,6,7,8,9]):
    """
    Returns the list of objective values for each candidate k.
    """
    obj_val_list = []
    for k in k_candidates:
        _, _, obj_val = repeatKmeans(X, k)
        obj_val_list.append(obj_val)
    return obj_val_list

def kernelKmeans(X, kernel_func, k, init_Y, max_iter=1000):
    """
    Computes the kernel k-means clustering of the data points in X.
    Inputs:
        X: n x d matrix of data points
        kernel_func: function to compute the kernel matrix
        k: number of clusters
        init_Y: n x k matrix of initial one-hot encoded cluster assignments
        max_iter: maximum number of iterations
    Outputs:
        Y: n x k matrix of one-hot encoded cluster assignments
        obj_val: final objective value
    """
    
    n, d = X.shape
    assert max_iter > 0 and 1 < k < n
    K = kernel_func(X, X)
    Y = init_Y
    for i in range(max_iter):
        Y_plus = np.linalg.pinv(Y)
        D = np.diag(K)[:, None] @ np.ones((1, k)) + np.ones((n, 1)) @ np.diag(Y_plus @ K @ Y_plus.T)[None, :] - 2 * (K @ Y_plus.T) 
        old_Y = Y
        Y = np.eye(k)[np.argmin(D, axis=1)]
        if np.allclose(old_Y, Y):
            break
    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, obj_val