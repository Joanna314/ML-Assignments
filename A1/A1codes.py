import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import os

def minimizeL2(X, y):  
    X_transpose = X.T
    w = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    return w

def minimizeL1(X, y):
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

        train_loss[r, 0, 0] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=2)
        train_loss[r, 0, 1] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=1)
        train_loss[r, 0, 2] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=np.inf)
        train_loss[r, 1, 0] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=2)
        train_loss[r, 1, 1] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=1)
        train_loss[r, 1, 2] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=np.inf)
        train_loss[r, 2, 0] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=2)
        train_loss[r, 2, 1] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=1)
        train_loss[r, 2, 2] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=np.inf)
        
        test_loss[r, 0, 0] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=2)
        test_loss[r, 0, 1] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=1)
        test_loss[r, 0, 2] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=np.inf)
        test_loss[r, 1, 0] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=2)
        test_loss[r, 1, 1] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=1)
        test_loss[r, 1, 2] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=np.inf)
        test_loss[r, 2, 0] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=2)
        test_loss[r, 2, 1] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=1)
        test_loss[r, 2, 2] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=np.inf)

    avg_train_loss = np.mean(train_loss, axis=0)  # average over runs
    avg_test_loss = np.mean(test_loss, axis=0)  # average over runs

    return avg_train_loss, avg_test_loss

def preprocessCCS(dataset_folder):

    X = []
    y = []

    content = pd.read_excel(dataset_folder + '/Concrete_Data.xls', sheet_name='Sheet1')
    for index, row in content.iterrows():
        X.append(row.values[:-1])
        y.append(row.values[-1])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y

def runCCS(dataset_folder):

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

        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        train_loss[r, 0, 0] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=2)
        train_loss[r, 0, 1] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=1)
        train_loss[r, 0, 2] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=np.inf)
        train_loss[r, 1, 0] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=2)
        train_loss[r, 1, 1] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=1)
        train_loss[r, 1, 2] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=np.inf)
        train_loss[r, 2, 0] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=2)
        train_loss[r, 2, 1] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=1)
        train_loss[r, 2, 2] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=np.inf)
        
        test_loss[r, 0, 0] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=2)
        test_loss[r, 0, 1] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=1)
        test_loss[r, 0, 2] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=np.inf)
        test_loss[r, 1, 0] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=2)
        test_loss[r, 1, 1] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=1)
        test_loss[r, 1, 2] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=np.inf)
        test_loss[r, 2, 0] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=2)
        test_loss[r, 2, 1] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=1)
        test_loss[r, 2, 2] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=np.inf)

    avg_train_loss = np.mean(train_loss, axis=0)  # average over runs
    avg_test_loss = np.mean(test_loss, axis=0)  # average over runs

    return avg_train_loss, avg_test_loss

def preprocessBCW(dataset_folder):
    X = []
    y = []

    content = pd.read_csv(dataset_folder + '/wdbc.data', header=None)
    for _, row in content.iterrows():
        X.append(row.values[2:])
        y.append(1 if row.values[1] == 'M' else 0)

    X = np.array(X).astype(float)
    y = np.array(y).reshape(-1, 1)

    return X, y

def runBCW(dataset_folder):

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
        train_acc[r] = np.mean((Xtrain @ w >= 0.5) == ytrain)

        # The model's accuracy on the test data
        test_acc[r] = np.mean((Xtest @ w >= 0.5) == ytest)

    # The average accuracies over runs
    avg_train_acc = np.mean(train_acc)
    avg_test_acc = np.mean(test_acc)

    return avg_train_acc, avg_test_acc

if __name__ == "__main__":
    # avg_train_loss, avg_test_loss = runCCS(os.path.join(os.path.abspath("A1/data_folder")))
    # print("Average Train Loss:\n", avg_train_loss)
    # print("Average Test Loss:\n", avg_test_loss)

    avg_train_acc, avg_test_acc = runBCW(os.path.join(os.path.abspath("A1/data_folder")))
    print("BCW Dataset")
    print("Average Training Accuracy:", avg_train_acc)
    print("Average Test Accuracy:", avg_test_acc)