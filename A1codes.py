import numpy as np
from cvxopt import matrix, solvers

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

        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the training
        # data). Save them to`train_loss`
        train_loss[r, 0, 0] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=2)
        train_loss[r, 0, 1] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=1)
        train_loss[r, 0, 2] = np.linalg.norm(Xtrain @ w_L2 - ytrain, ord=np.inf)
        train_loss[r, 1, 0] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=2)
        train_loss[r, 1, 1] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=1)
        train_loss[r, 1, 2] = np.linalg.norm(Xtrain @ w_L1 - ytrain, ord=np.inf)
        train_loss[r, 2, 0] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=2)
        train_loss[r, 2, 1] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=1)
        train_loss[r, 2, 2] = np.linalg.norm(Xtrain @ w_Linf - ytrain, ord=np.inf)
        
        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the test
        # data). Save them to`test_loss`
        test_loss[r, 0, 0] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=2)
        test_loss[r, 0, 1] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=1)
        test_loss[r, 0, 2] = np.linalg.norm(Xtest @ w_L2 - ytest, ord=np.inf)
        test_loss[r, 1, 0] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=2)
        test_loss[r, 1, 1] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=1)
        test_loss[r, 1, 2] = np.linalg.norm(Xtest @ w_L1 - ytest, ord=np.inf)
        test_loss[r, 2, 0] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=2)
        test_loss[r, 2, 1] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=1)
        test_loss[r, 2, 2] = np.linalg.norm(Xtest @ w_Linf - ytest, ord=np.inf)

    # TODO: compute the average losses over runs
    avg_train_loss = np.mean(train_loss, axis=0)  # average over runs
    avg_test_loss = np.mean(test_loss, axis=0)  # average over runs

    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return avg_train_loss, avg_test_loss

if __name__ == "__main__":
	train_loss, test_loss = synRegExperiments()
	print("Training loss: ", train_loss)
	print("Testing loss: ", test_loss)