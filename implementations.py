import numpy as np

### HELPERS ###
def compute_loss(y, tx, w):
    """
    Compute the MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The weight vector.

    Returns:
        loss: float. The value of the loss corresponding to the input weight vector w.
    """
    N = len(y)
    e = np.array(y) - tx @ w
    loss = 1/(2*N) * np.sum(np.pow(e, 2))

    return loss

def compute_gradient(y, tx, w):
    """
    Compute the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The weight vector.

    Returns:
        gradient: numpy array of shape (2, ).
    """
    N = len(y)
    e = np.array(y) - tx @ w
    gradient = -(1/N) * tx.transpose() @ e

    return gradient

def sigmoid(t):
    """
    Apply sigmoid function on t.
    """
    return 1 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """
    Compute the negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: float. The value of the loss corresponding to the input weight vector w.
    """
    N = tx.shape[0]
    s = sigmoid(tx @ w)
    loss = -(1/N) *  np.squeeze(np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))).item()

    return loss

def compute_logistic_gradient(y, tx, w):
    """
    Compute the gradient at w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        gradient: numpy array of shape (2, ).
    """
    N = tx.shape[0]
    s = np.array(y) - sigmoid(tx @ w)
    gradient = -(1/N) * tx.transpose() @ s

    return gradient


### FUNCTIONS ###
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial weight vector.
        max_iters: int. A scalar denoting the total number of iterations of GD.
        gamma: int. A scalar denoting the step size.

    Returns:
        w: numpy array of shape=(2, ). The last weight vector.
        loss: float. The corresponding loss value.
    """
    # Define a parameter to store w
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient
        gradient = compute_gradient(y, tx, w)
        # Update the weights
        w = w - gamma * gradient

    # Compute the loss
    loss = compute_loss(y, tx, w)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial weight vector.
        max_iters: int. A scalar denoting the total number of iterations of GD.
        gamma: int. A scalar denoting the step size.

    Returns:
        w: numpy array of shape=(2, ). The last weight vector.
        loss: float. The corresponding loss value.
    """

    # Define a parameter to store w
    w = initial_w

    for n_iter in range(max_iters):
        # Standard mini-batch size is one.
        idx = np.random.randint(len(y))
        minibatch_y, minibatch_tx = y[idx:idx+1], tx[idx:idx+1]
        # Compute the gradient
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        # Update the weights
        w = w - gamma * gradient

    # Compute the loss
    loss = compute_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    Args:
        y: numpy array of shape (N,). N is the number of samples.
        tx: numpy array of shape (N,D). D is the number of features.

    Returns:
        w: numpy array of shape=(2, ). The last weight vector.
        loss: float. The corresponding loss value.
    """
    # Solve the linear system
    A = tx.transpose() @ tx
    b = tx.transpose() @ y
    w = np.linalg.solve(A, b)

    # Compute the loss
    loss = compute_loss(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,). N is the number of samples.
        tx: numpy array of shape (N,D). D is the number of features.
        lambda_: float. The regularization parameter.

    Returns:
        w: numpy array of shape=(2, ). The last weight vector.
        loss: float. The corresponding loss value.
    """
    N = tx.shape[0]
    D = tx.shape[1]

    # Solve the linear system
    A = (tx.transpose() @ tx) + (lambda_*2*N) * np.identity(D)
    b = tx.transpose() @ y
    w = np.linalg.solve(A, b)

    # Compute the loss
    loss = compute_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y in {0,1}).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial weight vector.
        max_iters: int. A scalar denoting the total number of iterations of GD.
        gamma: int. A scalar denoting the step size.

    Returns:
        w: numpy array of shape=(2, ). The last weight vector.
        loss: float. The corresponding loss value.
    """
    # Define a parameter to store w
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient
        gradient = compute_logistic_gradient(y, tx, w)
        # Update the weights
        w = w - gamma * gradient


    # Compute the loss
    loss = compute_logistic_loss(y, tx, w)

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent (y in {0,1}).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: float. The regularization parameter.
        initial_w: numpy array of shape=(2, ). The initial weight vector.
        max_iters: int. A scalar denoting the total number of iterations of GD.
        gamma: int. A scalar denoting the step size.

    Returns:
        w: numpy array of shape=(2, ). The last weight vector.
        loss: float. The corresponding loss value.
    """
    # Define a parameter to store w
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient
        gradient = compute_logistic_gradient(y, tx, w) + (2 * lambda_ * w)
        # Update the weights
        w = w - gamma * gradient

    # Compute the loss (without the penalty term)
    loss = compute_logistic_loss(y, tx, w)

    return w, loss
