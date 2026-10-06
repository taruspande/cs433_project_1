### All useful functions that are called in the implementations.ipynb



def compute_mse(e):
    """
    Calculate the mse for vector e.
    """
    return 1 / 2 * np.mean(e**2)

def compute_mae(e):
    """
    Calculate the mae for vector e.
    """
    return np.mean(np.abs(e))


def sigmoid(t):
    """
    Apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        value of the sigmoid function applied to t

    """

    return 1.0 / (1 + np.exp(-t))





def compute_mse_loss(y, tx, w):
    """
    Calculate the loss using MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    #verify the matrix computation will work
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_mae_loss(y, tx, w):
    """
    Calculate the loss using MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    #verify the matrix computation will work
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    e = y - tx.dot(w)
    return calculate_mae(e)


def compute_sigmoid_loss (y, tx, w):
     """
     Compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        the value of non-negative loss

    """
    #verify the matrix computation will work
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])





def compute_gradient(y, tx, w):
    """
    Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        (grad, err): array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    #verify the matrix computation will work
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_gradient_sigmoid(y, tx, w, lambda_= 0):
    """
    Compute the (penalized) gradient of loss using the sigmoid function

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        lambda_: float ; value in case the regression is penalized (≠0)

    Returns:
        a vector of shape (D, )

    """
    #verify the matrix computation will work
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0]) + 2 * lambda_ * w
    return grad






#TAKEN FROM AN OLD LAB 
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]
