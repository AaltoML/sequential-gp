import numpy as np
import tensorflow as tf
from src.util import mean_cov_to_natural_param

def random_shuffle(x_batch, model, m):
    Z_old = model.inducing_variable.Z.numpy()
    Z_batch = np.concatenate([Z_old, x_batch], axis=0)
    np.random.shuffle(Z_batch)

    Z_new = Z_batch[:m]
    new_q_mu, new_f_cov = m.predict_f(Z_new, full_output_cov=True)
    K_zz = m.kernel(m.inducing_variable.Z)
    new_l1, new_l2 = mean_cov_to_natural_param(new_q_mu, new_f_cov, K_zz)

    return new_l1, new_l2, Z_new


# Sampling functions
def piv_chol(new_batch, old_batch, model, m_z, lamb=None, use_lamb=False):
    combined_batch = np.concatenate([old_batch, new_batch], axis=0)
    K_zb = model.kernel(combined_batch)

    if use_lamb is True:
        K_zb = np.diag(np.sqrt(lamb)) @ K_zb @ np.diag(np.sqrt(lamb))

    get_diag = lambda: np.diag(K_zb).copy()
    get_row = lambda i: K_zb[i, :]
    _, pi = pivoted_chol(get_diag, get_row, M=m_z)
    Z_new = combined_batch[pi]
    return Z_new, pi


def fixed_Z(new_batch, old_batch, model, m_z, update=True):
    Z_new = old_batch
    return Z_new, None


def update_lambda_Z_move(model, z_new, z_old):
    old_l1 = model.lambda_1
    old_l2 = model.lambda_2

    K_zf = model.kernel(z_old, z_new)
    A_p = tf.linalg.solve(model.kernel(z_old), K_zf)

    new_l1 = tf.transpose(A_p) @ old_l1
    new_l2 = tf.transpose(A_p) @ old_l2 @ A_p
    return new_l1, new_l2 # TODO: write test here for broadcasting


def pivoted_chol(get_diag, get_row, M, err_tol=1e-6):
    """
    A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive
    semi-definite operator.

    Args:
        - get_diag: A function which takes no arguments and returns the diagonal of the matrix when called.
        - get_row: A function which takes 1 integer argument and returns the desired row (zero indexed).
        - M: The maximum rank of the approximate decomposition; an integer.

    Returns: 
        - R, an upper triangular matrix of column dimension equal to the target matrix.
        - pi, the index of the pivots.
    """

    d = np.copy(get_diag())
    N = len(d)

    pi = list(range(N))

    R = np.zeros([M, N])


    m = 0
    while (m < M):  # and (err > err_tol):

        i = m + np.argmax([d[pi[j]] for j in range(m, N)])
        tmp = pi[m]
        pi[m] = pi[i]
        pi[i] = tmp

        R[m, pi[m]] = np.sqrt(d[pi[m]])
        Apim = get_row(pi[m])
        for i in range(m + 1, N):
            if m > 0:
                ip = np.inner(R[:m, pi[m]], R[:m, pi[i]])
            else:
                ip = 0
            R[m, pi[i]] = (Apim[pi[i]] - ip) / R[m, pi[m]]
            d[pi[i]] -= pow(R[m, pi[i]], 2)

        m += 1

    R = R[:m, :]
    return R, pi[:m]


def compute_lev(model, x_data, y_data):
    mean, f_varM = model.predict_f(x_data, full_cov=False, full_output_cov=False)
    with tf.GradientTape(persistent=True) as g:
        g.watch(mean)
        g.watch(f_varM)
        var_expI = model.likelihood.variational_expectations(mean, f_varM, y_data)
    d_exp_dv = g.gradient(var_expI, f_varM)
    del g

    lamb = tf.squeeze(-2*d_exp_dv)
    lev = tf.abs(tf.reduce_sum(f_varM * lamb, axis=1))

    if lamb.ndim > 1:
        lamb = tf.reduce_sum(lamb, axis=1)

    return lev.numpy(), lamb.numpy()


def memory_picker(old_batch, model, mem_size):  

    x_old, y_old = old_batch
    lev, lamb = compute_lev(model, x_old, y_old) 

    # Weighted sampling
    ind = np.random.choice(np.arange(y_old.shape[0]), mem_size, p=lev/np.sum(lev))
    return None, ind  


def random_picker(data, model, mem_size):
    """
    Picks random memory

    Note: model parameter is there for uniform function definition.
    """
    x_old, y_old = data
    ind = np.random.choice(np.arange(y_old.shape[0]), mem_size)
    return None, ind

