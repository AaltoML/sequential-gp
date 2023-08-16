"""
Module to declare Gaussian Exponential Family sites objects.
"""

import abc
from typing import Optional

import tensorflow as tf
from gpflow.base import Module, Parameter
from gpflow.config import default_float
from gpflow.utilities import positive, triangular


class Sites(Module, metaclass=abc.ABCMeta):
    """
    The base sites class
    """

    def __init__(self, name: Optional[str] = None):
        """
        :param name: optional kernel name.
        """
        super().__init__(name=name)


class DiagSites(Sites):
    """
    Sites with diagonal lambda_2
    """

    def __init__(self, lambda_1, lambda_2, name: Optional[str] = None):
        """
        :param lambda_1: first order natural parameter
        :param lambda_2: second order natural parameter
        :param name: optional kernel name.
        """
        super().__init__(name=name)

        self.lambda_1 = Parameter(lambda_1, dtype=default_float(), trainable=False)  # [M, P]
        self.lambda_2 = Parameter(lambda_2, transform=positive(), trainable=False)  # [M, P]


class DenseSites(Sites):
    """
    Sites with dense lambda_2 save as a Cholesky factor
    """

    def __init__(self, lambda_1, lambda_2_sqrt=None, lambda_2=None, name: Optional[str] = None):
        """
        :param lambda_1: first order natural parameter
        :param lambda_2_sqrt: second order natural parameter
        :param name: optional kernel name.
        """
        super().__init__(name=name)

        self.lambda_1 = Parameter(lambda_1, dtype=default_float(), trainable=False)  # [M, P]
        self.num_latent_gps = lambda_1.shape[0]

        assert (lambda_2_sqrt is not None) or (lambda_2 is not None)

        if lambda_2_sqrt is not None:
            self.factor = True
            self._lambda_2_sqrt = Parameter(lambda_2_sqrt, transform=triangular(), trainable=False)  # [L|P, M, M]
        else:
            self._lambda_2 = Parameter(lambda_2, trainable=False)  # [L|P, M, M]
            self.factor = False

    @property
    def lambda_2(self):
        """second natural parameter"""
        if self.factor:
            return self._lambda_2_sqrt @ tf.linalg.matrix_transpose(self._lambda_2_sqrt)
        return self._lambda_2

    @property
    def lambda_2_sqrt(self):
        """Cholesky factor of the second natural parameter"""
        if self.factor:
            return self._lambda_2_sqrt
        return tf.linalg.cholesky(self._lambda_2)
