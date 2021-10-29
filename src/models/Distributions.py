# Please note that some code in this class builds upon work done by Kristoffer Vinther Olesen (@DTU)
import torch
from torch.distributions import Distribution


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A class that defines a multivariate isotropic Gaussian distribution (Sigma = sigma^2 I)

    N(y | mu, sigma) compatible with the reparameterization trick given epsilon ~ N(0, 1).
    Having a diagonal variance-covariance matrix drastically reduces the number of parameters

    ...

    Attributes
    ----------
    mu : float
        The Gaussian location (mean) parameter mu
    sigma : float
        The Guassian scale (standard deviation) parameter sigma >= 0

    Methods
    -------
    sample_epsilon()
        Sample random noise epsilon ~ N(0, I)

    sample()
        Sample from the distribtuion z ~ N(z | mu, sigma) (without gradients)

    rsample()
        Sample from the distribution z ~ N(z | mu, sigma) using the reparameterization trick
        to get way from the sampling operations that are non-differentiable and be able to
        apply backpropagation through a stochastic node

    log_prob()
        Computes the log probability log p(z) under this distribution
    """

    def __init__(self, mu, sigma):
        """
        Parameters
        ----------
        mu : Tensor
            Tensor of Gaussian location parameters mu. That is,
            each dimension has its own mean value
        sigma : Tensor
            Tensor of Guassian scale parameters sigma >= 0
        """
        assert (
            mu.shape == sigma.shape
        ), f"The tensors mu: {mu.shape} and sigma: {sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = sigma

    def sample_epsilon(self):
        """Sample random noise epsilon ~ N(0, I)"""
        return torch.empty_like(self.mu).normal_()

    def sample(self):
        """Sample from the distribtuion z ~ N(z | mu, sigma) (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self):
        """
        Sample from the distribution z ~ N(z | mu, sigma) using the reparameterization trick
        to get way from the sampling operations that are non-differentiable and be able to
        apply backpropagation through a stochastic node
        """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z):
        """
        Computes the log probability log p(z) under this distribution

        Parameters
        ----------
        z : Tensor
            Tensor for which to compute the log probability
        """
        dist = torch.distributions.Normal(loc=self.mu, scale=self.sigma)
        return dist.log_prob(z)
