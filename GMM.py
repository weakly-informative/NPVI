import torch
from torch import nn
from torch import distributions
from torch.utils.data import Dataset
import pandas as pd
from sklearn import mixture
import numpy as np
from torch.distributions import MultivariateNormal



class ISO_GMM(nn.Module):
    # Mixture of equally weighted isotropic multivariate gaussians
    # K: number of Normal components
    # D: dimension of the random variable whose posterior we wish to approximate
    def __init__(self, K=4, D=2, spos=2):
        super(ISO_GMM, self).__init__()
        self.mu =  torch.empty((K, D)).uniform_(-1.0, 1.0)
        self.mu.requires_grad=True
 
        self.var = torch.empty(K).uniform_(0.0, 1.0) # (-5, -1)
        self.var.requires_grad=True
       
        self.weights = torch.ones(K, requires_grad=False) * 1.0/K
        self.var_map = nn.Softplus()
        self.K = K
        self.D = D

    def vars(self, k):
        return self.var_map(self.var[k]) # + 1e-6

    def log_prob(self, theta):
        lpdfs = torch.zeros((theta.shape[0], self.K))
        for k in range(self.K):
            lpdfs[:, k] = MultivariateNormal(self.mu[k, :], scale_tril=(self.vars(k) ** 0.5) * torch.eye(self.D) ).log_prob(theta)
        return torch.logsumexp(lpdfs, dim=1) / self.K

    def sample(self, dims):
        memberships = distributions.Categorical(probs=self.weights).sample(dims)
        ss = []
        for n in range(dims[0]):
            k = memberships[n]
            s = torch.randn(self.D) * ( self.vars(k) ** 0.5) + self.mu[k, :]
            ss.append(s)
        
        samples = torch.stack(ss).squeeze()
        return samples
