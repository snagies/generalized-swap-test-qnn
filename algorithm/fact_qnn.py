import torch
import torch.nn as nn

class FactorizedQNNClassical(nn.Module):

    def __init__(self, N, k, d):
        super(FactorizedQNNClassical, self).__init__()
        self.N = N
        self.k = k
        self.d = d

        # parameters
        self.w = nn.Parameter(torch.randn(N, k, d + 1))  # w_ijz
        self.alpha = nn.Parameter(torch.randn(N))  # alpha_i
        self.beta = nn.Parameter(torch.randn(1))  # beta

    def forward(self, x):
        """
        x: shape (batch_size, d)
        """
        batch_size = x.shape[0]

        # add bias
        x_bias = torch.cat([x, torch.ones(batch_size, 1)], dim=1)

        # norms
        x_norm = torch.norm(x, dim=1, keepdim=True)  # (batch_size, 1)
        w_norm = torch.norm(self.w, dim=2, keepdim=True)  # (N, k, 1)

        # normalize
        x_normalized = x_bias / x_norm  # (batch_size, d+1)
        w_normalized = self.w / w_norm  # (N, k, d+1)

        # squared inner products over d
        squared_inner_products = torch.matmul(w_normalized, x_normalized.T) ** 2  # (N, k, batch_size)

        # product over k
        product_terms = torch.prod(squared_inner_products, dim=1)  # (N, batch_size)

        # weighted sum over N
        sum_terms = torch.matmul(self.alpha, product_terms)  # (batch_size,)

        #output
        y = sum_terms + self.beta

        return y
