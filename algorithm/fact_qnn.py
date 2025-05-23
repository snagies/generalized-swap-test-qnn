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
        device = x.device

        batch_size = x.shape[0]

        # add bias
        x_bias = torch.cat([x, torch.ones(batch_size, 1, device=device)], dim=1)

        # norms
        x_norm = torch.norm(x_bias, dim=1, keepdim=True)  # (batch_size, 1)
        w_norm = torch.norm(self.w, dim=2, keepdim=True)  # (N, k, 1)

        # normalize
        x_normalized = x_bias / x_norm  # (batch_size, d+1)
        w_normalized = self.w / w_norm  # (N, k, d+1)

        # squared inner products over d
        squared_inner_products = torch.matmul(w_normalized, x_normalized.T) ** 2  # (N, k, batch_size)

        #m = nn.Dropout(p=0.2)
        #squared_inner_products = m(squared_inner_products)

        # product over k
        product_terms = torch.prod(squared_inner_products, dim=1)  # (N, batch_size)

        # weighted sum over N
        sum_terms = torch.matmul(self.alpha, product_terms)  # (batch_size,)

        #output
        y = sum_terms + self.beta

        return y


class FactorizedQNNClassical_split(nn.Module):

    def __init__(self, Nsplit, k, d, quadrants = 4):
        super(FactorizedQNNClassical_split, self).__init__()
        self.k = k
        self.d = d
        self.split = quadrants #Image is reordered into quadrants, e.g. for quadrants = 4: 1st quarter of features corresponds to top left image sector and so on
        self.Nsplit = Nsplit #Each quadrant is fed into Nsplit different product modules, each factor module within gets the same quadrant as input
        assert self.d % self.split == 0
        
        self.split_d = int(self.d / self.split) #calculate number of features in each quadrant

        # parameters
        self.w = nn.Parameter(torch.randn(self.split, self.Nsplit, k, self.split_d + 1))  # w_ijz, 
        self.alpha = nn.Parameter(torch.randn(self.split * self.Nsplit))  # alpha_i, total of self.split * self.Nsplit product modules
        self.beta = nn.Parameter(torch.randn(1))  # beta

    def forward(self, x):
        """
        x: shape (batch_size, d)
        """
        device = x.device
        batch_size = x.shape[0]
        
        x_splits = torch.split(x, self.split_d, dim=1)  # tuple, shape (batch_size, split_d)

        split_results = []
        for i, x_part in enumerate(x_splits):
            x_bias = torch.cat([x_part, torch.ones(batch_size, 1, device=device)], dim=1)   #add dummy feature to each quadrant seperately
            w_part_selected = self.w[i, :, :, :]  # Shape: (Nsplit, k, self.split_d+1)
            
            x_norm = torch.norm(x_bias, dim=1, keepdim=True)  # (batch_size, 1)
            w_norm = torch.norm(w_part_selected, dim=2, keepdim=True)  # (Nsplit, k, 1)
            
            x_normalized = x_bias / x_norm  # (batch_size, split_d+1)
            w_normalized = w_part_selected / w_norm  # (Nsplit, k, split_d+1)
            
            squared_inner_products = torch.matmul(w_normalized, x_normalized.T) ** 2  # (Nsplit, k, batch_size) doublecheck if this is correct
            
            product_terms_split = torch.prod(squared_inner_products, dim=1) # (Nsplit, batch_size)
            
            split_results.append(product_terms_split)
                
        total_product_terms = torch.cat(split_results, dim=0) # (N * Nsplit, batch_size)
        
        sum_terms = torch.matmul(self.alpha, total_product_terms)  # (batch_size,)

        y = sum_terms + self.beta

        return y