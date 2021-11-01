import torch
import torch.nn as nn
from torchvision.models import resnet50


class BarlowTwins(nn.Module):
  def __init__(self, in_features, proj_channels):
    super(BarlowTwins, self).__init__()
    self.encoder = resnet50(zero_init_residual=True,)
    self.encoder.fc = nn.Identity()
    proj_layers = []

    #Creating the 3-layer projector
    for i in range(3):
      if (i==0):
          proj_layers.append(nn.Linear(in_features, proj_channels, bias=False ))
      else:
          proj_layers.append(nn.Linear(proj_channels, proj_channels, bias = False ))
      if(i<2):
        proj_layers.append(nn.BatchNorm1d(proj_channels))
        proj_layers.append(nn.ReLU(inplace=True))

    self.proj = nn.Sequential(*proj_layers)
    self.bn= nn.BatchNorm1d(proj_channels, affine=False)

  def forward(self, x1, x2):
    #Feeding the data through the encoder and projector
    z1 = self.proj(self.encoder(x1))
    z2= self.proj(self.encoder(x2))
    
    
    
    return z1, z2


class BarlowTwins_FT(nn.Module):
  def __init__(self, barlow_twins_model, z_dim, num_cat):
    super(BarlowTwins_FT, self).__init__()
    
    # Get the trained BarlowTwins
    self.bt = barlow_twins_model

    # Apply a linear layer that will be trained to 
    # evaluate the embeddings obtained with BarlowTwins
    self.linear = nn.Linear(z_dim, num_cat)

  def forward(self, x):
    
    x1, x2 = self.bt(x, x)
    out = self.linear(x1)
    
    return out


def loss_fun(z1, z2, lmbda):

  #Normalize the projector's output across the batch
  norm_z1 = (z1 - z1.mean(0))/ z1.std(0)
  norm_z2 = (z2 - z2.mean(0))/ z2.std(0)

  #Cross correlation matrix
  batch_size = z1.size(0)
  cc_M = torch.einsum('bi,bj->ij', (norm_z1, norm_z2)) / batch_size

  #Invariance loss
  diag = torch.diagonal(cc_M)
  invariance_loss = ((torch.ones_like(diag) - diag) ** 2).sum()

  #Zero out the diag elements and flatten the matrix to compute the loss
  cc_M.fill_diagonal_(0)
  redundancy_loss = (cc_M.flatten() ** 2 ).sum()
  loss = invariance_loss + lmbda * redundancy_loss

  return loss