import torch
import models
l=777
d=models.Discriminator(70, l)
x=torch.rand(16,l,70)
d(x)
