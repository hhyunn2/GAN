from PIL import Image

import os
import os.path as osp
import random
import torch
from torch import optim
import torch.nn.utils
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd

%matplotlib inline


#train
if __name__ == '__main__':

    for epoch in range(num_epoch):

      model_G.train()
      model_D.train()

      for i, (img, labels) in enumerate(train_loader):
          rand_idx = torch.randperm(labels.size(0))
          label_targets = labels[rand_idx]

          label = labels.clone()
          label_target = label_targets.clone()

          img = img.to(device)
          label = label.to(device)
          label_target = label_target.to(device)

          ### Update discriminator model
          model_D.zero_grad()

          real_img, real_label = model_D(img)
          d_loss_real = - torch.mean(real_img)

          classification_r_loss = F.binary_cross_entropy_with_logits(real_label, label) / real_label.size(0)

          out_img = model_G(img, label_target)
          fake_img, fake_label = model_D(out_img.detach())
          d_loss_fake = torch.mean(fake_img)

          alpha = torch.rand(img.size(0), 1, 1, 1).to(device)
          x_hat = (alpha * img.data + (1 - alpha) * out_img.data).requires_grad_(True)
          out_src, _ = model_D(x_hat)

          weight = torch.ones(out_src.size()).to(device)
          dydx = torch.autograd.grad(outputs=out_src, inputs=x_hat, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
          dydx = dydx.view(dydx.size(0), -1)
          dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
          d_loss_gp = torch.mean((dydx_l2norm-1)**2)

          loss_D = d_loss_real + d_loss_fake + lambda_cls * classification_r_loss + lambda_gp * d_loss_gp

          loss_D.backward()
          optimizer_D.step()

          ### Update generator model
          model_G.zero_grad()

          out_img_target = model_G(img, label_target)
          fake_img_target, fake_label_target = model_D(out_img_target)
          g_loss_fake = -torch.mean(fake_img_target)
          classification_f_loss = F.binary_cross_entropy_with_logits(fake_label_target, label_target) / fake_label_target.size(0)

          out_reconstruction = model_G(out_img_target, label)
          reconstruction_loss = torch.mean(torch.abs(img - out_reconstruction))

          loss_G = g_loss_fake + lambda_rec * reconstruction_loss + lambda_cls * classification_f_loss

          loss_G.backward()
          optimizer_G.step()

          if i % 2000 == 0 and i != 0:
              print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, num_epoch, i, len(train_loader),
                       loss_D.item(), loss_G.item()))
              torch.save({
                  'model_G': model_G.state_dict()
              }, '/content/drive/My Drive/StarGAN/generator_more.pt')
