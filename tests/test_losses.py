import torch
import pytest
from easyppisp.losses import (
    exposure_mean_loss,
    vignetting_center_loss,
    vignetting_non_pos_loss,
    vignetting_channel_var_loss,
    color_mean_loss,
    crf_channel_var_loss,
)

def test_exposure_loss():
    exp = torch.tensor([0.1, -0.1, 0.2])
    loss = exposure_mean_loss(exp)
    assert loss >= 0
    
    zero_exp = torch.zeros(3)
    assert exposure_mean_loss(zero_exp) == 0.0

def test_vignetting_losses():
    center = torch.tensor([0.1, -0.2])
    assert vignetting_center_loss(center) > 0
    assert vignetting_center_loss(torch.zeros(2)) == 0.0
    
    alpha = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert vignetting_non_pos_loss(alpha) > 0 # due to 0.1
    
    alpha_var = torch.tensor([[[0.1, 1.0, 2.0], [0.2, 1.0, 2.0], [0.3, 1.0, 2.0]]])
    assert vignetting_channel_var_loss(alpha_var) > 0

def test_color_loss():
    latent = torch.zeros(8)
    assert color_mean_loss(latent) == 0.0
    
    latent_nonzero = torch.ones(8) * 0.1
    assert color_mean_loss(latent_nonzero) > 0

def test_crf_loss():
    tau = torch.zeros(3)
    eta = torch.zeros(3)
    xi = torch.zeros(3)
    gamma = torch.zeros(3)
    assert crf_channel_var_loss(tau, eta, xi, gamma) == 0.0
    
    tau_var = torch.tensor([0.1, 0.0, 0.0])
    assert crf_channel_var_loss(tau_var, eta, xi, gamma) > 0
