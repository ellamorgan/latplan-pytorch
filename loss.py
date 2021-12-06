import torch
import torch.nn as nn
import torch.nn.functional as F

'''
All loss functions found here
'''

# Gumbel Softmax loss - formula provided in section 3.1.6 Gumbel Softmax
def gs_loss(logit_q, logit_p, eps=1e-20):
    q = F.softmax(logit_q, dim=-1)
    q = q / torch.sum(q, dim=-1, keepdim=True)
    p = F.softmax(logit_p, dim=-1)
    p = p / torch.sum(p, dim=-1, keepdim=True)
    log_q = torch.log(q + eps)
    log_p = torch.log(p + eps)
    loss = q * (log_q - log_p)
    loss_sum = torch.sum(loss)
    return loss_sum


# Binary Concrete loss - formula provided in section 3.1.7 Binary Concrete
# logit_p for logits (network outputs before activation), p is for the Bernoulli(p) prior
def bc_loss(logit_q, logit_p=None, p=None, eps=1e-20):
    if logit_p is None and p is None:
        raise ValueError("Both logit_p and p cannot be None")
    elif p is None:
        p = torch.sigmoid(logit_p)
    q = torch.sigmoid(logit_q)
    log_q0 = torch.log(q + eps)
    log_q1 = torch.log(1 - q + eps)
    log_p0 = torch.log(p + eps)
    log_p1 = torch.log(1 - p + eps)
    loss = q * (log_q0 - log_p0) + (1 - q) * (log_q1 - log_p1)
    loss_sum = torch.sum(loss)
    return loss_sum


# Follows equations given in section 3.1.8 Loss Functions
def total_loss(out, p, beta_z, beta_d, losses=None):

    # KL losses
    z0_prior = bc_loss(out['l_0'], p=p)
    z1_prior = bc_loss(out['l_1'], p=p)

    l0_l3 = bc_loss(out['l_0'], logit_p=out['l_3'])
    l1_l2 = bc_loss(out['l_1'], logit_p=out['l_2'])

    a_app = gs_loss(out['a_l'], out['app'])
    a_reg = gs_loss(out['a_l'], out['reg'])

    # Reconstruction losses
    criterion = nn.MSELoss()

    x0_recon = criterion(out['x_0'], out['x_dec_0'])
    x1_recon = criterion(out['x_1'], out['x_dec_1'])

    x0_x3 = criterion(out['x_0'], out['x_aae_3'])
    x1_x2 = criterion(out['x_1'], out['x_aae_2'])

    # Store losses for future plotting
    if losses is not None:
        losses['z0_prior'].append(z0_prior.detach().cpu().numpy())
        losses['z1_prior'].append(z1_prior.detach().cpu().numpy())
        losses['l0_l3'].append(l0_l3.detach().cpu().numpy())
        losses['l1_l2'].append(l1_l2.detach().cpu().numpy())
        losses['a_app'].append(a_app.detach().cpu().numpy())
        losses['a_reg'].append(a_reg.detach().cpu().numpy())
        losses['x0_recon'].append(x0_recon.detach().cpu().numpy())
        losses['x1_recon'].append(x1_recon.detach().cpu().numpy())
        losses['x0_x3'].append(x0_x3.detach().cpu().numpy())
        losses['x1_x2'].append(x1_x2.detach().cpu().numpy())
    
    # Follows formulas provided in paper
    forward_loss1 = beta_z * z0_prior + x0_recon + a_app + beta_d * l1_l2 + x1_recon
    forward_loss2 = beta_z * z0_prior + x0_recon + a_app + x1_x2
    backward_loss1 = beta_z * z1_prior + x1_recon + a_reg + beta_d * l0_l3 + x0_recon
    backward_loss2 = beta_z * z1_prior + x1_recon + a_reg + x0_x3
    total_loss = (forward_loss1 + forward_loss2 + backward_loss1 + backward_loss2) / 4

    recon_loss = (x0_recon + x1_recon + x0_x3 + x1_x2) / 4

    return total_loss, losses