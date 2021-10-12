import torch
import torch.nn.functional as F

# logit_p and logit_q should be the outputs of the action network and applicable/regressable before activations
def gs_loss(self, logit_q, logit_p):       # logit_p, logit_q: (batch, f)
    q = F.softmax(logit_q)
    p = F.softmax(logit_p)
    # Normalize?
    log_q = torch.log(q)
    log_p = torch.log(p)
    loss = q * (log_q - log_p)
    loss_sum = torch.sum(loss)
    return loss_sum

def bc_loss(self, logit_q, logit_p=None, p=None):
    if logit_p is None and p is None:
        raise ValueError("Both logit_p and p cannot be None")
    elif p is None:
        p = F.sigmoid(logit_p)
    q = F.sigmoid(logit_q)
    log_q0 = torch.log(q)
    log_q1 = torch.log(1 - q)
    log_p0 = torch.log(p)
    log_p1 = torch.log(1 - p)
    loss = q0 * (log_q0 - log_p0) + (1 - q) * torch.log(log_q1 - log_p1)
    loss_sum = torch.sum(loss)
    return loss_sum