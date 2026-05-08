import torch.nn.functional as F
import torch
import math
import torch.nn as nn
from einops import rearrange
from helpers import permute_dims

eps = 1e-8


def discriminator_loss(discriminator, z, r=None):
    z = z.detach()
    z_perm = permute_dims(z).detach()

    if r is not None:
        r = r.detach()
        z_joint = r * z
        z_perm = r * z_perm
    else:
        z_joint = z

    joint_logits = discriminator(z_joint)
    perm_logits = discriminator(z_perm)

    joint_targets = torch.ones_like(joint_logits)
    perm_targets = torch.zeros_like(perm_logits)

    loss_joint = F.binary_cross_entropy_with_logits(
        joint_logits, joint_targets)
    loss_perm = F.binary_cross_entropy_with_logits(perm_logits, perm_targets)

    return loss_joint + loss_perm


def tc_loss(z, discriminator, r=None):
    if r is not None:
        z = r * z
    return discriminator(z).mean()

def entropic_loss(r):
    return -torch.sum(
        r * torch.log(r + eps)
        + (1.0 - r) * torch.log(1.0 - r + eps)
    )

def kl_per_dim(mu, logvar):
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())


def kl_loss(mu, logvar):
    kl = kl_per_dim(mu, logvar)   # [B, D]
    return kl.sum(dim=1).mean()


def relevance_weighted_kl_loss(
    mu,
    logvar,
    r,
    lambda_max=10.0,
    lambda_min=0.1,
):
    kl = kl_per_dim(mu, logvar)  # [B, D]
    lambda_r = lambda_max - r * (lambda_max - lambda_min)  # [D]
    weighted_kl = lambda_r * kl  # broadcasts [D] over [B, D]
    return weighted_kl.sum(dim=1).mean()


def reconstruction_loss(x_logits, x, reduction="sum"):
    if reduction == "sum":
        return F.binary_cross_entropy_with_logits(
            x_logits,
            x,
            reduction="sum",
        ) / x.size(0)
    elif reduction == "mean":
        return F.binary_cross_entropy_with_logits(
            x_logits,
            x,
            reduction="mean",
        )
    else:
        raise ValueError("reduction must be either 'sum' or 'mean'")

def r_regularization_loss(r):
    return torch.sum(r)


def rf_vae_loss(
    x,
    x_hat,
    mu,
    logvar,
    z,
    discriminator,
    r,
    gamma=10.0,
    lambda_max=10.0,
    lambda_min=0.1,
    eta_s=1.0,
    eta_h=1.0,
):
    """
    Full RF-VAE objective.

    L = recon
        + relevance-weighted KL
        + gamma * TC(r * z)
        + eta_s * ||r||_1
        + eta_h * H(r)
    """
    recon = reconstruction_loss(x_hat, x)

    weighted_kl = relevance_weighted_kl_loss(
        mu=mu,
        logvar=logvar,
        r=r,
        lambda_max=lambda_max,
        lambda_min=lambda_min,
    )

    tc = tc_loss(z, discriminator, r=r)

    r_l1 = r_regularization_loss(r)
    r_entropy = entropic_loss(r)

    loss = (
        recon
        + weighted_kl
        + gamma * tc
        + eta_s * r_l1
        + eta_h * r_entropy
    )

    return loss, recon, weighted_kl, tc, r_l1, r_entropy
