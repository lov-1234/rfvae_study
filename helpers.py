import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

        return False


def make_splits_and_loaders(
    dataset_cls, dataset_kwargs, batch_size=256, seed=42,
    num_workers=4, train_frac=0.8, val_frac=0.1
):
    full_dataset = dataset_cls(**dataset_kwargs)

    n_total = len(full_dataset)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

def permute_dims(z):
    batch_size, latent_dim = z.size()
    permuted_z = torch.zeros_like(z)
    for j in range(latent_dim):
        permuted_z[:, j] = z[torch.randperm(batch_size, device=z.device), j]    # Independently permute each dimension across the batch
    return permuted_z


def train_one_epoch_rfvae(
    model,
    discriminator,
    dataloader,
    vae_optimizer,
    discriminator_optimizer,
    r_logits,
    device,
    gamma=10.0,
    lambda_max=10.0,
    lambda_min=0.1,
    eta_s=1.0,
    eta_h=1.0,
    vae_scheduler=None,
    discriminator_scheduler=None,
):
    model.train()
    discriminator.train()

    total_loss = 0.0
    total_recon = 0.0
    total_weighted_kl = 0.0
    total_tc = 0.0
    total_r_l1 = 0.0
    total_r_entropy = 0.0
    total_d_loss = 0.0
    total_kl = 0.0

    total_encoder_kl_per_dim = None
    samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        batch_size = x.size(0)

        # ---------------------------------------------------
        # 1. Discriminator update
        # ---------------------------------------------------
        with torch.no_grad():
            _, mu_d, logvar_d, z_d = model(x)
            r_d = torch.sigmoid(r_logits)

        d_loss = discriminator_loss(
            discriminator=discriminator,
            z=z_d,
            r=r_d,
        )

        discriminator_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        discriminator_optimizer.step()

        if discriminator_scheduler is not None:
            discriminator_scheduler.step()

        # ---------------------------------------------------
        # 2. VAE + relevance vector update
        # ---------------------------------------------------
        for p in discriminator.parameters():
            p.requires_grad_(False)

        x_hat, mu, logvar, z = model(x)
        r = torch.sigmoid(r_logits)

        loss, recon, weighted_kl, tc, r_l1, r_entropy = rf_vae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            discriminator=discriminator,
            r=r,
            gamma=gamma,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            eta_s=eta_s,
            eta_h=eta_h,
        )

        vae_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        vae_optimizer.step()

        if vae_scheduler is not None:
            vae_scheduler.step()

        for p in discriminator.parameters():
            p.requires_grad_(True)

        # ---------------------------------------------------
        # Logging
        # ---------------------------------------------------
        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_weighted_kl += weighted_kl.item() * batch_size
        total_tc += tc.item() * batch_size
        total_r_l1 += r_l1.item() * batch_size
        total_r_entropy += r_entropy.item() * batch_size
        total_d_loss += d_loss.item() * batch_size

        standard_kl = kl_loss(mu, logvar)
        total_kl += standard_kl.item() * batch_size

        kl_per_dim_batch = kl_per_dim(mu, logvar).mean(dim=0)  # [D]

        if total_encoder_kl_per_dim is None:
            total_encoder_kl_per_dim = kl_per_dim_batch.detach() * batch_size
        else:
            total_encoder_kl_per_dim += kl_per_dim_batch.detach() * batch_size

        samples += batch_size

    avg_loss = total_loss / samples
    avg_recon = total_recon / samples
    avg_weighted_kl = total_weighted_kl / samples
    avg_tc = total_tc / samples
    avg_r_l1 = total_r_l1 / samples
    avg_r_entropy = total_r_entropy / samples
    avg_d_loss = total_d_loss / samples
    avg_kl = total_kl / samples
    avg_encoder_kl_per_dim = total_encoder_kl_per_dim / samples
    final_r = torch.sigmoid(r_logits).detach().cpu()

    print(f"Standard KL (TRAIN): {avg_kl:.4f}")
    print(f"r values (TRAIN): {final_r}")

    return (
        avg_loss,
        avg_recon,
        avg_weighted_kl,
        avg_tc,
        avg_r_l1,
        avg_r_entropy,
        avg_d_loss,
        avg_encoder_kl_per_dim.cpu(),
        final_r,
    )


def validate_rfvae(
    model,
    discriminator,
    dataloader,
    r_logits,
    device,
    gamma=10.0,
    lambda_max=10.0,
    lambda_min=0.1,
    eta_s=1.0,
    eta_h=1.0,
    desc="Validation",
):
    model.eval()
    discriminator.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_weighted_kl = 0.0
    total_tc = 0.0
    total_r_l1 = 0.0
    total_r_entropy = 0.0
    total_kl = 0.0

    total_encoder_kl_per_dim = None
    samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            batch_size = x.size(0)

            x_hat, mu, logvar, z = model(x)
            r = torch.sigmoid(r_logits)

            loss, recon, weighted_kl, tc, r_l1, r_entropy = rf_vae_loss(
                x=x,
                x_hat=x_hat,
                mu=mu,
                logvar=logvar,
                z=z,
                discriminator=discriminator,
                r=r,
                gamma=gamma,
                lambda_max=lambda_max,
                lambda_min=lambda_min,
                eta_s=eta_s,
                eta_h=eta_h,
            )

            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_weighted_kl += weighted_kl.item() * batch_size
            total_tc += tc.item() * batch_size
            total_r_l1 += r_l1.item() * batch_size
            total_r_entropy += r_entropy.item() * batch_size

            standard_kl = kl_loss(mu, logvar)
            total_kl += standard_kl.item() * batch_size

            kl_per_dim_batch = kl_per_dim(mu, logvar).mean(dim=0)

            if total_encoder_kl_per_dim is None:
                total_encoder_kl_per_dim = kl_per_dim_batch.detach() * batch_size
            else:
                total_encoder_kl_per_dim += kl_per_dim_batch.detach() * batch_size

            samples += batch_size

    avg_loss = total_loss / samples
    avg_recon = total_recon / samples
    avg_weighted_kl = total_weighted_kl / samples
    avg_tc = total_tc / samples
    avg_r_l1 = total_r_l1 / samples
    avg_r_entropy = total_r_entropy / samples
    avg_kl = total_kl / samples
    avg_encoder_kl_per_dim = total_encoder_kl_per_dim / samples
    final_r = torch.sigmoid(r_logits).detach().cpu()

    print(f"Standard KL (VALIDATION): {avg_kl:.4f}")
    print(f"r values (VALIDATION): {final_r}")

    return (
        avg_loss,
        avg_recon,
        avg_weighted_kl,
        avg_tc,
        avg_r_l1,
        avg_r_entropy,
        avg_encoder_kl_per_dim.cpu(),
        final_r,
    )


def train_pipeline_rfvae(
    model,
    discriminator,
    train_dataloader,
    val_dataloader,
    vae_optimizer,
    discriminator_optimizer,
    r_logits,
    device,
    epochs,
    gamma=10.0,
    lambda_max=10.0,
    lambda_min=0.1,
    eta_s=1.0,
    eta_h=1.0,
    vae_scheduler=None,
    discriminator_scheduler=None,
    early_stopping=None,
    scheduler_step_per_batch=True,
):
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_weighted_kl": [],
        "train_tc": [],
        "train_r_l1": [],
        "train_r_entropy": [],
        "train_d_loss": [],
        "train_encoder_kl_per_dim": [],
        "train_r": [],

        "val_loss": [],
        "val_recon": [],
        "val_weighted_kl": [],
        "val_tc": [],
        "val_r_l1": [],
        "val_r_entropy": [],
        "val_encoder_kl_per_dim": [],
        "val_r": [],
    }

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")

        (
            train_loss,
            train_recon,
            train_weighted_kl,
            train_tc,
            train_r_l1,
            train_r_entropy,
            train_d_loss,
            train_encoder_kl_per_dim,
            train_r,
        ) = train_one_epoch_rfvae(
            model=model,
            discriminator=discriminator,
            dataloader=train_dataloader,
            vae_optimizer=vae_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            r_logits=r_logits,
            device=device,
            gamma=gamma,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            eta_s=eta_s,
            eta_h=eta_h,
            vae_scheduler=vae_scheduler if scheduler_step_per_batch else None,
            discriminator_scheduler=discriminator_scheduler if scheduler_step_per_batch else None,
        )

        (
            val_loss,
            val_recon,
            val_weighted_kl,
            val_tc,
            val_r_l1,
            val_r_entropy,
            val_encoder_kl_per_dim,
            val_r,
        ) = validate_rfvae(
            model=model,
            discriminator=discriminator,
            dataloader=val_dataloader,
            r_logits=r_logits,
            device=device,
            gamma=gamma,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            eta_s=eta_s,
            eta_h=eta_h,
        )

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_weighted_kl"].append(train_weighted_kl)
        history["train_tc"].append(train_tc)
        history["train_r_l1"].append(train_r_l1)
        history["train_r_entropy"].append(train_r_entropy)
        history["train_d_loss"].append(train_d_loss)
        history["train_encoder_kl_per_dim"].append(train_encoder_kl_per_dim)
        history["train_r"].append(train_r)

        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_weighted_kl"].append(val_weighted_kl)
        history["val_tc"].append(val_tc)
        history["val_r_l1"].append(val_r_l1)
        history["val_r_entropy"].append(val_r_entropy)
        history["val_encoder_kl_per_dim"].append(val_encoder_kl_per_dim)
        history["val_r"].append(val_r)

        if vae_scheduler is not None and not scheduler_step_per_batch:
            vae_scheduler.step()

        if discriminator_scheduler is not None and not scheduler_step_per_batch:
            discriminator_scheduler.step()

        print(
            f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, "
            f"W-KL: {train_weighted_kl:.4f}, TC: {train_tc:.4f}, "
            f"R-L1: {train_r_l1:.4f}, R-H: {train_r_entropy:.4f}, "
            f"D: {train_d_loss:.4f}\n"
            f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, "
            f"W-KL: {val_weighted_kl:.4f}, TC: {val_tc:.4f}, "
            f"R-L1: {val_r_l1:.4f}, R-H: {val_r_entropy:.4f}"
        )

        if early_stopping is not None:
            early_stopping.step(val_loss, model)
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if early_stopping is not None and early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return history
