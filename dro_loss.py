import torch
import torch.nn as nn

class LossComputer:
    def __init__(self, criterion, is_robust, dataset, step_size, device, args):
        self.criterion = criterion
        self.is_robust = is_robust
        self.step_size = step_size
        self.device = device
        self.args = args

        if hasattr(criterion, 'reduction') and criterion.reduction != 'none':
            raise ValueError("criterion harus diinisialisasi dengan reduction='none'")

        self.n_groups = dataset.n_groups
        try:
            self.group_counts = dataset.group_counts().to(device)
        except AttributeError:
            print("dataset.group_counts() tidak ditemukan, menggunakan distribusi seragam.")
            self.group_counts = torch.ones(self.n_groups).to(device)

        self.alpha = getattr(args, 'alpha', 0.2)
        if self.alpha > 0:
            self.adj = self.alpha / torch.sqrt(self.group_counts)
        else:
            self.adj = torch.zeros(self.n_groups).to(device)

        self.adv_probs = torch.ones(self.n_groups).to(device) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(device)
        self.gamma = getattr(args, 'gamma', 0.1)

    def loss(self, logits, y, group_idx, is_training=True):
        if group_idx.dtype != torch.long:
            group_idx = group_idx.long()
        
        per_sample_losses = self.criterion(logits, y)
        
        group_losses = torch.zeros(self.n_groups).to(self.device)
        group_counts = torch.zeros(self.n_groups).to(self.device)

        for g in range(self.n_groups):
            mask = (group_idx == g)
            if mask.any():
                group_losses[g] = per_sample_losses[mask].mean()
                group_counts[g] = mask.sum().item()

        if is_training:
            for g in range(self.n_groups):
                if group_counts[g] > 0:
                    self.exp_avg_loss[g] = (1 - self.gamma) * self.exp_avg_loss[g] + \
                                           self.gamma * group_losses[g].data
            
            if self.is_robust:
                self.adv_probs = self.adv_probs * torch.exp(self.step_size * (self.exp_avg_loss + self.adj))
                self.adv_probs = self.adv_probs / self.adv_probs.sum()

        if self.is_robust:
            actual_loss = torch.sum(self.adv_probs * group_losses)
        else:
            actual_loss = per_sample_losses.mean()

        return actual_loss

