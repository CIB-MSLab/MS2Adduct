"""
train_utils.py
Training/validation/test utilities shared by the final model entry point
(train_dreaMS_harmonictsfm_2.py): average meter, focal loss, ion auxiliary
loss, and the live (finetune/scratch) loops.

Only the live path is kept: the final model needs raw peaks for the delta
branch, so precomputed-embedding cache mode is not supported.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              top_k_accuracy_score)
from tqdm import tqdm
from torch.cuda.amp import autocast

import config


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


# 离子模式 aux loss 固定用 plain CE (2类基本均衡, 不受 9 类 focal/alpha 影响)
ION_CE = nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)"""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def _hierarchical(model):
    return hasattr(model, 'ion_mode_head') and model.ion_mode_head is not None


def _polarity_kwargs(model, batch, device):
    """离子模式 (正/负) 是采集时已知的输入, 不是预测目标 (见 config 的说明块)。

    只有显式声明 `accepts_polarity` 的模型才收到它 —— 其余模型的前向签名不含
    polarity, 保持逐行不变。
    """
    if not getattr(model, 'accepts_polarity', False):
        return {}
    pol = batch.get('polarity')
    if pol is None:
        return {}
    return {'polarity': pol.to(device)}


def train_one_epoch_live(model, loader, optimizer, criterion, device, scaler):
    model.train(); losses = AverageMeter(); correct = total = 0
    pbar = tqdm(loader, desc="[Train live]", leave=False,
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for batch in pbar:
        mz = batch['mz'].to(device); intensity = batch['intensity'].to(device)
        labels = batch['label'].to(device); ce = batch['ce'].to(device)
        precursor_mz = batch['precursor_mz'].squeeze(-1).to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(mz, intensity, ce=ce, precursor_mz=precursor_mz,
                           **_polarity_kwargs(model, batch, device))
            loss = criterion(logits, labels)
            if _hierarchical(model):
                ion_logits = model.get_ion_logits()
                ion_labels = model.ion_mode_labels(labels)
                loss = loss + config.ION_LOSS_WEIGHT * ION_CE(ion_logits, ion_labels)
        if torch.isnan(loss): continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), mz.size(0))
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        pbar.set_postfix(L=f"{losses.avg:.3f}", A=f"{correct/total:.3f}")
    return losses.avg, correct / total if total > 0 else 0.0


@torch.no_grad()
def validate_live(model, loader, criterion, device):
    model.eval(); losses = AverageMeter(); all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="[Val live]", leave=False,
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for batch in pbar:
        mz = batch['mz'].to(device); intensity = batch['intensity'].to(device)
        labels = batch['label'].to(device); ce = batch['ce'].to(device)
        precursor_mz = batch['precursor_mz'].squeeze(-1).to(device)
        logits = model(mz, intensity, ce=ce, precursor_mz=precursor_mz,
                       **_polarity_kwargs(model, batch, device))
        loss = criterion(logits, labels)
        losses.update(loss.item(), mz.size(0))
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist()); all_labels.extend(labels.cpu().tolist())
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix(L=f"{losses.avg:.3f}", A=f"{acc:.3f}")
    acc = accuracy_score(all_labels, all_preds)
    # labels 显式给全 9 类: 若某类在本 split 中零样本, sklearn 默认会把它整个
    # 剔出 macro 平均, 同一个模型的 F1 会凭空变高且与其他数据集不可比。
    f1 = f1_score(all_labels, all_preds, average='macro',
                  labels=list(range(config.NUM_ADDUCT_CLASSES)), zero_division=0)
    return losses.avg, acc, f1, all_preds, all_labels


@torch.no_grad()
def test_evaluation_live(model, loader, criterion, device, log_file=None):
    model.eval(); all_preds, all_labels, all_logits = [], [], []
    test_losses = AverageMeter()
    pbar = tqdm(loader, desc="[Test live]", leave=False)
    for batch in pbar:
        mz = batch['mz'].to(device); intensity = batch['intensity'].to(device)
        labels = batch['label'].to(device); ce = batch['ce'].to(device)
        precursor_mz = batch['precursor_mz'].squeeze(-1).to(device)
        logits = model(mz, intensity, ce=ce, precursor_mz=precursor_mz,
                       **_polarity_kwargs(model, batch, device))
        loss = criterion(logits, labels)
        test_losses.update(loss.item(), mz.size(0))
        all_logits.append(logits.cpu())
        preds = logits.argmax(dim=-1).cpu()
        all_preds.extend(preds.tolist()); all_labels.extend(labels.cpu().tolist())
    return _print_test_results(all_logits, all_labels, all_preds, test_losses, log_file)


def _print_test_results(all_logits, all_labels, all_preds, test_losses, log_file):
    all_logits = torch.cat(all_logits)
    test_acc = accuracy_score(all_labels, all_preds)
    n_classes = config.NUM_ADDUCT_CLASSES
    test_f1 = f1_score(all_labels, all_preds, average='macro',
                       labels=list(range(n_classes)), zero_division=0)
    top3 = top_k_accuracy_score(np.array(all_labels), all_logits.numpy(),
                                k=3, labels=range(n_classes))
    lines = ["=" * 60, "Test Evaluation", "=" * 60,
             f"Test Loss:         {test_losses.avg:.4f}",
             f"Test Accuracy:     {test_acc:.4f}",
             f"Test Macro F1:     {test_f1:.4f}",
             f"Test Top-3 Acc:    {top3:.4f}", "",
             "Per-class F1:"]
    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                            labels=list(range(n_classes)), zero_division=0)
    for i, f1_val in enumerate(per_class_f1):
        name = config.IDX_TO_ADDUCT[i]
        count = (np.array(all_labels) == i).sum()
        lines.append(f"  {name:20s}: F1={f1_val:.4f} (n={count})")
    lines.append("")
    lines.append("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    header = "         " + "".join(
        f"{config.IDX_TO_ADDUCT[i][:6]:>8s}" for i in range(n_classes))
    lines.append(header)
    for i in range(n_classes):
        row = "".join(f"{cm[i, j]:8d}" for j in range(n_classes))
        lines.append(f"{config.IDX_TO_ADDUCT[i]:>8s} {row}")
    lines.append("=" * 60)
    for line in lines: print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return test_acc, test_f1
