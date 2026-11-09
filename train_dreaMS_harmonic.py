"""
train_dreaMS_harmonic.py
DreaMS 质谱编码 + 中性丢失 Harmonic → Adduct 分类 (9类) 训练脚本。
逻辑与 train_dreaMS.py 完全一致 (同一套 hierarchical loss / 选优 / 早停 / 评估),
仅把模型换成 DreaMSAdductHarmonicClassifier (DreaMS backbone + δ harmonic concat),
实验前缀 MS2ADDUCT_DreaMSH_ 以便与 DreaMS_XXX 系列区分。

DREAMS_MODE:
  finetune / scratch : 实时编码 + harmonic (推荐, 需要 precursor_mz)
  cache              : 预计算嵌入 (无原始峰, harmonic 分支退化为补零 → 不建议)
"""
import os, argparse, random, shutil, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              top_k_accuracy_score)
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import config
from model_dreaMS_harmonic import DreaMSAdductHarmonicClassifier
from dataloader import get_dataloader
from torch.utils.data import Dataset, DataLoader


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


class EmbeddingDataset(Dataset):
    """Loads precomputed DreaMS embeddings from .npz."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.ce = torch.tensor(data['ce'], dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'emb': self.embeddings[idx], 'label': self.labels[idx],
                'ce': self.ce[idx]}


def make_next_record_dir(base_dir, prefix="MS2ADDUCT_DreaMSH_"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    indices = []
    for d in existing:
        try: indices.append(int(d[len(prefix):].split("_")[0]))
        except ValueError: continue
    max_idx = max(indices, default=-1)
    full_path = os.path.join(base_dir, f"{prefix}{max_idx + 1:03d}")
    os.makedirs(full_path, exist_ok=True)
    return full_path


def backup_project_files(dest_dir):
    files = ["config.py", "model_dreaMS.py", "model_dreaMS_harmonic.py",
             "train_dreaMS_harmonic.py", "dataloader.py", "model.py"]
    backup_path = os.path.join(dest_dir, "codes")
    os.makedirs(backup_path, exist_ok=True)
    for f in files:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(backup_path, os.path.basename(f)))
    print(f"[*] Code backed up to: {backup_path}")


def get_lr_scheduler(optimizer, warmup_epochs, gamma):
    def lr_lambda(epoch):
        if epoch < warmup_epochs: return float(epoch + 1) / float(warmup_epochs)
        return gamma ** (epoch - warmup_epochs)
    return LambdaLR(optimizer, lr_lambda)


def _hierarchical(model):
    return hasattr(model, 'ion_mode_head') and model.ion_mode_head is not None


def _polarity_kwargs(model, batch, device):
    """离子模式 (正/负) 是采集时已知的输入, 不是预测目标 (见 config 的说明块)。

    只有显式声明 `accepts_polarity` 的模型才收到它 —— 其余模型 (如
    DreaMSAdductHarmonicClassifier) 的前向签名不含 polarity, 保持逐行不变。
    """
    if not getattr(model, 'accepts_polarity', False):
        return {}
    pol = batch.get('polarity')
    if pol is None:
        return {}
    return {'polarity': pol.to(device)}


def train_one_epoch_cache(model, loader, optimizer, criterion, device, scaler=None):
    model.train(); losses = AverageMeter(); correct = total = 0
    pbar = tqdm(loader, desc="[Train cache]", leave=False,
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for batch in pbar:
        emb = batch['emb'].to(device); labels = batch['label'].to(device)
        ce = batch['ce'].to(device)
        optimizer.zero_grad()
        logits = model(None, None, ce=ce, precomputed_emb=emb)
        loss = criterion(logits, labels)
        if _hierarchical(model):
            ion_logits = model.get_ion_logits()
            ion_labels = model.ion_mode_labels(labels)
            loss = loss + config.ION_LOSS_WEIGHT * ION_CE(ion_logits, ion_labels)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.update(loss.item(), emb.size(0))
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        pbar.set_postfix(L=f"{losses.avg:.3f}", A=f"{correct/total:.3f}")
    return losses.avg, correct / total if total > 0 else 0.0


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
def validate_cache(model, loader, criterion, device):
    model.eval(); losses = AverageMeter(); all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="[Val cache]", leave=False,
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for batch in pbar:
        emb = batch['emb'].to(device); labels = batch['label'].to(device)
        ce = batch['ce'].to(device)
        logits = model(None, None, ce=ce, precomputed_emb=emb)
        loss = criterion(logits, labels)
        losses.update(loss.item(), emb.size(0))
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist()); all_labels.extend(labels.cpu().tolist())
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix(L=f"{losses.avg:.3f}", A=f"{acc:.3f}")
    acc = accuracy_score(all_labels, all_preds)
    # labels 显式给全 9 类: 若某类在本 split 中零样本 (如 enveda 的 [M]+), sklearn 默认
    # 会把它整个剔出 macro 平均, 同一个模型的 F1 会凭空变高且与其他数据集不可比。
    f1 = f1_score(all_labels, all_preds, average='macro',
                  labels=list(range(config.NUM_ADDUCT_CLASSES)), zero_division=0)
    return losses.avg, acc, f1, all_preds, all_labels


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
    # labels 显式给全 9 类: 若某类在本 split 中零样本 (如 enveda 的 [M]+), sklearn 默认
    # 会把它整个剔出 macro 平均, 同一个模型的 F1 会凭空变高且与其他数据集不可比。
    f1 = f1_score(all_labels, all_preds, average='macro',
                  labels=list(range(config.NUM_ADDUCT_CLASSES)), zero_division=0)
    return losses.avg, acc, f1, all_preds, all_labels


@torch.no_grad()
def test_evaluation_cache(model, loader, criterion, device, log_file=None):
    model.eval(); all_preds, all_labels, all_logits = [], [], []
    test_losses = AverageMeter()
    pbar = tqdm(loader, desc="[Test cache]", leave=False)
    for batch in pbar:
        emb = batch['emb'].to(device); labels = batch['label'].to(device)
        ce = batch['ce'].to(device)
        logits = model(None, None, ce=ce, precomputed_emb=emb)
        loss = criterion(logits, labels)
        test_losses.update(loss.item(), emb.size(0))
        all_logits.append(logits.cpu())
        preds = logits.argmax(dim=-1).cpu()
        all_preds.extend(preds.tolist()); all_labels.extend(labels.cpu().tolist())
    return _print_test_results(all_logits, all_labels, all_preds, test_losses, log_file)


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


def main():
    parser = argparse.ArgumentParser(description='DreaMS + δ(MLP/mean-pool) + fusion')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=['concat', 'hadamard', 'gated', 'ffn',
                                 'gated_cat', 'bilinear', 'film', 'coattn'])
    parser.add_argument('--fraction', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--prefix', type=str, default="MS2ADDUCT_DreaMSH_")
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0,
                        help='训练随机种子, 默认 0 (固定以保证可复现); 多 seed 实验显式传 1/2/3. '
                             '注意: 数据划分固定用 42 (data/split_by_smiles.py), 与本种子是独立的两套')
    args = parser.parse_args()
    if args.fusion is not None:
        config.FUSION = args.fusion
    if args.fraction is not None:
        config.DATA_FRACTION = args.fraction
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.warmup is not None:
        config.WARMUP_EPOCHS = args.warmup
    if args.patience is not None:
        config.EARLY_STOP_PATIENCE = args.patience
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LR = args.lr
    if args.mode is not None:
        config.DREAMS_MODE = args.mode

    # ── 固定随机种子 (默认 seed 0, 保证可复现; 多 seed 实验显式传 1/2/3...) ──
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = config.DEVICE
    print(f"Device: {device}")
    print(f"DreaMS mode: {config.DREAMS_MODE}")
    print(f"HIERARCHICAL_HEADS: {config.HIERARCHICAL_HEADS} | "
          f"harmonic bases: {len(config.HARMONIC_BASE_FREQUENCIES)} | "
          f"n_harmonics/base: {config.N_HARMONICS_PER_BASE} | "
          f"FUSION: {config.FUSION} | fraction: {config.DATA_FRACTION}")

    experiment_dir = make_next_record_dir("ckpt", prefix=args.prefix)
    print(f"[*] Experiment dir: {experiment_dir}")
    backup_project_files(experiment_dir)
    log_file = os.path.join(experiment_dir, "trainlog.txt")

    mode = config.DREAMS_MODE
    if mode == "cache":
        train_ds = EmbeddingDataset(
            os.path.join(config.DREAMS_CACHE_DIR, "train.npz"))
        val_ds = EmbeddingDataset(
            os.path.join(config.DREAMS_CACHE_DIR, "val.npz"))
        test_ds = EmbeddingDataset(
            os.path.join(config.DREAMS_CACHE_DIR, "test.npz"))
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                                shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=0, pin_memory=True)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        model = DreaMSAdductHarmonicClassifier(
            ckpt_path=os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt"),
            freeze_backbone=True).to(device)
        train_fn = train_one_epoch_cache
        val_fn = validate_cache
        test_fn = test_evaluation_cache
        scaler = None
    elif mode == "finetune":
        train_loader = get_dataloader(config.TRAIN_PKL, config.BATCH_SIZE,
                                      augment=True, shuffle=True)
        val_loader = get_dataloader(config.VAL_PKL, config.BATCH_SIZE,
                                    augment=False, shuffle=False)
        test_loader = get_dataloader(config.TEST_PKL, config.BATCH_SIZE,
                                     augment=False, shuffle=False)
        print(f"Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, "
              f"Test: {len(test_loader.dataset)}")
        model = DreaMSAdductHarmonicClassifier(
            ckpt_path=os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt"),
            freeze_backbone=False).to(device)
        train_fn = train_one_epoch_live
        val_fn = validate_live
        test_fn = test_evaluation_live
        scaler = GradScaler()
    elif mode == "scratch":
        train_loader = get_dataloader(config.TRAIN_PKL, config.BATCH_SIZE,
                                      augment=True, shuffle=True)
        val_loader = get_dataloader(config.VAL_PKL, config.BATCH_SIZE,
                                    augment=False, shuffle=False)
        test_loader = get_dataloader(config.TEST_PKL, config.BATCH_SIZE,
                                     augment=False, shuffle=False)
        print(f"Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, "
              f"Test: {len(test_loader.dataset)}")
        model = DreaMSAdductHarmonicClassifier(
            ckpt_path=None, freeze_backbone=False).to(device)
        train_fn = train_one_epoch_live
        val_fn = validate_live
        test_fn = test_evaluation_live
        scaler = GradScaler()
    else:
        raise ValueError(f"Unknown DREAMS_MODE: {mode}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_harm = sum(p.numel() for p in model.harmonic_enc.parameters())
    print(f"Model params: {n_params/1e6:.2f}M (trainable: {n_trainable/1e6:.2f}M) | "
          f"harmonic: {n_harm/1e3:.1f}K")

    # ── 损失: CE 或 class-balanced focal (9 类) ──
    use_focal = getattr(config, 'USE_FOCAL', False) and mode != 'cache'
    if use_focal:
        cnt = Counter()
        for s in train_loader.dataset.samples:
            ad = s.get('adduct', '')
            if ad in config.ADDUCT_TO_IDX:
                cnt[config.ADDUCT_TO_IDX[ad]] += 1
        alpha = torch.zeros(config.NUM_ADDUCT_CLASSES)
        for i in range(config.NUM_ADDUCT_CLASSES):
            alpha[i] = 1.0 / max(cnt.get(i, 1), 1)
        alpha = alpha / alpha.sum() * config.NUM_ADDUCT_CLASSES
        criterion = FocalLoss(gamma=getattr(config, 'FOCAL_GAMMA', 2.0),
                              alpha=alpha.to(device))
        print(f"[Loss] FocalLoss gamma={getattr(config,'FOCAL_GAMMA',2.0)} "
              f"alpha={alpha.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[Loss] CrossEntropyLoss")

    optimizer = AdamW(model.parameters(), lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY)
    warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 5)
    lr_gamma = getattr(config, 'LR_GAMMA', 0.98)
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, lr_gamma)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = getattr(config, 'EARLY_STOP_PATIENCE', 5)
    counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_fn(model, train_loader, optimizer,
                                         criterion, device, scaler=scaler)
        val_loss, val_acc, val_f1, _, _ = val_fn(model, val_loader, criterion,
                                                  device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        msg = (f"Epoch {epoch:3d}/{config.EPOCHS} | LR: {current_lr:.6f} | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss; improved = True
        if val_acc > best_val_acc:
            best_val_acc = val_acc; improved = True

        if improved:
            counter = 0
            torch.save(model.state_dict(),
                       os.path.join(experiment_dir, "best_model.pth"))
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement in loss or acc for {patience} epochs)")
                break

    print("\n[*] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(
        os.path.join(experiment_dir, "best_model.pth")))
    test_acc, test_f1 = test_fn(model, test_loader, criterion, device, log_file)

    final_dir = f"{experiment_dir}_Acc_{test_acc:.3f}_F1_{test_f1:.3f}"
    try:
        os.rename(experiment_dir, final_dir)
        print(f"\n[+] Done. Dir: {final_dir}")
    except Exception as e:
        print(f"\n[!] Rename failed: {e}. Results in: {experiment_dir}")


if __name__ == '__main__':
    main()
