"""
train_dreaMS_harmonictsfm_2.py
DreaMSHT2 = a dreaMS branch that faithfully reproduces the original dreaMS
model + delta-Transformer encoding + searchable fusion -> 9-class adduct
classification.

The model is DreaMSAdductHarmonicTsfmClassifierV2 (see
model_dreaMS_harmonictsfm_2.py): the dreaMS spectrum encoder matches the
pretraining configuration of ssl_model.ckpt. The default is "fully consistent";
the DREAMS2_* switches in config control it, and the CLI flags below override
them for ablation:
  --heads 8|16      attention heads (8 = pretraining configuration; 16 is also
                    allowed to reproduce the older behaviour)
  --gd 0|1          whether to add the graphormer pairwise m/z difference
                    attention bias back
  --max_peaks N     None = keep config.MAX_PEAKS; 60 = pretraining value
  --dropout P       None = keep the backbone default 0.0; 0.1 = pretraining value

The dataset is selected with --data (default spectraverse):
  --data spectraverse  data/splits/         all nine classes, the main dataset
Self-check / regression: verify_dreaMS_fidelity.py
"""
import os, shutil, argparse, random, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              top_k_accuracy_score)
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from collections import Counter

import config
import model_dreaMS_harmonictsfm_2 as M2
from model_dreaMS_harmonictsfm_2 import DreaMSAdductHarmonicTsfmClassifierV2
from dataloader import get_dataloader
from train_utils import (
    AverageMeter, FocalLoss, ION_CE, _hierarchical,
    train_one_epoch_live, validate_live, test_evaluation_live,
)


def make_next_record_dir(base_dir, prefix):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    indices = []
    for d in existing:
        try: indices.append(int(d[len(prefix):].split("_")[0]))
        except ValueError: continue
    full_path = os.path.join(base_dir, f"{prefix}{max(indices, default=-1) + 1:03d}")
    os.makedirs(full_path, exist_ok=True)
    return full_path


def backup_project_files(dest_dir):
    files = ["config.py", "model_dreaMS.py",
             "model_dreaMS_harmonictsfm.py", "model_dreaMS_harmonictsfm_2.py",
             "train_utils.py", "train_dreaMS_harmonictsfm_2.py",
             "dataloader.py", "model.py", "fusion.py"]
    backup_path = os.path.join(dest_dir, "codes")
    os.makedirs(backup_path, exist_ok=True)
    for f in files:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(backup_path, os.path.basename(f)))
    print(f"[*] Code backed up to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description='DreaMS + δ-Transformer + fusion')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=['concat', 'hadamard', 'gated', 'ffn',
                                 'gated_cat', 'bilinear', 'film', 'coattn',
                                 'tri_cat', 'tri_attn', 'tri_xattn'])
    parser.add_argument('--ce_fusion', type=str, default=None,
                        choices=['gate', 'film', 'cat', 'none',
                                 'gate_delta', 'film_delta'])
    parser.add_argument('--fraction', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--prefix', type=str, default="MS2ADDUCT_DreaMSHT2_")
    parser.add_argument('--init_from', type=str, default=None,
                        help='warm start from an existing best_model.pth (strict load, overrides the '
                             'dreaMS pretrained backbone); used for single-variable '
                             'experiments that change one input relative to a given ckpt')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--bases', type=str, default=None,
                        choices=list(config.HARMONIC_BASE_SETS.keys()),
                        help='defaults to config.HARMONIC_BASES (config is the single source of truth)')
    parser.add_argument('--seed', type=int, default=1,
                        help='training seed, default 1 (fixed for reproducibility); pass 1/2/3 '
                             'explicitly for multi-seed runs. Note: the data split always uses '
                             'seed 42 (data/split_by_smiles.py), which is independent of this seed')
    parser.add_argument('--select', type=str, default='loss_acc',
                        choices=['loss_acc', 'f1'],
                        help='checkpoint selection / early-stopping criterion: loss_acc = the original winning criterion, f1 = val macro F1')
    parser.add_argument('--data', type=str, default='spectraverse',
                        choices=list(config.DATA_SETS.keys()),
                        help='dataset combination (see config.DATA_SETS): spectraverse = the main dataset')
    parser.add_argument('--heads', type=int, default=None, choices=[8, 16],
                        help='attention heads; 8 = ssl_model.ckpt pretraining configuration (default), 16 = behaviour of the older port')
    parser.add_argument('--gd', type=int, default=None, choices=[0, 1],
                        help='1 = add the graphormer pairwise m/z difference attention bias back (default)')
    parser.add_argument('--max_peaks', type=int, default=None,
                        help='override config.MAX_PEAKS; defaults to config (100), 60 = pretraining value')
    parser.add_argument('--dropout', type=float, default=None,
                        help='override the backbone dropout; defaults to the backbone default (0.0), 0.1 = pretraining value')
    parser.add_argument('--hsfm_dim', type=int, default=None,
                        help='delta transformer token dimension; must be divisible by --hsfm_heads')
    parser.add_argument('--hsfm_layers', type=int, default=None,
                        help='number of delta transformer layers')
    parser.add_argument('--hsfm_heads', type=int, default=None,
                        help='number of delta transformer heads; must divide --hsfm_dim')
    parser.add_argument('--ce_bins', type=int, default=None,
                        help='number of CE bins (CEAdapter num_ce_bins)')
    args = parser.parse_args()

    if args.fusion is not None:
        config.FUSION = args.fusion
    if args.ce_fusion is not None:
        config.CE_FUSION = args.ce_fusion
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

    config.TRAIN_PKL, config.VAL_PKL, config.TEST_PKL = config.DATA_SETS[args.data]

    if args.bases is not None:
        config.HARMONIC_BASES = args.bases
        config.HARMONIC_BASE_FREQUENCIES = config.HARMONIC_BASE_SETS[args.bases]
        config.DELTA_ENCODER = "bases"

    if args.heads is not None:
        config.DREAMS2_N_HEADS = args.heads
    if args.gd is not None:
        config.DREAMS2_GRAPHORMER = bool(args.gd)
    if args.max_peaks is not None:
        config.DREAMS2_MAX_PEAKS = args.max_peaks
        config.MAX_PEAKS = args.max_peaks
    elif getattr(config, 'DREAMS2_MAX_PEAKS', None) is not None:
        config.MAX_PEAKS = config.DREAMS2_MAX_PEAKS
    if args.dropout is not None:
        config.DREAMS2_DROPOUT = args.dropout

    if args.hsfm_dim is not None:
        config.HSFM_DIM = args.hsfm_dim
    if args.hsfm_layers is not None:
        config.HSFM_LAYERS = args.hsfm_layers
    if args.hsfm_heads is not None:
        config.HSFM_HEADS = args.hsfm_heads
    if args.ce_bins is not None:
        config.CE_NUM_BINS = args.ce_bins
    assert config.HSFM_DIM % config.HSFM_HEADS == 0, \
        f"HSFM_DIM({config.HSFM_DIM}) must be divisible by HSFM_HEADS({config.HSFM_HEADS})"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = config.DEVICE
    print(f"Device: {device} | data={args.data} | FUSION={config.FUSION} | "
          f"fraction={config.DATA_FRACTION} | "
          f"batch={config.BATCH_SIZE} | lr={config.LR} | epochs={config.EPOCHS} | "
          f"warmup={config.WARMUP_EPOCHS} | USE_FOCAL={getattr(config, 'USE_FOCAL', False)} | "
          f"POLARITY_AS_INPUT={config.POLARITY_AS_INPUT} | "
          f"POLARITY_MASK_OUTPUT={config.POLARITY_MASK_OUTPUT}")

    experiment_dir = make_next_record_dir("ckpt", args.prefix)
    print(f"[*] Experiment dir: {experiment_dir}")
    backup_project_files(experiment_dir)
    log_file = os.path.join(experiment_dir, "trainlog.txt")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"data={args.data} | train={config.TRAIN_PKL} | val={config.VAL_PKL} | "
                f"test={config.TEST_PKL}\n")
        f.write(f"bases={config.HARMONIC_BASES} | n_bases={len(config.HARMONIC_BASE_FREQUENCIES)} | "
                f"n_harmonics={config.N_HARMONICS_PER_BASE} | seed={args.seed} | "
                f"select={args.select} | patience={config.EARLY_STOP_PATIENCE}\n")
        f.write(f"fusion={config.FUSION} | ce={config.CE_FUSION} | "
                f"ce_token={getattr(config, 'CE_TOKEN_ENABLED', True)} | "
                f"hsfm_dim={config.HSFM_DIM} | hsfm_layers={config.HSFM_LAYERS} | "
                f"hsfm_heads={config.HSFM_HEADS} | ce_bins={config.CE_NUM_BINS} | "
                f"batch={config.BATCH_SIZE} | lr={config.LR} | data_fraction={config.DATA_FRACTION}\n")
        f.write(f"DREAMS2: mode={config.DREAMS_MODE} | n_heads={config.DREAMS2_N_HEADS} | graphormer={config.DREAMS2_GRAPHORMER} | "
                f"fourier_strategy={config.DREAMS2_FOURIER_STRAT} | "
                f"max_peaks={config.MAX_PEAKS} (DREAMS2_MAX_PEAKS={config.DREAMS2_MAX_PEAKS}) | "
                f"dropout={config.DREAMS2_DROPOUT if config.DREAMS2_DROPOUT is not None else 0.0} | "
                f"gd_exact={M2.GD_EXACT}\n")
        f.write(f"polarity_input={config.POLARITY_AS_INPUT} | "
                f"polarity_mask={config.POLARITY_MASK_OUTPUT} | "
                f"polarity_mask_start={config.POLARITY_MASK_START_EPOCH} | "
                f"polarity_emb_dim={config.POLARITY_EMB_DIM} | "
                f"ion_aux={config.ION_AUX_ENABLED} | "
                f"pairdiff_enabled={config.PAIRDIFF_ENABLED} | "
                f"pairdiff_base_set={config.PAIRDIFF_BASE_SET} | "
                f"pairdiff_harmonics={config.PAIRDIFF_HARMONICS} | "
                f"pairdiff_min_delta={config.PAIRDIFF_MIN_DELTA} | "
                f"pairdiff_sham={config.PAIRDIFF_SHAM}\n")
        f.write(f"delta_encoder={config.DELTA_ENCODER} | "
                f"delta_fourier_strategy={config.FOURIER_STRATEGY} | "
                f"delta_fourier_num_freqs={config.FOURIER_NUM_FREQS} | "
                f"delta_token_mode={getattr(config, 'DELTA_TOKEN_MODE', 'peak')} | "
                f"init_from={args.init_from}\n")
        f.write("HARMONIC_BASE_FREQUENCIES = [\n")
        for i in range(0, len(config.HARMONIC_BASE_FREQUENCIES), 6):
            row = ", ".join(f"{v:.6f}" for v in config.HARMONIC_BASE_FREQUENCIES[i:i + 6])
            f.write(f"    {row},\n")
        f.write("]\n" + "=" * 70 + "\n")

    mode = config.DREAMS_MODE
    if mode == "cache":
        raise NotImplementedError("the delta-Transformer needs raw peaks; use DREAMS_MODE=finetune/scratch")
    else:
        train_loader = get_dataloader(config.TRAIN_PKL, config.BATCH_SIZE,
                                      augment=True, shuffle=True)
        val_loader = get_dataloader(config.VAL_PKL, config.BATCH_SIZE,
                                    augment=False, shuffle=False)
        test_loader = get_dataloader(config.TEST_PKL, config.BATCH_SIZE,
                                     augment=False, shuffle=False)
        freeze = (mode != 'finetune')
        ckpt = os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt")
        if mode == 'scratch':
            ckpt, freeze = None, False
        print(f"[mode] {mode} | ckpt={ckpt if ckpt else 'None (random init)'} | "
              f"freeze_backbone={freeze}")
        model = DreaMSAdductHarmonicTsfmClassifierV2(
            ckpt_path=ckpt, freeze_backbone=freeze).to(device)
        if args.init_from:
            init_sd = torch.load(args.init_from, map_location=device)
            model.load_state_dict(init_sd)
            print(f"[init] warm start from {args.init_from} (strict)")
        train_fn = train_one_epoch_live
        val_fn = validate_live
        test_fn = test_evaluation_live
        scaler = GradScaler()

    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_hm = sum(p.numel() for p in model.harmonic_enc.parameters())
    print(f"Params: {n_all/1e6:.2f}M (trainable {n_tr/1e6:.2f}M) | δ-tsfm {n_hm/1e3:.0f}K")

    use_focal = getattr(config, 'USE_FOCAL', False)
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
        print(f"[Loss] FocalLoss alpha={alpha.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[Loss] CrossEntropyLoss")

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (
        float(ep + 1) / config.WARMUP_EPOCHS if ep < config.WARMUP_EPOCHS
        else config.LR_GAMMA ** (ep - config.WARMUP_EPOCHS)))

    best_val_loss, best_val_acc = float('inf'), 0.0
    best_val_f1 = -1.0
    counter = 0
    for epoch in range(1, config.EPOCHS + 1):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion,
                                         device, scaler=scaler)
        val_loss, val_acc, val_f1, _, _ = val_fn(model, val_loader, criterion, device)
        scheduler.step()
        msg = (f"Epoch {epoch:3d}/{config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if args.select == 'f1':
            improved = val_f1 > best_val_f1
        else:
            improved = val_loss < best_val_loss or val_acc > best_val_acc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if improved:
            counter = 0
            torch.save(model.state_dict(), os.path.join(experiment_dir, "best_model.pth"))
            print(f"  -> Saved best (val_f1={best_val_f1:.4f} val_loss={best_val_loss:.4f} "
                  f"val_acc={best_val_acc:.4f})")
        else:
            counter += 1
            if counter >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_f1={best_val_f1:.4f} "
                      f"@ epoch {epoch - config.EARLY_STOP_PATIENCE})")
                break

    print("\n[*] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(experiment_dir, "best_model.pth")))
    test_acc, test_f1 = test_fn(model, test_loader, criterion, device, log_file)
    final_dir = f"{experiment_dir}_Acc_{test_acc:.3f}_F1_{test_f1:.3f}"
    try:
        os.rename(experiment_dir, final_dir)
        print(f"\n[+] Done. Dir: {final_dir}")
    except Exception as e:
        print(f"\n[!] Rename failed: {e}. Results in: {experiment_dir}")
    return {'fusion': config.FUSION, 'best_val_acc': best_val_acc,
            'test_acc': test_acc, 'test_f1': test_f1}


if __name__ == '__main__':
    main()
