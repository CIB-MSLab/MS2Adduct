# best_model — 当前最优模型的完整训练代码

本目录是仓库中**当前最优模型**（2026-11-04 定案）的完整、自包含训练代码快照。
所有 `.py` 文件都从仓库根目录复制而来，唯一改动是 `config.py` 里的 `ION_AUX_ENABLED=False`
（见第 1 节与第 7 节第 8 条），其余与根目录完全一致。

最优模型 = **DreaMSHT2 谱图编码器 + Fourier δ 分支 + tri_xattn 三路 cross-attention + CE token
+ 去掉离子模式辅助头 + 推理期极性屏蔽**。

---

## 1. 模型与结果

| 项目 | 内容 |
|---|---|
| 谱图编码器 | dreaMS 预训练骨干（8 头 / 无 graphormer bias / 100 峰 / finetune） |
| δ 分支 | 每个峰一个 token，δ = precursor_mz − mz，`DELTA_ENCODER="fourier"`（dreaMS 风格固定傅里叶频率）+ 2 层小 Transformer + [CLS] |
| 融合 | `FUSION="tri_xattn"`：dreaMS 特征作 query，[δ, CE] 作 key/value 的 cross-attention |
| CE | `CE_FUSION="cat"`，CE 描述向量作为第三个 token（`CE_TOKEN_ENABLED=True`） |
| 分类头 | 分层头（正/负离子各一个分类头）；**离子模式辅助头已移除**（`ION_AUX_ENABLED=False`） |
| 极性 | `POLARITY_AS_INPUT=False`，`POLARITY_MASK_OUTPUT=True`，`POLARITY_MASK_INFERENCE_ONLY=True`：极性不进网络，只在推理期屏蔽反极性类 |
| 空间选择 | Macro F1（`--select f1`），不是 `loss_acc` |

**3 seeds test 结果**（spectraverse，全量，bs=26，lr=1e-5，focal loss）：

| 协议 | s1 | s2 | s3 | 均值 |
|---|---|---|---|---|
| 盲协议（训练/评测都不用极性） | 0.855 / 0.773 | 0.858 / 0.773 | 0.859 / 0.782 | Acc 0.8572 / F1 0.7759 |
| 部署协议（推理期极性屏蔽，本目录默认） | 0.882 / 0.789 | 0.884 / 0.787 | 0.888 / 0.797 | **Acc 0.8844 / F1 0.7909** |

（表中为 Acc / Macro F1；对应权重 `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s{1,2,3}_*`。）

与上一版（保留离子辅助头）的配对差：masked ΔF1 **+0.0028**（t=0.76, n.s.）、free ΔF1 +0.0041（n.s.），
方向 2/3。**点估计略优、统计上不可区分**；按用户决定以点估计选为当前最优。

---

## 2. 文件职责与依赖关系

```
train_dreaMS_harmonictsfm_2.py      ← 训练入口（本目录唯一需要执行的脚本）
├── config.py                       ← 全部超参/开关（全局单例，唯一真源）
├── dataloader.py                   ← pkl → top-100 峰 + padding + 增强
├── model_dreaMS_harmonictsfm_2.py  ← DreaMSHT2：忠实版 dreaMS 编码器
│   ├── model_dreaMS_harmonictsfm.py ← δ-Transformer + FusionBlock + 分层头 + 极性屏蔽
│   │   ├── model_dreaMS.py          ← dreaMS 骨干、FourierFeatures、权重加载
│   │   ├── fusion.py                ← FusionBlock / CEAdapter（tri_xattn 在这里）
│   │   └── model.py                 ← EnergyGate
│   └── model_dreaMS_harmonic.py     ← 旧 δ-MLP 分类器（被 train_dreaMS_harmonic 引用）
└── train_dreaMS_harmonic.py         ← 复用训练/验证/测试循环、FocalLoss、ION_CE
```

- `train_dreaMS_harmonictsfm_2.py` 是入口；`train_dreaMS_harmonic.py` 只被 import 其工具函数。
- `model_dreaMS_harmonic.py` 仅因 `train_dreaMS_harmonic.py` 的模块级 import 而需要保留。
- 其余文件（`layers.py`、`model_dreaMS.py` 里的 cache 模式等）不在本模型的运行路径上。

---

## 3. 环境

开发与实验使用：

- Python 3.8.10
- torch 2.1.1+cu121（代码使用 `torch.cuda.amp.autocast/GradScaler` 旧接口）
- numpy 1.24.4、scikit-learn 1.3.0、tqdm 4.67.1
- 单卡 NVIDIA GPU（RTX 4090 实测可单卡跑 bs=26 / 100 峰，并可 3–4 进程并发）

```bash
/home/ssc/anaconda3/envs/torch/bin/python -V   # 本项目实际使用的解释器
```

依赖见 `requirements.txt`。

---

## 4. 数据与预训练权重

训练脚本使用**相对路径**，因此必须在**仓库根目录**下运行（或自行把 `config.py` 里的路径改成绝对路径）。

```
data/splits/train.pkl   392,513 条
data/splits/val.pkl
data/splits/test.pkl
```

每条记录是 dict，训练实际读取的字段：

| 字段 | 说明 |
|---|---|
| `mz`, `intensity` | 峰列表（原始强度，dataloader 会 top-100 + 归一化） |
| `adduct` | 九类加合物之一，见 `config.ADDUCT_TO_IDX` |
| `precursor_mz` | 前体 m/z，δ 分支的锚点 |
| `normalized_collision_energy` | CE（0–150 线性分 64 箱） |
| `smiles` | 仅用于按分子划分数据（本训练脚本不读） |
| `ion_mode` / `polarity`（可选） | 若存在则作为已知极性；不存在时按 adduct 推导 |

划分方式：`data/split_by_smiles.py`（按 SMILES、seed 42、8:1:1），与训练 seed 是两套独立随机源。

预训练权重：

```
dreaMS/ssl_model.ckpt    1.39 GB, dreaMS 自监督预训练 Lightning ckpt
```

路径由 `config.DREAMS_CKPT_DIR` 决定。`model_dreaMS.py` 内部用自定义 Unpickler 绕过 `msml` 依赖，
**不需要安装 dreaMS 官方包**。缺失该文件时脚本只会打印 WARNING 并以随机初始化继续（结果不可比）。

---

## 5. 训练（复现最优模型）

在仓库根目录执行：

```bash
bash best_model/run_train.sh              # 默认 seeds 1 2 3，顺序执行
SEEDS="1" bash best_model/run_train.sh    # 只跑一个 seed
```

等价的手动命令（单个 seed）：

```bash
cd /media/E/AI/MS2ADDUCT
/home/ssc/anaconda3/envs/torch/bin/python best_model/train_dreaMS_harmonictsfm_2.py \
    --seed 1 --select f1 \
    --fusion tri_xattn --ce_fusion cat \
    --prefix MS2ADDUCT_DreaMSHT2best_s1_
```

要点：

- `--select f1` **必须显式给出**。脚本默认是 `loss_acc`（冠军原判据），那会选出不同的 checkpoint。
- `--fusion tri_xattn` 与 `--ce_fusion cat` 是耦合的：tri_xattn 需要 CE token 作为第三路 KV。
- **不要传 `--bases`**。`config.DELTA_ENCODER="fourier"` 下手工基频表不参与计算；显式传 `--bases`
  会强制把 `DELTA_ENCODER` 切回 `"bases"`（旧架构）。
- **离子辅助头由本目录的 `config.py` 关闭**（`ION_AUX_ENABLED=False`）。手动运行时请确认
  trainlog 头部是 `ion_aux=False`；`run_train.sh` 会在启动前 assert 这一点。
- 其余全部超参都在 `config.py`，默认值即最优值，无需 CLI 覆盖。
- 训练产出：`ckpt/<prefix>NNN_Acc_x.xxx_F1_x.xxx/best_model.pth` 与 `trainlog.txt`
  （脚本会顺带把当时的代码备份到该目录的 `codes/`）。
- 训练结束会自动加载 best_model.pth 并在 test split 上评估，指标写入 trainlog。

---

## 6. 两种评测协议

`config.py` 默认是**部署协议**：

```python
POLARITY_AS_INPUT = False
POLARITY_MASK_OUTPUT = True
POLARITY_MASK_INFERENCE_ONLY = True   # 训练期不屏蔽 → 训练与盲协议逐位一致
```

- **部署协议（默认）**：训练不接触极性；评测/推理时用给定极性把反极性 4/5 类置 `-1e4`。
  报告的 0.8844 / 0.7909 就是这个协议。
- **盲协议**：训练与评测都不使用极性。做消融、和旧的 `ckpt/MS2ADDUCT_DreaMSHT2_s*` 控制臂
  对比时必须显式改回：

```python
POLARITY_AS_INPUT = False
POLARITY_MASK_OUTPUT = False
POLARITY_MASK_INFERENCE_ONLY = False
```

三种设置下**训练出的权重完全相同**（训练期屏蔽被跳过），差别只在评测输出。

---

## 7. 复现注意事项

1. **`config.py` 是全局单例**：训练脚本在 `main()` 里直接改写 `config.*`。不要在 import 之后、
   改配置之前构造模型；也不要在同一进程里连续跑两个不同配置的实验。
2. **数据划分 seed 固定 42**，与 `--seed`（训练随机源）无关；多 seed 实验只改 `--seed 1/2/3`。
3. **`DREAMS2_GRAPHORMER` 与 `MAX_PEAKS` 必须配套**：默认 100 峰 + 关 graphormer 是当前最优组合；
   改成 60 峰时应同时开 graphormer（`--max_peaks 60 --gd 1`）。100 峰 + 开 gd 是已知最差组合。
4. **峰对差分（`PAIRDIFF_ENABLED`）保持 False**：实测负结果，开启后 3/3 seed 变差。
5. **`DELTA_TOKEN_MODE` 保持 `"peak"`**：`pair` / `peak_top5` 是已证伪的消融分支。
6. **结果噪声**：Macro F1 的 seed 标准误约 0.0045，单 seed 差异小于 ~0.01 不可区分，结论需 3 seeds。
7. **本目录是快照**。若根目录代码有更新，用 `bash best_model/sync_from_root.sh` 重新同步：
   它会复制根目录文件，并**自动重新应用** `ION_AUX_ENABLED=False` 这一覆盖（同步后请看一眼它的输出）。
8. **本快照的 `config.py` 与仓库根目录的唯一差异是 `ION_AUX_ENABLED=False`**。根目录保持 `True`，
   因为历史消融臂（`tri_xattn`、`control` 等）依赖该默认值才能复现旧 ckpt；新的无离子头锚点
   `tx_no_ion_aux` 在 `ablation/run_arms.py` 里显式覆盖为 `False`。在根目录跑任何旧臂时，
   不要以本目录的取值为准。

---

## 8. 自检

不训练、只验证代码与权重能装配：

```bash
cd /media/E/AI/MS2ADDUCT
/home/ssc/anaconda3/envs/torch/bin/python best_model/model_dreaMS_harmonictsfm_2.py
```

预期输出（末尾）：

```
backbone: n_heads=8, head_dim=128, graphormer=False, fourier_strategy=lin_float_int
params: 101.49M (trainable 101.47M)
logits: (4, 9)  有限值=True
自检 PASSED
```
