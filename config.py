import torch

# === 数据路径 ===
TRAIN_PKL = "data/splits/train.pkl"
VAL_PKL = "data/splits/val.pkl"
TEST_PKL = "data/splits/test.pkl"
DATA_FRACTION = 1.0            # 快速验证: 1.0=全量 0.2=20% (供融合选优短训用)

# === 数据集选择 (train_dreaMS_harmonictsfm_2.py --data) ===
# 每项为 (train, val, test) 三元组; --data 选定后写入上面的 TRAIN_PKL/VAL_PKL/TEST_PKL。
#   spectraverse : 9 类齐全, 主数据集
DATA_SETS = {
    "spectraverse": ("data/splits/train.pkl",
                     "data/splits/val.pkl",
                     "data/splits/test.pkl"),
}

# === MS 编码器超参数 (CSU-MS2) ===
MAX_PEAKS = 100
MS_PE_DIM = 256
MS_TRANSFORMER_LAYERS = 12
MS_TRANSFORMER_HEADS = 8
MS_TRANSFORMER_FFN_DIM = 512
MS_DROPOUT = 0.1

# === 分类头 ===
NUM_ADDUCT_CLASSES = 9
DROPOUT_RATE = 0.4
HIERARCHICAL_HEADS = True          # 分层分类：先判正/负离子模式，再判具体加合物
ION_LOSS_WEIGHT = 0.3              # 离子模式辅助 loss 权重
# 离子模式辅助头: 训练一个 Linear(256→2) 去**预测**正/负离子模式。
# 既然模式在采集时已由操作者选定、谱图元数据 (ion_mode 字段) 直接给出,
# 就没有任何理由让模型再猜一遍。False = 完全不构造 ion_mode_head, 也就不加该项辅助 loss
# (训练循环的 _hierarchical() 会自动跳过)。
# 注意: 只去掉"预测极性"的辅助头; positive_head / negative_head 这两个**同极性内**
# 的分类头保留 (它们不是极性预测器)。
# ★ 2026-11-04 定案: 最终架构去掉离子模式辅助头 (ION_AUX_ENABLED=False) 为当前最优。
#   Fourier δ + tri_xattn, 盲协议 3 seeds (spectraverse test):
#     free   Acc 0.8572 / Macro F1 0.7759   (带离子头 0.8583 / 0.7717)
#     masked Acc 0.8844 / Macro F1 0.7909   (带离子头 0.8833 / 0.7880)
#   配对 ΔF1 +0.0028 (t=0.76, n.s.), 方向 2/3 —— 点估计略优, 按点估计定为最优。
#   组件/融合排序不变 (见 ablation/results/noion_all_masked.json, anchor=tx_no_ion_aux)。
# 注意: 本快照 (best_model/) 显式设为 False; 仓库根目录 config.py 仍为 True,
#   因为历史消融臂 (tri_xattn / control 等) 依赖该默认值才能复现旧 ckpt。
#   在根目录重跑任何旧臂时, 不要以 best_model/ 的取值为准。
ION_AUX_ENABLED = False

# === CE (碰撞能量) 门控 ===
CE_NUM_BINS = 64
CE_EMBED_DIM = 32
CE_GATE_TYPE = "sigmoid"
CE_FUSION = "cat"      # CE 与谱图/δ 的融合: gate(乘性门控 d) / film(FiLM d) / cat(显式拼接) / none / gate_delta(只门控δ) / film_delta
                       # 2026-10-20: 默认改为 cat —— 当前最优 FUSION="tri_xattn" 需要 CE 作为
                       # 第三个 token; 两者是耦合的, 改 FUSION 时需同步改这里。
CE_AUX_DIM = 64        # cat 模式下 CE 描述向量维度 (三方融合时自动 = MS_PE_DIM)
# ── CE token 消融开关 (2026-10-21) ──
# False: 三方融合里**不提供** CE token/描述向量, 但保留 CEAdapter 模块本身。
#   对 tri_xattn 而言, KV 从 [δ, CE] 变成 [δ] (2-token → 1-token)。
#   这是"去掉 CE"的因果消融; 不能用 CE_FUSION="none" 代替 —— tri_xattn 在
#   aux 为 None 时会直接报错 (该检查保留, 用于防误用), 由本开关显式放行。
CE_TOKEN_ENABLED = True

# === 数据增强超参数 ===
MS_AUG_REMOVE_THRESHOLD = 5.0
MS_AUG_REMOVE_RATIO = 0.2
MS_AUG_JITTER_RANGE = 0.4
MS_AUG_MAX_NOISE_PEAKS = 10

# === 训练超参数 ===
BATCH_SIZE = 26
EPOCHS = 100
LR = 1e-5
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 5
LR_GAMMA = 0.98
EARLY_STOP_PATIENCE = 5

# === Focal Loss (train_utils.py) ===
USE_FOCAL = True              # 用 class-balanced focal loss 代替 CE (救稀有类)
FOCAL_GAMMA = 2.0

# === 硬件 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DreaMS 编码器 ===
DREAMS_MODE = "finetune"        # cache / finetune / scratch
DREAMS_CKPT_DIR = "dreaMS"
DREAMS_EMB_DIM = 1024
DREAMS_CACHE_DIR = "data/dreams_cache"

# ════════════════════════════════════════════════════════════════════
# dreaMS 原型一致性开关 (只作用于 model_dreaMS_harmonictsfm_2.py / DreaMSHT2)
#
# 背景: model_dreaMS.py 的移植版有两处与 dreaMS 原型不一致, 且都不报错:
#   (1) n_heads=16, 而 ssl_model.ckpt 是按 n_heads=8 预训练的。qkv 权重形状
#       (4096,1024) 与头数无关, load_state_dict(strict=False) 静默通过 ——
#       head_dim 变成 64 而非 128, scale 变成 1/8 而非 1/sqrt(128)。
#   (2) 丢掉了 graphormer 成对 m/z 差 attention bias (预训练
#       args.graphormer_mz_diffs=True)。注意它在预训练实际用的配置
#       (graphormer_parametrized=False) 下退化为 per-key 标量偏置 s_i - s_j,
#       query 侧的 s_i 在 softmax 中精确抵消, 因此真实作用是给每个 key 加
#       -s_j 先验 —— 实测主要效果是把注意力从 padding 汇点搬回真实峰
#       (真实 token 分给 padding key 的质量 20.0% → 10.7%)。
# 复现脚本: verify_dreaMS_fidelity.py (含与 ckpt 记录值的逐项对账)
#
# ── 六臂对照的实测结论 (2026-09-28, 各 3 seeds 全量, 见 analyze_fidelity_arms.py) ──
#   gd 的作用随 padding 比例**变号**, 两个独立指标给出同一交互项:
#     60 峰(预训练比例)下开 gd 有益 : Acc +0.0140 ± 0.0050 (3/3 同向)
#     100峰(本工程比例)下开 gd 有害 : Acc −0.0115 ± 0.0062 (3/3 同向)
#     交互项 Acc +0.0255 ± 0.0080 (t≈3.2)、Top-3 +0.0065 ± 0.0020 (t≈3.2)
#   → gd 的抑制强度是按预训练的 60 峰 padding 校准的, 拿到 100 峰上就过度抑制了
#     那个共享汇点。**gd 与 MAX_PEAKS 必须配套使用**。
#   当前取 B(= 8头/无gd/100峰): 三项指标全部排前二, 单跑 Acc 0.8544 为全项目最高。
#   若把 MAX_PEAKS 改成 60, 则应同时把 DREAMS2_GRAPHORMER 开回 True。
#   唯一应避免的组合是 100峰 + 开gd(= 六臂里唯一有可复现缺陷的 C, Top-3 显著垫底)。
# ════════════════════════════════════════════════════════════════════
DREAMS2_N_HEADS = 8             # 8 = ssl_model.ckpt 的预训练配置 (原移植版误用 16)
DREAMS2_GRAPHORMER = False      # 100峰下有害 (见上方实测结论); 与 MAX_PEAKS=60 配套时才应开
DREAMS2_FOURIER_STRAT = "lin_float_int"   # 与 ckpt 一致; 只影响 scratch 模式的初始化
                                          # (权重加载时频率值会被 ckpt 覆盖, 故对 finetune 无影响)
# --- 以下两项是训练/数据侧选择, 不是结构错配, 默认关 (None = 沿用现状) ---
DREAMS2_MAX_PEAKS = None        # None = 沿用 config.MAX_PEAKS(100); 60 = 预训练值(须同时开 gd)
DREAMS2_DROPOUT = None          # None = 沿用 backbone 默认 0.0; 0.1 = 预训练值

# === 中性丢失 Harmonic 编码 (model_dreaMS_harmonictsfm.py) ===
HARMONIC_EMB_DIM = MS_PE_DIM      # harmonic 投影输出维度 (与谱图特征对齐, concat 后 = 2*MS_PE_DIM)
N_HARMONICS_PER_BASE = 3          # 每个基频的谐波阶数
HARMONIC_CE_GATE = False          # 是否把 CE 门控同样作用到 harmonic 特征上 (实测 False 更优)

# === δ 分支 Transformer 编码 + 融合 (model_dreaMS_harmonictsfm.py) ===
HSFM_DIM = 256                    # δ transformer token 维度 (= 输出维度)
HSFM_LAYERS = 2                   # δ transformer 层数
HSFM_HEADS = 4                    # 头数
HSFM_DROPOUT = 0.1
HSFM_FFN_MULT = 4
FUSION = "tri_xattn"              # δ 分支与 DreaMS 特征融合。
                                  # 2026-10-20 用户定案: tri_xattn 为当前最优模型
                                  # (Fourier δ + cross-attention, 推理期极性屏蔽
                                  #  3 seeds test Acc 0.8833 / Macro F1 0.7880;
                                  #  与 bilinear 的配对差 +0.0011 F1 在噪声内, 按 F1 点估计选它)。
                                  # 依赖 CE_FUSION="cat" (CE 作为第三 token)。
                                  # 历史臂/旧 ckpt 的复现由 run_arms.py / load_ckpt.py 显式回退。
                                  # 可选: concat/hadamard/gated/ffn/gated_cat/bilinear/film/coattn/tri_cat/tri_attn/tri_xattn
# ════════════════════════════════════════════════════════════════════
# 中性丢失基频表 (Da, 精确单同位素质量)
#   训练脚本用 --bases {orig,adduct,chem,both} 选择, 见 train_dreaMS_harmonictsfm_2.py
#   ORIG   = 35 个化学中性丢失 (原表, 与 MSEcoding 一致)
#   ADDUCT = 26 个加合物位移差值 (9 类加合物位移的全两两 |Δ|≥1 Da, 去掉与 ORIG/CHEM 退化者)
#   CHEM   = 36 个数据驱动补充: 训练集池化 δ 直方图高频峰 ∩ 分子式穷举 <15 ppm 匹配 ∩ 不在 ORIG
#   退化判据 0.005 Da; 被排除的有 NH4−H≡NH3, HCOOH−H≡CO2, HCOOH−H−[M-H]-≡HCOOH,
#   CH3COOH−H−[M-H]-≡CH3COOH, Cl−H≡HCl, HCOOH−H−CH3COOH−H≡CH2, [M+H]+−[M-H]-≡H2
# ════════════════════════════════════════════════════════════════════
# --- ORIG: 化学含义明确的中性丢失 (原表, 与 MSEcoding 实验一致) ---
HARMONIC_BASES_ORIG = [
    14.015650, 15.023475, 15.994915, 17.002740, 17.026549, 18.010565,
    20.006228, 26.015650, 27.010899, 27.994915, 28.006148, 28.031300,
    29.002740, 30.010565, 31.018390, 31.989830, 33.987721, 35.976678,
    42.010565, 43.005814, 43.989830, 44.013640, 45.992904, 46.005480,
    46.041865, 56.026215, 56.062600, 59.013305, 60.021130, 60.057515,
    79.956816, 79.966332, 97.976896, 162.052824, 176.032089,
]

# --- ADDUCT: 加合物位移差值 (覆盖不同加合离子类型) ---
HARMONIC_BASES_ADDUCT = [
    1.007280,   # [M+H]+  - [M]+
    3.993760,   # [M+K]+  - [M+Cl]-
    4.955390,   # [M+Na]+ - [M+NH4]+
    6.034500,   # [M+HCOOH-H]- - [M+K]+
    10.028250,  # [M+HCOOH-H]- - [M+Cl]-
    11.980180,  # [M+Na]+ - [M+Cl]-
    15.973940,  # [M+Na]+ - [M+K]+
    16.935580,  # [M+NH4]+ - [M+Cl]-
    18.033820,  # [M+NH4]+ - [M]+
    19.041100,  # [M-H]-  - [M+NH4]+
    20.050150,  # [M+K]+  - [M+CH3COOH-H]-
    20.929330,  # [M+NH4]+ - [M+K]+
    21.981940,  # [M+H]+  - [M+Na]+     ★ [M+Na]+ 实测 δ 富集 x12.6
    22.008440,  # [M+Na]+ - [M+HCOOH-H]-
    22.989220,  # [M+Na]+ - [M]+
    23.996490,  # [M-H]-  - [M+Na]+
    24.043900,  # [M+CH3COOH-H]- - [M+Cl]-
    26.963830,  # [M+NH4]+ - [M+HCOOH-H]-
    33.962120,  # [M+H]+  - [M+Cl]-
    34.969400,  # [M+Cl]- - [M]+
    37.955880,  # [M+H]+  - [M+K]+     ★ [M+K]+ 实测 δ 富集 x3.6
    38.963160,  # [M+K]+  - [M]+
    39.970430,  # [M-H]-  - [M+K]+
    40.979480,  # [M+NH4]+ - [M+CH3COOH-H]-
    44.997650,  # [M+HCOOH-H]- - [M]+
    58.006030,  # [M+H]+  - [M+CH3COOH-H]-
]

# --- CHEM: 数据驱动补充的高频中性丢失 (训练集 δ 直方图 top 峰, 全部有分子式归属) ---
HARMONIC_BASES_CHEM = [
    2.015650,   # H2
    16.031300,  # CH4
    30.046950,  # C2H6
    32.026215,  # CH3OH
    36.021130,  # 2xH2O
    43.018390,  # C2H3O
    45.021464,  # CH3NO
    58.041865,  # C3H6O
    59.073499,  # C3H9N
    61.028954,  # C2H5O2
    62.036780,  # C2H6O2
    63.961901,  # SO2
    72.021130,  # C3H4O2
    73.028954,  # C3H5O2
    74.036779,  # C3H6O2    ★ 全池第 5 大中性丢失 (丙酸)
    76.052430,  # C3H8O2
    78.031690,  # C2H6O3
    79.926160,  # HBr
    86.036779,  # C4H6O2
    87.979659,  # C2O4
    88.016045,  # C3H4O3
    90.031695,  # C3H6O3
    92.047345,  # C3H8O3
    100.052430, # C5H8O2
    102.031695, # C4H6O3
    104.047345, # C4H8O3
    106.041865, # C7H6O
    116.047345, # C5H8O3
    118.041865, # C8H6O
    120.042260, # C4H8O4
    130.062995, # C6H10O3
    132.042260, # C5H8O4    ★ 戊糖
    146.057909, # C6H10O4   ★ 脱氧己糖
    148.052430, # C9H8O2
    164.047345, # C9H8O3
    180.063390, # C6H12O6   ★ 己糖
]

HARMONIC_BASE_SETS = {
    "orig":   HARMONIC_BASES_ORIG,                                          # 35
    "adduct": HARMONIC_BASES_ORIG + HARMONIC_BASES_ADDUCT,                  # 61
    "chem":   HARMONIC_BASES_ORIG + HARMONIC_BASES_CHEM,                    # 71
    "both":   HARMONIC_BASES_ORIG + HARMONIC_BASES_ADDUCT + HARMONIC_BASES_CHEM,  # 97
}
HARMONIC_BASES = "adduct"         # 当前基频表 (训练脚本 --bases 覆盖)
#   2026-09-20 定案: 默认从 "orig"(35) 改为 "adduct"(61)。
#   依据: 3 seeds 全量对照, Macro F1 0.7597±0.0045 vs 控制组 0.7496±0.0018, +0.0101,
#   3/3 seed 完全分离 (精确置换检验 p=1/20=0.050)。增益主要来自 [M]+ F1 0.280→0.354,
#   机制是「δ=完整加合物质量」标记了「峰丢了整个加合物」。
#   "chem"(36 个高频中性丢失) 实测无效 (-0.0006), 且会稀释 [M]+ 增益, 故不并入。
#   详见 memory: ms2adduct-bases-extension-experiment
HARMONIC_BASE_FREQUENCIES = HARMONIC_BASE_SETS[HARMONIC_BASES]

# ── 峰间差分中性丢失边偏置 (2026-10-18, 默认关) ──
PAIRDIFF_ENABLED = False      # True: δ-Transformer 加入所有峰对质量差的 attention 边偏置
PAIRDIFF_BASE_SET = "both"    # 边偏置使用的基频表 (HARMONIC_BASE_SETS 的键)
PAIRDIFF_HARMONICS = 1        # 每个基频的谐波阶数 (1 = 2*B 维边特征)
PAIRDIFF_MIN_DELTA = 1.0      # Da; Δ <= 此值的边偏置置 0 (同位素/重复峰)
PAIRDIFF_SHAM = False         # True: 用频率范围匹配的 sham 基频 (特异性对照)
PAIRDIFF_SHAM_SEED = 123

# ── 节点 δ 分支编码方式 (2026-10-18) ──
#   "bases"   : 手工基频表 (HARMONIC_BASE_FREQUENCIES), 旧 ckpt 的架构
#   "fourier" : dreaMS 风格固定傅里叶频率 (FourierFeatures), 不用手工表
# 2026-10-19 定案: 默认改为 "fourier" —— 域内 spectraverse 上 3 seeds 均优于 bases
# (盲协议 ΔF1 +0.0143; 叠加推理期极性屏蔽后 Acc 0.8817 / F1 0.7869, 项目最优)。
# 旧 ckpt (bases 架构) 的加载由 load_ckpt/eval_external 的 state_dict 键自动判据保证。
DELTA_ENCODER = "fourier"
# δ 分支的 token 定义 (2026-10-20):
#   "peak"      = 每个峰一个 token, δ_i = precursor_mz − mz_i  (当前最优模型)
#   "pair"      = 每个峰对一个 token, pd_ij = |δ_i − δ_j| = |mz_i − mz_j|
#   "peak_top5" = 全部峰 δ (前体中性丢失) + 强度前 K 个峰锚定的峰间差
#                 |δ_anchor − δ_j|, j 只取 m/z 小于锚点的有效峰 (单向向下)
#                 (K = DELTA_ANCHOR_K)
# 三者参数完全相同 (可 strict 互载); pair 模式逐样本前向, 最多 C(100,2)=4,950 个 token;
# peak_top5 单向配对, 最多 100 + K*99/2 量级的 token (K=5, k=100 时约 350)。
DELTA_TOKEN_MODE = "peak"
DELTA_ANCHOR_K = 5      # peak_top5 模式的锚点数 (按强度取前 K 个有效峰)
FOURIER_STRATEGY = "lin_float_int"   # dreaMS FourierFeatures 策略
FOURIER_X_MIN = 1e-4                 # Da; 与 dreaMS 预训练 max_tbxic_stdev 一致
FOURIER_X_MAX = 2000.0               # δ 的 clamp 上限
FOURIER_TRAINABLE = False            # 与 dreaMS 预训练一致: 固定频率
FOURIER_NUM_FREQS = 5998             # 仅 voronov_et_al 使用; lin_float_int 忽略

# === MLP / Transformer 通用模型 (layers.py 组件) ===
MS_GEN_EMBED_DIM = 256            # 嵌入维度
MS_GEN_ENCODER = "tsfm"            # MLP / tsfm (Transformer)
MS_GEN_NHEAD = 8                  # Transformer 头数
MS_GEN_TRANSFORMER_LAYERS = 6     # Transformer 层数
MS_GEN_DROPOUT = 0.4              # Encoder dropout
MS_GEN_POOLER = "ATT"             # ATT / AVG / MEAN / ESA
MS_GEN_APPLY_GATE = True          # 是否使用 CE Gate
MS_INTENSITY_LOG1P = False        # 是否对 intensity 做 log1p 变换

# === Adduct 类别映射 ===
ADDUCT_TO_IDX = {
    "[M+H]+": 0,
    "[M-H]-": 1,
    "[M+Na]+": 2,
    "[M+NH4]+": 3,
    "[M+HCOOH-H]-": 4,
    "[M+K]+": 5,
    "[M+CH3COOH-H]-": 6,
    "[M+Cl]-": 7,
    "[M]+": 8,
}
IDX_TO_ADDUCT = {v: k for k, v in ADDUCT_TO_IDX.items()}

# ════════════════════════════════════════════════════════════════════
# 离子模式 (正/负) 作为**已知输入**
#
# 动机: 正/负离子模式是采集时由操作者选定的, 谱图/元数据里直接给出 (ion_mode 字段
#   即 ESI+/ESI-), 不需要模型去猜。而当前分层头 (HIERARCHICAL_HEADS) 把
#   "先判正/负"当成一个预测任务: 九类 logits 里正负两侧自由竞争。
#
# 实证 (analyze_polarity.py, 控制臂 ckpt/MS2ADDUCT_DreaMSHT2_s*):
#   - 离子模式头与九类组装 logits 隐含极性的一致率 99.74%, 二者高度冗余;
#   - 按**预测**极性门控 = 零影响 (1630 个跨极性错误里 96.3% 极性头自己也判错);
#   - 按**真实**极性门控 = Acc 0.8544→0.8744 (+3.00%)、Macro F1 0.764→0.776。
#   即: 这个增益只有在极性"被给出"时才拿得到 —— 正是本开关要做的。
#
# 两个开关相互独立, 便于单变量归因 (默认全 False = 与既有控制臂逐行一致):
#   POLARITY_AS_INPUT    : 极性嵌入与 CE 嵌入拼接后一起送进 CE adapter
#                          (即"加一个输入, 和 CE 一块")
#   POLARITY_MASK_OUTPUT : 用给定极性把九类 logits 里反极性的 4/5 类置 -inf
#                          (既然已知, 就不该再让它参与 argmax 竞争)
# ════════════════════════════════════════════════════════════════════
# ★ 2026-10-18 定案 (12 臂 × 3 seeds):
#   采用方案 A —— 极性**不编码进网络**, 只在推理期作为输出约束。
#   test 0.8748 / Macro F1 0.7735 (盲协议基线 0.8467 / 0.7556)。
#   训练与原来的 control 臂逐位一致, 因此**不需要重训**: 已有的
#   ckpt/MS2ADDUCT_DreaMSHT2_s{1,2,3} 即该协议下的最优模型。
POLARITY_AS_INPUT = False
POLARITY_MASK_OUTPUT = True
POLARITY_EMB_DIM = 8            # 极性嵌入维度 (2 类, 8 维足够)
# 训练期屏蔽的预热轮数: 第 < N 轮不加屏蔽, 之后再加。0 = 从第 1 轮就屏蔽 (即 pol_full 的做法)。
# 为什么需要它 (2026-10-17 诊断):
#   一旦屏蔽, 反极性 logits 的 softmax 概率恒为 0 ⇒ **梯度恒为 0**, 极性嵌入唯一有效的
#   学习信号 ("把反极性压下去") 从第一步就不存在, 只剩经门控的弱调制。实测 pol_full
#   (全程屏蔽) 0.8633, 反而低于只加输入的 pol_input 0.8724。预热让九类自由竞争先把
#   特征塑造出来, 之后再切换到约束空间。
#   注意: 验证/评测始终按"部署形态"加屏蔽, 不受此开关影响。
POLARITY_MASK_START_EPOCH = 0
# 推荐部署形态: **训练期完全不加屏蔽** (此时训练与 control 臂逐行一致, 无需重训),
# 只在推理期用给定极性约束输出。与 POLARITY_AS_INPUT 互不影响。
# 用法: 与 POLARITY_MASK_OUTPUT=True 搭配 —— 训练照常 (九类自由竞争), 评估/部署时屏蔽。
# 依据: 推理期屏蔽对 Acc 单调不减 (真类必在给定极性内), 且 test 上
#   control + 事后屏蔽 0.8748 已等于该先验可带来的全部收益 (oracle)。
POLARITY_MASK_INFERENCE_ONLY = True   # 训练期不屏蔽 => 与 control 逐位一致

# 九类 → 离子模式 (0=正, 1=负); 与 ADDUCT_TO_IDX 同序, 与
# model_dreaMS.DreaMSAdductClassifier.GLOB_TO_ION 是同一张表 (dataloader 不 import 模型)
GLOB_TO_ION = [0, 1, 0, 0, 1, 0, 1, 1, 0]
POLARITY_OF_ADDUCT = {a: GLOB_TO_ION[i] for a, i in ADDUCT_TO_IDX.items()}
