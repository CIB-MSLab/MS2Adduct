"""
build_cib_dataset.py
把课题组自有的 .ms 谱图目录 (MSDIR) 解析并保存为单个 PKL。

数据来源: data/cib-ms/MSDIR/*.ms —— 5189 个文本谱图文件, Thermo QExactive,
          ESI 正离子模式, 1726 个分子 × 3 档碰撞能量 (20/30/40 eV)。

文件格式 (每个 .ms 一个谱图):
    HEADER: FTMS + c ESI d Full ms2 242.1431@hcd20.00 [50.0000-265.0000]
    INSTRUMENT: Thermo QExactive
    INCHIKEY: AAEVYOVXGOFMJO-UHFFFAOYSA-N
    SMILES: CC(C)NC1=NC(=NC(=N1)SC)NC(C)C
    FORMULA: C10H19N5S
    EMASS: 241.1361                 ← 中性分子单同位素质量
    PRECURSOR: 242.14339
    ION_MODE: ESI+
    ION_TYPE: [M+H]+
    COLLISION_ENERGY: 20.0 eV
    SCAN_NO: 4543
    CHROM_COND: ...
    CCS: 166.27
    RT: 15.088866 min
    NUM_PEAKS: 12
    Mass	Intensity	Rel. Intensity
    200.0968	10700110.0000	8.4514
    ...
文件名同时冗余编码了 (INCHIKEY, ION_TYPE, CE): <INCHIKEY>_<ION_TYPE>@<CE>eV.ms

输出的每条记录字段与 spectraverse / enveda 一致 (见 build_enveda_dataset.py 与
dataloader.AdductDataset 读取的键), 另加本数据集独有的元数据:
    smiles, adduct, precursor_mz, normalized_collision_energy, mz, intensity
    + inchikey, formula, emass, ion_mode, collision_energy_ev, ccs, rt, scan_no,
      precursor_ppm_error, source_file

两处需要说明的数据处理:

1) ION_TYPE 归一化 [M+]+ → [M]+
   18 个谱图 (6 个分子 × 3 档 CE) 的 ION_TYPE 写作 "[M+]+", 对应季铵/永久正离子,
   化学上就是 config.ADDUCT_TO_IDX 里的 [M]+ 类。不归一化会被 dataloader 直接过滤掉。
   例: GYFSYEVKFOOLFZ (SMILES 含 .[Cl-], 是盐酸盐) 的 EMASS 357.0768 为含氯中性质量,
   PRECURSOR 322.10738 = 357.0768 − 34.9694 (丢 Cl), 即 [M]+ —— 与 ION_TYPE 自洽。

2) 碰撞能量按绝对 eV 存储, 不做 NCE 换算
   spectraverse 的 normalized_collision_energy 是 NCE(0-100), 本数据集只有绝对 eV。
   fusion.CEEncoder._embed 把能量 clamp 到 [0,150] 后按 energy/150*CE_NUM_BINS 分箱
   (CE_NUM_BINS=64 → 箱宽 2.34), 20/30/40 eV 落在 3 个不同的箱子, 无需换算即可用。
   同时额外存一份 collision_energy_ev 保留原始单位, 避免字段名 "normalized" 造成误读。

强度归一化到 0-100 (除以谱内最大绝对强度), 与 build_enveda_dataset.py 一致。
不丢弃任何谱图: 无 SMILES / 无峰 / 非 9 类加合物的记录都保留, 由 dataloader 侧过滤
(与 enveda 的处理方式相同)。

用法:
    python data/cib-ms/build_cib_dataset.py                 # 全量 → cib-ms.pkl
    python data/cib-ms/build_cib_dataset.py --limit 200     # 冒烟测试

    # 与本项目训练集去重, 生成可用于外部验证的干净版本 (需 torch 环境, 依赖 rdkit):
    python data/cib-ms/build_cib_dataset.py \\
        --exclude-molecules-from data/splits/train.pkl \\
        --output-pkl data/cib-ms/cib-ms-dedup-train.pkl

去重 (--exclude-molecules-from) 的判定标准是**分子级**的: 只要某个分子在给定 pkl 里出现过,
它的全部谱图 (所有加合物 × 所有 CE) 都被剔除 —— 模型见过某分子的 [M+H]+ 谱, 就等于见过这个
化合物本身, 它的 [M+Na]+ 谱留在测试集里同样是泄漏。
身份判定用 RDKit 规范化的 InChIKey, **不能用 SMILES 字符串直接比**: 本数据集与 spectraverse
的 SMILES 规范形不同, 字符串比对只能找出 26 个重合分子, 而真实重合有 578 个 —— 差 20 倍,
靠字符串去重会得到一份看似干净、实则泄漏率极高的「验证集」。故缺 rdkit 时直接报错退出, 不静默退化。
"""
import argparse
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(HERE, "MSDIR")
DEFAULT_OUTPUT_PKL = os.path.join(HERE, "cib-ms.pkl")

# 文件名 ION_TYPE → 9 类目标命名。只有 [M+]+ 需要改写 (写法问题, 化学实体相同)。
ADDUCT_ALIAS = {
    "[M+]+": "[M]+",
}

# 9 类目标加合物 (与 config.ADDUCT_TO_IDX 一致)
CANONICAL_ADDUCTS = {
    "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+HCOOH-H]-",
    "[M+K]+", "[M+CH3COOH-H]-", "[M+Cl]-", "[M]+",
}

# 中性分子 → 离子 的质量位移 (Da, 单同位素), 仅用于前体质量自洽性检查
ADDUCT_SHIFT = {
    "[M+H]+": 1.0072765,
    "[M+Na]+": 22.9892180,
    "[M+K]+": 38.9631580,
    "[M+NH4]+": 18.0338255,
}

FILENAME_RE = re.compile(r'^(?P<inchikey>.+)_(?P<adduct>\[[^\]]*\]\+?[+-]?)@(?P<ce>[0-9.]+)eV\.ms$')

# 峰表起始行; 其后的行都是 "mz intensity rel_intensity"
PEAK_TABLE_HEADER = "Mass\tIntensity"


def parse_header_line(line):
    """'KEY: VALUE' → (KEY, VALUE); 无冒号返回 None。只切第一个冒号 —— SMILES/CHROM_COND 里含冒号。"""
    if ':' not in line:
        return None
    key, _, value = line.partition(':')
    return key.strip(), value.strip()


def parse_ms_file(path, name):
    """解析单个 .ms 文件。返回 (记录 dict, 告警列表); 无法解析时返回 (None, 跳过原因)。"""
    header = {}
    peaks = []
    in_peaks = False
    malformed_peak_lines = 0

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            if not line.strip():
                continue

            if not in_peaks:
                if line.startswith(PEAK_TABLE_HEADER):
                    in_peaks = True
                    continue
                kv = parse_header_line(line)
                if kv is not None:
                    header[kv[0]] = kv[1]
                continue

            parts = line.split()
            if len(parts) < 2:
                malformed_peak_lines += 1
                continue
            try:
                peaks.append((float(parts[0]), float(parts[1])))
            except ValueError:
                malformed_peak_lines += 1

    # --- 文件名与头部互相校验 (两处都冗余编码了 inchikey / adduct / ce) ---
    m = FILENAME_RE.match(name)
    if m is None:
        return None, 'filename_unparsable'

    fn_inchikey = m.group('inchikey')
    fn_adduct = ADDUCT_ALIAS.get(m.group('adduct'), m.group('adduct'))
    fn_ce = float(m.group('ce'))

    if not peaks:
        return None, 'no_peaks'

    smiles = header.get('SMILES', '').strip()
    if not smiles:
        return None, 'no_smiles'

    adduct_raw = header.get('ION_TYPE', fn_adduct).strip()
    adduct = ADDUCT_ALIAS.get(adduct_raw, adduct_raw)

    inchikey = header.get('INCHIKEY', fn_inchikey).strip()

    # CE: 头部 "20.0 eV" 优先, 退回文件名的 @20.0eV
    ce = None
    ce_match = re.search(r'([0-9.]+)', header.get('COLLISION_ENERGY', ''))
    if ce_match:
        ce = float(ce_match.group(1))
    else:
        ce = fn_ce

    max_int = max(p[1] for p in peaks)
    if max_int <= 0:
        return None, 'zero_intensity'

    rec = {
        'smiles': smiles,
        'adduct': adduct,
        'precursor_mz': float(header['PRECURSOR']) if header.get('PRECURSOR') else 0.0,
        # 与 dataloader / 其余数据集同名的字段。本数据集里是绝对 eV, 非 NCE, 见模块 docstring。
        'normalized_collision_energy': ce,
        'mz': [p[0] for p in peaks],
        'intensity': [p[1] / max_int * 100.0 for p in peaks],
        # --- 以下为本数据集独有元数据, dataloader 不读取 ---
        'inchikey': inchikey,
        'formula': header.get('FORMULA', ''),
        'emass': float(header['EMASS']) if header.get('EMASS') else None,
        'ion_mode': header.get('ION_MODE', ''),
        'collision_energy_ev': ce,
        'ccs': float(header['CCS']) if header.get('CCS', '').replace('.', '', 1).isdigit() else None,
        'rt': float(header['RT'].split()[0]) if header.get('RT') else None,
        'scan_no': int(header['SCAN_NO']) if header.get('SCAN_NO', '').isdigit() else None,
        'source_file': name,
    }
    if rec['ccs'] == 0.0:      # 部分文件 CCS 记为 0, 视为缺失
        rec['ccs'] = None

    # 前体质量自洽性: PRECURSOR ?= EMASS + 加合物位移, 存成 ppm 供外部验证时过滤。
    # 全库 93.9% 在 ±10 ppm 内; 6.0% 严重不符 (整体子不符, 见 report 的说明)。
    # [M]+ 的 EMASS 是含反离子的中性质量 (如盐酸盐), 不适用, 存 None。
    rec['precursor_ppm_error'] = None
    if rec['emass'] and adduct in ADDUCT_SHIFT:
        theory = rec['emass'] + ADDUCT_SHIFT[adduct]
        rec['precursor_ppm_error'] = (rec['precursor_mz'] - theory) / theory * 1e6

    flags = []
    if adduct != fn_adduct:
        flags.append('adduct_mismatch')
    if inchikey != fn_inchikey:
        flags.append('inchikey_mismatch')
    if abs(ce - fn_ce) > 1e-6:
        flags.append('ce_mismatch')
    if malformed_peak_lines:
        flags.append(f'malformed_peak_lines={malformed_peak_lines}')

    return rec, flags


def parse_dir(input_dir, limit=None):
    print(f"[1/3] 解析 .ms 目录: {input_dir}")
    t0 = time.time()

    names = sorted(n for n in os.listdir(input_dir) if n.endswith('.ms'))
    if limit:
        names = names[:limit]
    print(f"    发现 {len(names)} 个 .ms 文件")

    records = []
    skipped = Counter()
    flagged = defaultdict(list)

    for name in names:
        rec, flags = parse_ms_file(os.path.join(input_dir, name), name)
        if rec is None:
            skipped[flags] += 1
            continue
        for flag in flags:
            flagged[flag].append(name)
        records.append(rec)

    print(f"    → 提取 {len(records)} 个谱图, 耗时 {time.time() - t0:.1f}s")
    if skipped:
        print("    跳过: " + ", ".join(f"{k} {v} 条" for k, v in skipped.most_common()))
    if flagged:
        print("    文件名/头部不一致:")
        for flag, files in sorted(flagged.items()):
            print(f"      {flag}: {len(files)} 条, 例 {files[0]}")
    return records


def save_pkl(records, pkl_path):
    print(f"\n[2/3] 保存 PKL → {pkl_path}")
    t0 = time.time()
    with open(pkl_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    {len(records)} 条, {os.path.getsize(pkl_path) / 1024 / 1024:.1f} MB, "
          f"耗时 {time.time() - t0:.1f}s")


def report(records):
    n = len(records)
    unique_smiles = len(set(r['smiles'] for r in records))
    unique_inchikey = len(set(r['inchikey'] for r in records))
    peak_counts = sorted(len(r['mz']) for r in records)
    n_few = sum(1 for c in peak_counts if c < 10)

    adduct_counts = Counter(r['adduct'] for r in records)
    n_canonical = sum(v for k, v in adduct_counts.items() if k in CANONICAL_ADDUCTS)
    ce_counts = Counter(r['normalized_collision_energy'] for r in records)

    print("\n" + "=" * 64)
    print("[3/3] 统计:")
    print(f"  总谱图数:      {n}")
    print(f"  唯一 SMILES:   {unique_smiles}")
    print(f"  唯一 InChIKey: {unique_inchikey}")
    print(f"  峰数:          最少 {peak_counts[0]}, 中位 {peak_counts[n // 2]}, "
          f"最多 {peak_counts[-1]}, 平均 {sum(peak_counts) / n:.1f}")
    print(f"  峰数 < 10:     {n_few} ({n_few / n * 100:.1f}%)  "
          f"(MAX_PEAKS=100 时不足 100 峰的谱图会 padding)")
    print(f"  有 CCS:        {sum(1 for r in records if r['ccs'] is not None)} / {n}")
    print("=" * 64)

    print(f"\n9 类目标加合物覆盖: {n_canonical} / {n} ({n_canonical / n * 100:.1f}%)")
    print(f"\n{'加合物':<14}{'谱图数':>8}{'占比':>9}  {'9类内':>6}")
    print("-" * 44)
    for k, v in adduct_counts.most_common():
        print(f"{k:<14}{v:>8}{v / n * 100:>8.1f}%  {'  ✓' if k in CANONICAL_ADDUCTS else '  -'}")
    missing = sorted(CANONICAL_ADDUCTS - set(adduct_counts))
    print(f"本数据集缺失的加合物类 ({len(missing)}/9): {' '.join(missing) if missing else '无'}")
    print("  ↑ 缺类会让 Macro F1 虚高、confusion_matrix 越界 (见 memory: "
          "ms2adduct-data-switch-and-metric-labels), 单独用本数据集评测时须留意")

    print("\n碰撞能量分布 (绝对 eV, 存于 normalized_collision_energy):")
    for k, v in sorted(ce_counts.items()):
        print(f"  {k} eV: {v} 条")

    # 前体质量自洽性: PRECURSOR ?= EMASS + 加合物位移。只对「中性分子 + 一个离子」成立;
    # [M]+ 的 EMASS 是含反离子的中性质量 (如盐酸盐), 不适用, 故存 None 不检查。
    print("\n前体质量自洽性 (PRECURSOR vs EMASS + 加合物位移, 存于 precursor_ppm_error):")
    print(f"  {'加合物':<10}{'n':>6}{'中位 ppm':>12}{'>10ppm':>9}{'>100ppm':>9}")
    print("  " + "-" * 46)
    bucket = Counter()
    for adduct, _ in ADDUCT_SHIFT.items():
        errs = sorted(r['precursor_ppm_error'] for r in records
                      if r['adduct'] == adduct and r['precursor_ppm_error'] is not None)
        if not errs:
            continue
        n_10 = sum(1 for e in errs if abs(e) > 10.0)
        n_100 = sum(1 for e in errs if abs(e) > 100.0)
        print(f"  {adduct:<10}{len(errs):>6}{errs[len(errs) // 2]:>12.2f}{n_10:>9}{n_100:>9}")
        for e in errs:
            bucket['≤10 ppm' if abs(e) <= 10 else '10–100 ppm' if abs(e) <= 100
                   else '>100 ppm'] += 1
    total = sum(bucket.values())
    for k in ('≤10 ppm', '10–100 ppm', '>100 ppm'):
        print(f"  {k:<12}{bucket[k]:>6} ({bucket[k] / total * 100:5.1f}%)")
    print("  >100 ppm 的谱图里, 标注的 FORMULA/EMASS 与真正被碎裂的前体不是同一个物种")
    print("  (实测 102/104 个分子-加合物组合的 3 档 CE 偏移完全一致 → 属系统性标注错, 非随机噪声):")
    print("    例 AQQFWSYLUDMDBP: SMILES=C28H32O16 / EMASS=624.169, 但谱图是 m/z 401.159 的 MS2")
    print("  外部验证时可据此字段剔除 (precursor_ppm_error 已写入每条记录)。")
    print("=" * 64)


def self_check(records, input_dir):
    """抽样回溯原始 .ms 文件, 逐峰校验 mz / intensity 与源文件一致。"""
    print("\n[自检] 抽样回溯原始 .ms 文件 ...")
    random.seed(0)
    ok = True
    for rec in random.sample(records, min(5, len(records))):
        with open(os.path.join(input_dir, rec['source_file']),
                  'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
        i = next(idx for idx, l in enumerate(lines) if l.startswith(PEAK_TABLE_HEADER))
        raw_peaks = [tuple(map(float, l.split()[:2])) for l in lines[i + 1:] if l.strip()]
        max_int = max(p[1] for p in raw_peaks)
        same = (len(raw_peaks) == len(rec['mz'])
                and all(abs(a - b) < 1e-9 for a, b in zip((p[0] for p in raw_peaks), rec['mz']))
                and all(abs(a - b) < 1e-9 for a, b in zip((p[1] / max_int * 100.0
                                                           for p in raw_peaks), rec['intensity'])))
        ok = ok and same
        print(f"  {'✓' if same else '✗'} {rec['source_file']}  "
              f"{len(rec['mz'])} 峰, 最大强度归一化到 {max(rec['intensity']):.1f}")
    print(f"  自检{'通过' if ok else '失败'}")
    return ok


def mol_key(smiles):
    """SMILES → RDKit 规范化 InChIKey, 作为分子身份。无法解析返回 None。"""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def load_excluded_keys(paths):
    """读入若干 split pkl, 返回其中出现过的全部分子身份 (InChIKey) 集合。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    keys = set()
    for path in paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        n_mol = len(set(r['smiles'] for r in data if r.get('smiles')))
        before = len(keys)
        unparsed = 0
        for r in data:
            smi = r.get('smiles')
            if not smi:
                continue
            k = mol_key(smi)
            if k:
                keys.add(k)
            else:
                unparsed += 1
        print(f"    排除集 {os.path.basename(path)}: {len(data)} 谱图 / {n_mol} 唯一 SMILES "
              f"→ 新增 {len(keys) - before} 个唯一 InChIKey"
              + (f" (⚠ {unparsed} 条 SMILES 无法解析, 未计入)" if unparsed else ""))
    return keys


def dedup_against(records, exclude_paths):
    """剔除与 exclude_paths 中任一分子重合的记录 (分子级: 该分子所有谱图一并剔除)。"""
    print(f"\n[去重] 排除集: {', '.join(exclude_paths)}")
    excl = load_excluded_keys(exclude_paths)

    # 本数据集侧也统一走 RDKit 规范化, 与排除集口径一致; 解析失败才退回头部 INCHIKEY
    keys = {}
    n_fallback = 0
    for smi in set(r['smiles'] for r in records):
        k = mol_key(smi)
        if k is None:
            n_fallback += 1
        keys[smi] = k
    if n_fallback:
        print(f"    ⚠ 本数据集 {n_fallback} 个 SMILES 无法解析, 退回头部 INCHIKEY 比对")
    for r in records:
        if keys[r['smiles']] is None:
            keys[r['smiles']] = r['inchikey']

    kept = [r for r in records if keys[r['smiles']] not in excl]
    dropped = len(records) - len(kept)
    dropped_mol = len(set(r['smiles'] for r in records)) - len(set(r['smiles'] for r in kept))
    n_mol = len(set(r['smiles'] for r in records))
    print(f"    本数据集 {n_mol} 个分子中 {dropped_mol} 个与排除集重合 "
          f"({dropped_mol / n_mol * 100:.1f}%)")
    print(f"    剔除 {dropped} / {len(records)} 条谱图 ({dropped / len(records) * 100:.1f}%), "
          f"保留 {len(kept)} 条 / {len(set(r['smiles'] for r in kept))} 个分子")

    if dropped:
        print(f"    剔除前后加合物分布:")
        b = Counter(r['adduct'] for r in records)
        a = Counter(r['adduct'] for r in kept)
        for k in sorted(b, key=lambda x: -b[x]):
            print(f"      {k:<10}{b[k]:>6} → {a[k]:>6}")
    return kept


def main():
    ap = argparse.ArgumentParser(description="把 MSDIR/*.ms 解析并保存为单个 PKL")
    ap.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)
    ap.add_argument('--output-pkl', default=DEFAULT_OUTPUT_PKL)
    ap.add_argument('--limit', type=int, default=None, help='只解析前 N 个文件 (冒烟测试用)')
    ap.add_argument('--exclude-molecules-from', nargs='+', default=None, metavar='PKL',
                    help='剔除在这些 split pkl 中出现过的分子 (分子级去重, 需 rdkit)')
    args = ap.parse_args()

    print("=" * 64)
    print("构建 cib-ms 数据集 (课题组自有数据)")
    print("=" * 64)

    records = parse_dir(args.input_dir, limit=args.limit)
    if not records:
        raise SystemExit("没有解析出任何谱图, 退出")

    if args.exclude_molecules_from:
        records = dedup_against(records, args.exclude_molecules_from)
        if not records:
            raise SystemExit("去重后没有剩余谱图, 退出")

    save_pkl(records, args.output_pkl)
    report(records)
    self_check(records, args.input_dir)


if __name__ == '__main__':
    main()
