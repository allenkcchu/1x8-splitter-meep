# 1×8 MMI Splitter Inverse Design — Design Log

**Platform:** SiN 800nm platform (SiO₂ cladding, λ=1550nm)
**Started:** 2026-05-10

---

## 目標

設計一個 single-stage 1×8 MMI power splitter：
- 不用傳統樹狀（3-stage cascade），改用單一 MMI 元件做 1 input → 8 output
- 利用 inverse design 找到最佳 MMI 形狀，不受限於矩形 MMI 的標準設計規則
- 目標：均等功率分配（每 port 各得 1/8），最大化總傳輸效率

---

## 平台參數

| 參數 | 值 |
|------|-----|
| 材料（core） | Si₃N₄ (SiN) |
| 材料（cladding） | SiO₂ |
| SiN 厚度 | 800 nm（固定） |
| 波長 | 1550 nm |
| SiN 折射率 | 1.996 |
| SiO₂ 折射率 | 1.444 |
| Slab TE0 n_eff（EIM） | 1.8836 |
| 單模截止寬度 | 1.282 µm |
| 使用波導寬度 | 1.0 µm |

---

## 模擬方法

### Effective Index Method (EIM)
- 3D 結構簡化為 2D FDTD
- SiN 800nm slab 的 TE0 有效折射率 n_eff=1.8836 作為 2D core index
- SiO₂ n=1.444 作為 2D cladding index
- 求解 TE0 slab dispersion equation（對稱 SiO₂ cladding）

### 設計區域
- MMI 設計區：30 µm × 14 µm
- 8 個 output port，間距 1.75 µm（center-to-center）
- 設計變量 grid：240 × 112 pixels（8 px/µm）
- FDTD resolution：15 px/µm

### Adjoint Topology Optimization
- Meep 1.25.0-beta + `meep.adjoint`
- 每次 iteration = 1 次 forward FDTD + 1 次 adjoint FDTD
- Objective：最大化 8 個 output port 的 TE0 模耦合功率加總
- Optimizer：Adam（lr=0.02, β₁=0.9, β₂=0.999）

---

## 設計流程

### Stage 1：Gaussian Filter（探索 topology）
- Filter：Gaussian σ=300nm（軟性最小特徵尺寸約束）
- Beta schedule：4 → 8 → 16 → 32（iter 0/30/55/80）
- Iterations：100

### Stage 2：Conic Filter（製程約束收斂）
- 以 Stage 1 結果為 warm start
- Filter 換成 conic（半徑 300nm），對 solid/void minimum feature size 有更嚴格的數學保證
- Beta schedule：32 → 48 → 64（iter 0/20/35）
- Iterations：50，lr=0.01

---

## 遇到的問題與解法

| 問題 | 原因 | 解法 |
|------|------|------|
| `scipy` 未安裝，`meep.adjoint` 無法載入 | pmp conda env 缺 dependency | `pip install scipy autograd` |
| `update_design` 報錯：not 1D | API 要求 1D flat array，誤傳了 reshape 後的 2D array | 所有 `update_design` 呼叫改傳 flat 1D array |
| gradient shape 為 `()`，無法 reshape | `opt()` 回傳的 `dJ` 是 flat numpy array（非 list），`dJ[0]` 是第一個 pixel 的 gradient | 改用 `np.asarray(dJ).flatten()` |
| Jupyter notebook kernel 找不到 `pmp` | pmp env 未註冊為 Jupyter kernel | `python -m ipykernel install --user --name pmp` |
| Jupyter server 關掉後 background process 被殺 | bash -c 的子 process 隨 session 結束 | 改用 `Start-Process wsl ... -WindowStyle Minimized` 開獨立視窗 |
| Objective J 值為負（-53000） | variance penalty 係數過大（3×），未 normalized power 放大誤差 | 移除 variance penalty，改為純 maximize 總功率，讓 MMI 物理自然導向均等分配 |
| 傳輸效率量測 > 100%（第一版 eval） | output monitor 大小 3µm > port pitch 1.75µm，相鄰 monitor 大幅重疊 | 改用 1.4µm per-port monitor（< pitch，無重疊）+ 全寬 cross-section 做 sanity check |
| 傳輸效率量測仍 > 100%（第二版 eval，eigenmode） | input flux monitor 與 EigenModeSource 在同一 x 位置，量到的 Poynting flux 不準確 | 將 input monitor 移至 source 右側 1µm（x = -16µm），避免 source-plane singularity |

---

## 結果

### Stage 1（Gaussian filter, 100 iterations）
> ✅ 完成（2026-05-10）

| 指標 | 數值 |
|------|------|
| 初始 J | 565.5 |
| 最終 J | 1766.4 |
| Total transmission（full cross-section） | **86.7%** |
| Gray pixels（5–95%）| 11.0%（大部分已 binarized） |
| Mean per-port T | 10.7%（理想 12.5%） |
| Std dev（port-to-port） | 8.5% |
| Max imbalance | 23.3% |

**Per-port transmission（binary design，Gaussian filter σ=300nm，beta=64）：**

| Port | y (µm) | T (%) | 相對理想值 |
|------|--------|-------|-----------|
| 1 | −6.125 | 24.6% | +12.1% |
| 2 | −4.375 |  1.4% | −11.1% |
| 3 | −2.625 | 14.3% | +1.8% |
| 4 | −0.875 |  4.4% | −8.1% |
| 5 | +0.875 |  7.3% | −5.2% |
| 6 | +2.625 |  9.6% | −2.9% |
| 7 | +4.375 |  1.3% | −11.2% |
| 8 | +6.125 | 22.8% | +10.3% |

**觀察：**
- 總效率 86.7% 表示 ~13% 的光散射進非導模
- 分布高度不均：外側 Port 1/8 獲得過多功率，Port 2/7 接近 0
- J 最大化僅優化「總功率」，無 uniformity 約束，導致優化器傾向集中功率於耦合效率較高的 port
- Stage 2 的 conic filter 可改善 binarization，但不會自動解決 uniformity 問題
- 若需均等分配，應在 objective 中加入 variance penalty 或改用 min-port 最大化

### Stage 1 v2（Uniformity penalty + Geometry optimization, 100 iterations）
> ✅ 完成（2026-05-11）

**目標改動：**
- Objective 加入 uniformity penalty：`J = total × (1 − α × norm_CV²)`
  - `norm_CV² = Σ(P_i−mean)² / (N×mean²)`（無量綱，scale-invariant）
  - α schedule：0（iter 0–19）→ 0.05（iter 20–49）→ 0.12（iter 50–99）
  - α < 1/(N−1) ≈ 0.143，確保 J 恆為正
- Joint geometry optimization（central-difference FD，每 8 iter 更新一次）：
  - `port_pitch` ∈ [1.20, 1.85] µm；`wg_w_in / wg_w_out` ∈ [0.80, 1.50] µm

**最終幾何參數（優化後幾乎不變）：**

| 參數 | 初始值 | 最終值 | 說明 |
|------|--------|--------|------|
| port_pitch | 1.750 µm | **1.798 µm** | 稍微拉大間距 |
| wg_w_in | 1.000 µm | **1.009 µm** | 幾乎不變 |
| wg_w_out | 1.000 µm | **0.997 µm** | 幾乎不變 |

→ 幾何 FD gradient 極小，說明 1.0µm 寬度和 1.75µm pitch 已接近當前 objective 的最佳值

**效能指標：**

| 指標 | v1（純 J max） | v2（+uniformity） | 說明 |
|------|---------------|------------------|------|
| 初始 J | 565.5 | 565.5 | 相同起點 |
| 最終 J | 1766.4 | 1677.5 | v2 低因 penalty 壓制 |
| Total T | **86.7%** | **86.0%** | 相近 |
| Mean/port | 10.7% | 10.7% | 相同 |
| Std dev | 8.52% | **6.67%** | ↓ 22%（改善） |
| Max imbalance | 23.3% | **22.6%** | 微幅改善 |

**Per-port transmission（v2）：**

| Port | y (µm) | T (%) | v1 T (%) | 相對理想 |
|------|--------|-------|----------|---------|
| 1 | −6.292 | 13.1% | 24.6% | +0.6% |
| 2 | −4.494 |  3.1% |  1.4% | −9.4% |
| 3 | −2.696 | 11.5% | 14.3% | −1.0% |
| 4 | −0.899 |  9.9% |  4.4% | −2.6% |
| 5 | +0.899 |  7.8% |  7.3% | −4.7% |
| 6 | +2.696 | 11.3% |  9.6% | −1.2% |
| 7 | +4.494 |  3.1% |  1.3% | −9.4% |
| 8 | +6.292 | **25.7%** | 22.8% | +13.2% |

**觀察：**
- Uniformity penalty 成功將 Port 1 從 24.6% 壓至 13.1%，但 Port 8 反而升到 25.7%（設計不對稱）
- Port 2/7 仍然偏低（~3%），對應 MMI 中「低模密度」的區域
- 幾何參數幾乎不動：說明現有 port pitch/寬度已接近最佳，uniformity 問題根源在 topology，不在幾何
- std 從 8.5% 改善到 6.7%（~22% 改善），但仍離理想（std→0）很遠
- 根本瓶頸：30µm × 14µm 的 MMI 很難同時達到高效率 + 8-way 均等分配；可能需要更長的 MMI 或不同的 objective

### Parametric Study（L × W 掃描，2026-05-11 開始）
> 🔄 進行中

**動機：** L=30µm 設計 std=6.7%，Port 2/7 僅 3%，根本瓶頸是 MMI 太短導致模態分布不均。

**設計矩陣：**

| 變體 | mmi_L | mmi_W | 設計 vars | FDTD cell | 資料夾 | 狀態 |
|------|-------|-------|----------|-----------|--------|------|
| L30W14（基準） | 30 µm | 14 µm | 240×112 | 38×18 µm² | `stage1_v2/` | ✅ 完成 |
| L40W14 | 40 µm | 14 µm | 320×112 | 48×18 µm² | `stage1_L40W14/` | 🔄 進行中 |
| L40W16 | 40 µm | 16 µm | 320×128 | 48×20 µm² | `stage1_L40W16/` | 🔄 進行中 |
| L50W14 | 50 µm | 14 µm | 400×112 | 58×18 µm² | `stage1_L50W14/` | 🔄 進行中 |

**共用設定：**
- Objective：`J = total × (1 − α × norm_CV²)`，α schedule 同 v2
- Joint geometry optimization：port_pitch / wg_w_in / wg_w_out（FD，每 8 iter）
- W=14：port_pitch ∈ [1.20, 1.83]；W=16：port_pitch ∈ [1.50, 2.07]
- Script：`scripts/stage1_param.py --mmi_L {L} --mmi_W {W}`

**評比指標（跑完後填入）：**

| 變體 | J_final | Total T | Std dev | Max imbalance |
|------|---------|---------|---------|---------------|
| L30W14 | 1677.5 | 86.0% | 6.67% | 22.6% |
| L40W14 | — | — | — | — |
| L40W16 | — | — | — | — |
| L50W14 | — | — | — | — |

### Stage 2（Conic filter，以最佳 Stage 1 為 warm start）
> ⬜ 待 parametric study 完成後執行

---

## 檔案結構

```
meep_1x8_progress/
├── meep_1x8_splitter.ipynb         Stage 1 notebook（executed）
├── meep_1x8_stage2.ipynb           Stage 2 notebook（ready）
├── meep_1x8_eval.ipynb             Transmission eval notebook（executed）
├── meep_1x8_design_log.md          本文件
├── run_stage1.sh                   Stage 1 WSL 執行腳本
├── run_stage2.sh                   Stage 2 WSL 執行腳本
├── run_eval.sh                     Transmission eval WSL 執行腳本
├── log.json                        Stage 1 iteration log（含完整 J history）
├── progress_latest.png             Stage 1 最新設計圖
├── progress_iter***.png            Stage 1 各 checkpoint 快照
├── x_final.npy                     Stage 1 最終設計參數
├── transmission_eval.png           Binary design + Ez field + per-port bar chart
├── transmission_eval.json          Stage 1 binary design 傳輸效率數據
├── eval.log                        Eval 執行 log
├── scripts/
│   ├── stage1_optimize.py          Stage 1 standalone Python script
│   ├── stage2_refine.py            Stage 2 standalone Python script
│   └── eval_transmission.py        Transmission eval standalone script
├── stage1_v2/                      L=30 W=14（已完成，同 stage1_L30W14）
│   ├── log.json                    v2 iteration log
│   ├── progress_iter***.png        v2 設計圖快照
│   ├── x_final_v2.npy              v2 最終設計參數
│   ├── geo_final_v2.json           v2 最終幾何參數
│   ├── transmission_eval_v2.png    v2 binary design + Ez + per-port bar chart
│   └── transmission_eval_v2.json   v2 傳輸效率數據
├── stage1_L40W14/                  L=40 W=14（進行中）
│   ├── config.json                 run 配置（mmi_L/W, bounds, schedule 等）
│   ├── run.log                     執行 log
│   ├── log.json                    iteration log
│   ├── x_final.npy / x_latest.npy 設計參數
│   └── geo_final.json              最終幾何參數
├── stage1_L40W16/                  L=40 W=16（進行中）
│   └── ...（同上）
├── stage1_L50W14/                  L=50 W=14（進行中）
│   └── ...（同上）
├── scripts/
│   ├── stage1_optimize.py          Stage 1 v1 standalone
│   ├── stage1_v2_optimize.py       Stage 1 v2 standalone（L=30 W=14）
│   ├── stage1_param.py             通用參數化 script（--mmi_L --mmi_W）
│   ├── stage2_refine.py            Stage 2 standalone
│   ├── eval_transmission.py        Stage 1/2 transmission eval
│   └── eval_v2.py                  Stage 1 v2 transmission eval
├── run_L40W14.sh / run_L40W16.sh / run_L50W14.sh   各 variant 執行腳本
└── stage2/                         Stage 2（待執行）
```

---

## 後續計劃

- [ ] **Stage 2** 執行：v2 結果 warm start + conic filter（R=300nm），50 iterations
- [ ] **Uniformity 根本改善**：
  - 方案 A（推薦）：延長 MMI（30→40+µm），給優化器更多空間分配模態
  - 方案 B：改用 max-min objective（最大化最弱 port），autograd 用 softmin 近似
  - 方案 C：固定對稱性約束（x2d = x2d[::-1,:]），強制設計對稱，消除 Port 1/8 不對稱
- [ ] GDS export：用 KLayout + GDSFactory 將 binarized design 轉成 GDS，DRC check（最小特徵 ≥ 300nm）
- [ ] 多波長：考慮 1520–1580 nm broadband optimization
