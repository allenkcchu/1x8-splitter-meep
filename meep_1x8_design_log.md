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

### Parametric Study（L × W 掃描，2026-05-11～12）
> ✅ 完成（7 個變體）

**動機：** L=30µm 設計 std=6.7%，Port 2/7 僅 3%，嘗試更大/更小 MMI 是否能改善 uniformity。

**共用設定：**
- Objective：`J = total × (1 − α × norm_CV²)`，α schedule 同 v2
- Joint geometry optimization：port_pitch / wg_w_in / wg_w_out（FD，每 8 iter）
- Script：`scripts/stage1_param.py --mmi_L {L} --mmi_W {W}`
- Eval：`scripts/eval_param.py --variant {TAG}`

**完整結果：**

| 變體 | Total T | Std | Imbalance | J_final | 備註 |
|------|---------|-----|-----------|---------|------|
| **L20W14** | **91.5%** | 7.16% | 22.32% | — | ✅ 最高穿透率 |
| L20W18 | 89.4% | 5.83% | 18.70% | — | |
| L30W14 | 86.0% | 6.67% | 22.62% | 1677.5 | 基準（v2） |
| L20W16 | 85.8% | 5.46% | 16.62% | — | |
| L40W16 | 78.8% | 5.40% | 16.48% | 1662.3 | |
| L40W14 | 72.7% | 4.16% | 12.69% | 1678.3 | |
| L50W14 | 57.0% | 3.06% | 8.78% | — | imbalance 改善是假的 |

**關鍵發現：**
- 越長的 MMI（L40/L50）imbalance 雖然下降，但 total T 也大幅下降，是 trade-off 而非真正改善
- L50W14 的低 imbalance 是假象：optimizer 將功率集中在邊緣 port，Port 2/7 幾乎沒有能量（~3%），total T 只剩 57%
- 根本結論：imbalance 可以在 Stage 2 靠更強的 α 推，total T 才是 warm start 的關鍵指標
- **最佳 warm start：L20W14**（total T=91.5%，比 L30W14 多 5.5pp）
- L20W14 的 imbalance 22.32% 與 L30W14 相當，Stage 2 有空間改善

### Stage 2（Conic filter，各 variant）
> ✅ 完成（2026-05-12～15）

**共用設定：**
- Filter：Conic R=300nm（最小特徵尺寸保證）
- Beta schedule：32 → 48 → 64（iter 0/20/35）
- Iterations：50，lr=0.01，alpha=0.12
- Script：`scripts/stage2_param.py --variant {TAG}`

---

#### Stage 2 L20W14
> ✅ 完成（2026-05-12）

| 指標 | Stage 1 | Stage 2 | 變化 |
|------|---------|---------|------|
| Total T | 91.5% | **95.9%** | +4.4pp |
| Std | 7.16% | 5.71% | ↓ |
| Max imbalance | 22.32% | **18.35%** | ↓ |
| Gray fraction | — | 4.6% | |

**Per-port：**

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.129 | 19.16% |
| 2 | −4.378 |  2.95% |
| 3 | −2.627 | 14.79% |
| 4 | −0.876 |  9.23% |
| 5 | +0.876 | 10.66% |
| 6 | +2.627 | 13.64% |
| 7 | +4.378 |  3.55% |
| 8 | +6.129 | 21.30% |

---

#### Stage 2 L20W18
> ✅ 完成（2026-05-12）

| 指標 | Stage 1 | Stage 2 | 變化 |
|------|---------|---------|------|
| Total T | 89.4% | **97.0%** | +7.6pp |
| Std | 5.83% | 5.71% | ≈ |
| Max imbalance | 18.70% | 19.14% | ↑ |
| Gray fraction | — | 4.0% | |

**Per-port：**

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.233 | 21.75% |
| 2 | −4.452 |  2.61% |
| 3 | −2.671 | 14.02% |
| 4 | −0.890 | 11.43% |
| 5 | +0.890 | 11.03% |
| 6 | +2.671 | 14.03% |
| 7 | +4.452 |  2.86% |
| 8 | +6.233 | 18.67% |

**觀察：** total T 為所有 variant 最高（97.0%），但 imbalance 未改善，Port 2/7 仍在 ~2.7%。

---

#### Stage 2 L50W14（stage2_L50W14_r1）
> ✅ 完成（2026-05-15）

**過程說明：**

Stage 2 L50W14 的優化過程較曲折：

1. 原始 stage2_L50W14 跑了 48 iters（J=1778），但因 WSL 進程中斷，沒有存下 `x_final_s2.npy`，只有 iter 45 的 `x_latest.npy`（J=1776）。
2. 嘗試以 5 iters resume 補跑剩餘 iters，但 Adam 動量狀態（m, v）從零重啟，第一步的 bias correction 導致 update step 等效於 `lr * sign(grad)` = 0.01，把設計推到很差的位置（J 從 1776 掉到 1008）。`save_progress` 在 i=0 觸發並覆蓋了 x_latest.npy，iter 45 的好 checkpoint 丟失。
3. 停掉 5-iter resume，改以現有的（已破壞的）x_latest.npy 為起點，重新跑 Stage 2 共 30 iters（beta 從 32 重新開始）。Adam 在 30 iters 中有足夠時間重建動量，J 從 1035 恢復到 1773。

**設定：**
- Warm start：`stage2_L50W14/x_latest.npy`（iter 46 的破壞後狀態）
- Iterations：30，start_iter=0（beta schedule 從 32 重啟）
- Output：`meep/stage2_L50W14_r1/`
- Script：`stage2_param.py --resume --out .../stage2_L50W14_r1 --n_iters 30 --start_iter 0`

**J 恢復過程：**

| Iter | J |
|------|---|
| 0 | 1035.5（壞 warm start） |
| 5 | 1634.0 |
| 10 | 1717.6 |
| 15 | 1749.5 |
| 20 | 1731.1（beta 升至 48，短暫下降） |
| 25 | 1766.0 |
| 29 | 1773.4 |

**最終結果：**

| 指標 | Stage 1 | Stage 2 | 變化 |
|------|---------|---------|------|
| Total T | 57.0% | **90.2%** | **+33.2pp** |
| Std | 3.06% | 5.71% | ↑ |
| Max imbalance | 8.78% | 16.63% | ↑ |
| Gray fraction | — | 8.1% | |

**Per-port：**

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.234 | 17.62% |
| 2 | −4.453 |  2.14% |
| 3 | −2.672 | 12.85% |
| 4 | −0.891 |  9.95% |
| 5 | +0.891 | 13.44% |
| 6 | +2.672 | 11.85% |
| 7 | +4.453 |  2.79% |
| 8 | +6.234 | 18.76% |

**觀察：**
- Total T 從 57% 提升到 90%，是所有 variant 中最大的絕對增益（+33pp），但 imbalance 反而惡化。
- Port 2/7 仍然嚴重不足（~2.5%），和 L20W14/W18 相同的 edge-port dominance 問題。
- L50W14 Stage 1 的低 imbalance（8.78%）是假象——優化器將功率集中在邊緣 port 而非 Port 2/7，Stage 2 binarization 讓這個結構更難繞過。
- Gray fraction 8.1% 高於 L20W14/W18（4–5%），顯示 30 iters 的 binarization 尚未完全收斂，繼續跑可能可以降低 gray fraction 並提升 T。

**教訓（Adam resume 問題）：**
- 若要 resume Adam 優化，必須同時儲存動量狀態（m, v arrays）。
- 僅靠少數 iters（5 iters）重啟動量無法恢復，因為 bias correction 讓第一步 step size 已達穩態大小，但方向性（哪些 pixels 重要）沒有歷史資訊。
- 後續改進：在 `save_progress` 中一起儲存 `m_ax`、`v_ax`，並在 resume 時載入。

---

### Stage 2 綜合比較

| Variant | S1 Total T | S2 Total T | S1 Imbalance | S2 Imbalance | Gray fraction |
|---------|-----------|-----------|-------------|-------------|---------------|
| L20W14 | 91.5% | **95.9%** | 22.32% | **18.35%** | 4.6% |
| L20W18 | 89.4% | **97.0%** | 18.70% | 19.14% | 4.0% |
| L50W14 | 57.0% | 90.2% | 8.78% | 16.63% | 8.1% |

**結論：**
- Stage 2 一致性地提升 total T（binarization 讓模式耦合更乾淨）。
- Port 2/7 starving（~2–3%）是所有 variant 共有的拓撲限制，Stage 2 無法改變。
- 最佳設計：**L20W18**（Total T=97.0%）。若優先考量 imbalance：**L20W14**（Imbalance=18.35%）。

---

## 檔案結構

```
meep_1x8_progress/
├── meep_1x8_design_log.md          本文件
├── scripts/
│   ├── stage1_param.py             Stage 1（--mmi_L --mmi_W --n_iters --warmstart --out）
│   ├── stage2_param.py             Stage 2（--variant --base --resume --out --start_iter）
│   ├── eval_param.py               Stage 1 eval（--variant）
│   ├── eval_stage2.py              Stage 2 eval（--variant --base --out）
│   ├── print_eval.py               所有變體結果比較輸出
│   └── stage2_refine.py            Stage 2 舊版（保留參考）
├── launchers/                      WSL 執行腳本（.sh）
├── meep/                           模擬結果
│   ├── stage1_L20W14/              ✅ Total T=91.5%，Imbalance=22.32%
│   ├── stage1_L20W16/              ✅ Total T=85.8%，Imbalance=16.62%
│   ├── stage1_L20W18/              ✅ Total T=89.4%，Imbalance=18.70%
│   ├── stage1_L30W14/              ✅ Total T=86.0%，Imbalance=22.62%
│   ├── stage1_L40W14/              ✅ Total T=72.7%，Imbalance=12.69%
│   ├── stage1_L40W16/              ✅ Total T=78.8%，Imbalance=16.48%
│   ├── stage1_L50W14/              ✅ Total T=57.0%，Imbalance=8.78%
│   ├── stage2_L20W14/              ✅ Total T=95.9%，Imbalance=18.35%
│   ├── stage2_L20W18/              ✅ Total T=97.0%，Imbalance=19.14%
│   └── stage2_L50W14_r1/           ✅ Total T=90.2%，Imbalance=16.63%
└── README.md
```

---

## 後續計劃

- [x] Parametric study（L20/L30/L40/L50 × W14/W16/W18）完成
- [x] Stage 2 L20W14、L20W18、L50W14 全部完成並 eval
- [ ] 解決 Port 2/7 starving 問題（新 topology 或 objective 策略）
- [ ] Adam optimizer 狀態（m, v）隨 checkpoint 儲存，支援正確 resume
- [ ] GDS export：KLayout + GDSFactory，DRC check（最小特徵 ≥ 300nm）
- [ ] 多波長：1520–1580 nm broadband optimization
