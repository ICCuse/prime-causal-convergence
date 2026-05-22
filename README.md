# S(N): A Causal Convergence Metric for Prime Distribution

> English abstract below · 中文说明见后文

A novel metric **S(N)** measuring how closely the interval-wise distribution of primes
matches the Li(x) ideal beacon. S(N) is tested across scales from 10⁴ to 10¹⁰ with
statistical null models, revealing a non-monotonic convergence pattern with an
oscillatory decay component in ln(ln N) space.

**Key results:**
- 16 data points from N=100K to N=10B
- S(10B) = 0.748854, effect size η = 899.91σ
- Power-law fit: R² = 0.995; oscillatory correction: R² → 0.998
- Estimated limit: S_∞ ≈ 0.719, oscillation frequency ω ≈ 6.5 rad/ln(ln N)

**Run:**
```powershell
python "01_起步_万级.py"      # entry point, N=10K
pypy3 "05_亿级_扫描.py"       # large-scale, needs PyPy
```

---

# 素数因果连接强度 S(N) 计算项目

## 概述

本项目定义了一个新的度量 **S(N)**，用于衡量素数区间分布与 Li(x) 理想信标之间的"因果连接强度"。

核心公式：

```
SP_i = exp( -(obs_i - ideal_i)² / (2σ²) )
S = SP_mean / (1 + δSP)
```

其中 `obs_i` 是第 i 个区间内的实际素数个数，`ideal_i` 是 Li(x) 在该区间的预测值。S 越高，素数的区间密度模式越贴近 Li(x) 的预测。

通过随机重排素数位置生成零分布，计算效应量 η 和 p 值，验证 S 的统计显著性。

## 文件说明

| 文件 | 说明 | 输出图 | 运行方式 |
|---|---|---|---|
| `01_起步_万级.py` | N=10000，K=50，概念验证 | `01_S_distribution_10K.png` | `python` |
| `02_放大_百万级.py` | N=1,000,000，K=500，尺度放大 | `02_S_distribution_1M.png` | `python` |
| `03_跨尺度趋势.py` | 4 尺度对比，发现 S(N) 递减趋势 | `03_cross_scale_S.png` | `python` (慢) |
| `04_并行_多尺度.py` | 9 尺度点，N 推到 5M，S(N) 曲线 + 分解 | `04_multiscale_S_decomposition.png` | `python` |
| `05_亿级_扫描.py` | N 推到 100M，PyPy + bytearray + 多进程 | `05_100M_scan.png` | **PyPy** |
| `06_百亿_终极扫描.py` | N 推到 10B，分段筛 8 核并行，16 点 | `06_10B_scan.png` | **PyPy** |
| `07_局部窗口_欧拉点.py` | [1500,1800] 局部 S 验证 | `07_local_window_euler.png` | `python` |
| `08_宽域_相变.py` | [1700,3000] 宽域逐区间素数对比 + SP 相变 | `08_wide_phase_transition.png` | `python` |
| `09_极限常数_拟合.py` | 16 点拟合 S_∞，幂律 + 振荡修正 | `09_convergence_fit.png` | `python` |

## 运行方式

普通 Python（01-04, 07-09）：
```powershell
cd "f:\数据实验"
python "01_起步_万级.py"
```

PyPy（05-06，需大量内存和时间）：
```powershell
cd "f:\数据实验"
pypy3 "05_亿级_扫描.py"
pypy3 "06_百亿_终极扫描.py"
```

## 核心结果

- 16 点 S(N) 数据从 N=100K 到 N=10B
- S(10B) = 0.748854，效应量 η = 899.91σ
- S(N) 非单调收敛，存在系统振荡
- 振荡修正拟合：R² = 0.998，ω ≈ 6.5 rad/ln(ln N)，β ≈ 0.2
- S_∞ 估计约 0.719

## 依赖

- numpy, scipy（09 需要）
- matplotlib（所有均需要）
- PyPy 3.11（05、06 推荐）

## 作者说明

这是一个**计算现象学**（computational phenomenology）项目——它描述了一个观察到的数值事实，而非宣称任何数学定理。

- 目前 16 个数据点、7 参数拟合仍存在过拟合风险，需要更大尺度的数据验证
- S(N) 的解析形式和数学动机暂不公开，仓库聚焦在计算方法和数值结果
- 欢迎基于计算层面的合作（更大尺度扫描、拟合方法改进、代码优化）
- 如有解读、合作或疑问，请通过下方邮箱联系

## 联系

wangkukushe@163.com
