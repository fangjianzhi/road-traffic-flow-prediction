# 复现指南

## 环境

- Python 3.10 或更高版本
- Windows、macOS 或 Linux
- 推荐至少 4 GB 可用内存

安装依赖：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果系统没有 `SimHei` 或 `Microsoft YaHei`，Matplotlib 会回退到可用字体；图中的中文可能出现缺字，但数值计算不受影响。

## 脚本入口

| 脚本 | 内容 | 数据来源 |
|---|---|---|
| `src/problem_1_dual_branch.py` | 双支路汇入、分段线性拟合 | 脚本内置表 1 序列 |
| `src/problem_2_four_branch.py` | 四支路汇入与传播延迟 | 脚本内置表 2 序列 |
| `src/problem_3_signal_control.py` | 信号灯控制与误差分析 | `data/competition_attachment.xlsx` 表 3 |
| `src/problem_4_extended_model.py` | 扩展支路模型与全局/局部优化 | 脚本内置表 4 序列 |
| `src/problem_5_sampling_optimization.py` | 关键观测时刻选择 | 脚本内置模型特征 |

逐个运行示例：

```bash
python src/problem_1_dual_branch.py
python src/problem_2_four_branch.py
python src/problem_3_signal_control.py
python src/problem_4_extended_model.py
python src/problem_5_sampling_optimization.py
```

所有生成文件统一写入 `outputs/generated/`。绘图窗口在桌面环境中可能自动打开；在无图形界面的服务器上可先设置 Matplotlib 的 `Agg` 后端。

## 可复现性说明

- 局部优化通常能够稳定复现相近结果，但不同 SciPy/BLAS 版本可能导致末位数值差异。
- 差分进化脚本计算量较大；若只想验证流程，可临时降低 `maxiter` 和 `popsize`。
- `assets/` 中是论文整理阶段保留的代表性图，不会在运行脚本时被覆盖。
- 论文排版图与仓库脚本输出在配色、字号或图例上可能存在差异，数值模型与研究问题保持一致。

## 最小检查

```bash
python -m compileall -q src
```

该命令只检查语法。完整验证需实际运行脚本并检查 `outputs/generated/` 中的结果。
