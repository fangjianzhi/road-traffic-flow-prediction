import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取表2数据
data = pd.DataFrame({
    '时刻': ['07:00', '07:02', '07:04', '07:06', '07:08', '07:10', '07:12', '07:14', '07:16', '07:18', '07:20', '07:22',
             '07:24', '07:26', '07:28', '07:30', '07:32', '07:34', '07:36', '07:38', '07:40', '07:42', '07:44', '07:46',
             '07:48', '07:50', '07:52', '07:54', '07:56', '07:58', '08:00', '08:02', '08:04', '08:06', '08:08', '08:10',
             '08:12', '08:14', '08:16', '08:18', '08:20', '08:22', '08:24', '08:26', '08:28', '08:30', '08:32', '08:34',
             '08:36', '08:38', '08:40', '08:42', '08:44', '08:46', '08:48', '08:50', '08:52', '08:54', '08:56',
             '08:58'],
    '时间t': list(range(60)),
    '主路5车流量': [32.5, 34.1, 35.7, 37.3, 38.9, 40.5, 53.1, 54.7, 56.3, 57.9, 59.5, 61.1, 62.7, 64.3, 51.9, 53.5,
                    55.1, 56.7, 71.3, 71.9, 72.5, 73.1, 73.7, 74.3, 74.9, 75.5, 64.5, 64.5, 64.5, 64.5, 64.5, 64.5,
                    64.5, 64.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.9, 76.3, 76.7, 63.1, 63.5, 63.9, 64.3, 78.7, 79.1,
                    79.5, 79.9, 80.3, 80.7, 81.1, 81.5, 70.9, 71.3, 71.7, 72.1, 72.5, 72.9]
})

# 提取时间和车流量数据
t = data['时间t'].values
flow = data['主路5车流量'].values

# 延迟为2分钟
delay = 1  # 对应2分钟

def branch_flow(t, params):
    a, b1, b2, b3, t_break1, t_break2, c1, c2, c3, t_break3, d1, d2, d3, d4 = params

    # 支路1：稳定
    flow1 = np.full_like(t, a, dtype=float)

    # 支路2：分段线性
    flow2 = np.zeros_like(t, dtype=float)
    flow2[(t <= t_break1)] = b1 * t[(t <= t_break1)] + b2
    flow2[(t > t_break1) & (t <= t_break2)] = b3
    flow2[(t > t_break2)] = b1 * (t[(t > t_break2)] - t_break2) + b3

    # 支路3：先线性增长后稳定
    flow3 = np.zeros_like(t, dtype=float)
    flow3[t <= t_break3] = c1 * t[t <= t_break3] + c2
    flow3[t > t_break3] = c3

    # 支路4：周期性规律，使用正弦函数
    flow4 = d1 * np.sin(d2 * t + d3) + d4

    return flow1, flow2, flow3, flow4

def main_flow(t, params):
    flow1, flow2, flow3, flow4 = branch_flow(t, params)

    # 延迟处理：支路1和支路2的车流需要2分钟（delay=1）才能到达主路5的监测点
    flow1_delayed = np.zeros_like(t, dtype=float)
    flow2_delayed = np.zeros_like(t, dtype=float)

    # 对于t>=delay的时刻，使用t-delay时刻的支路1和支路2流量
    mask = t >= delay
    flow1_delayed[mask] = flow1[np.where(mask)[0] - delay]
    flow2_delayed[mask] = flow2[np.where(mask)[0] - delay]

    # 对于t<delay的时刻，假设支路1和支路2的流量为初始值
    flow1_delayed[~mask] = flow1[0]
    flow2_delayed[~mask] = flow2[0]

    # 主路5车流量 = 支路1(延迟) + 支路2(延迟) + 支路3 + 支路4
    return flow1_delayed + flow2_delayed + flow3 + flow4

def objective(params):
    flow_pred = main_flow(t, params)
    mse = np.mean((flow_pred - flow) ** 2)
    flow1, flow2, flow3, flow4 = branch_flow(t, params)
    penalty = 0
    if np.any(flow1 < 0):
        penalty += 1000 * np.sum(np.abs(flow1[flow1 < 0]))
    if np.any(flow2 < 0):
        penalty += 1000 * np.sum(np.abs(flow2[flow2 < 0]))
    if np.any(flow3 < 0):
        penalty += 1000 * np.sum(np.abs(flow3[flow3 < 0]))
    if np.any(flow4 < 0):
        penalty += 1000 * np.sum(np.abs(flow4[flow4 < 0]))
    return mse + penalty

# 设定初始参数
initial_params = [
    20.0, 0.5, 5.0, 20.0, 24.0, 37.0, 0.6, 5.0, 25.0, 20.0, 5.0, 0.5, 0.0, 10.0
]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution  # 改为使用差分进化算法
import pandas as pd

# [...] 前面的数据准备、模型定义等部分保持不变

# 差分进化算法
bounds = [
    (5.0, 30.0), (0.0, 2.0), (0.0, 20.0), (10.0, 30.0), (23.0, 25.0), (36.0, 38.0),
    (0.0, 2.0), (0.0, 20.0), (10.0, 40.0), (15.0, 25.0), (1.0, 15.0), (0.1, 1.0),
    (-np.pi, np.pi), (5.0, 20.0)
]

# 执行差分进化算法
result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=1000,        # 增加迭代次数
    popsize=30,          # 增加种群大小
    mutation=(0.5, 1),   # 变异因子范围
    recombination=0.7,   # 交叉概率
    tol=1e-6,            # 设置收敛容差
    disp=True            # 显示优化过程
)
params = result.x


# 以下部分保持不变，包含输出、绘图和保存结果的代码
print("优化结果：")
print(f"支路1稳定流量 (a): {params[0]:.4f}")
print(f"支路2参数 (b1, b2, b3): ({params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f})")
print(f"支路2转折点 (t_break1, t_break2): ({params[4]:.4f}, {params[5]:.4f})")
print(f"支路3参数 (c1, c2, c3): ({params[6]:.4f}, {params[7]:.4f}, {params[8]:.4f})")
print(f"支路3稳定开始时刻 (t_break3): {params[9]:.4f}")
print(f"支路4参数 (d1, d2, d3, d4): ({params[10]:.4f}, {params[11]:.4f}, {params[12]:.4f}, {params[13]:.4f})")

flow1, flow2, flow3, flow4 = branch_flow(t, params)
flow_pred = main_flow(t, params)
rmse = np.sqrt(np.mean((flow_pred - flow) ** 2))
print(f"RMSE: {rmse:.6f}")

t_730 = 15
t_830 = 45
flow1_730, flow2_730, flow3_730, flow4_730 = branch_flow(np.array([t_730]), params)
flow1_830, flow2_830, flow3_830, flow4_830 = branch_flow(np.array([t_830]), params)

print("\n7:30时刻各支路车流量：")
print(f"支路1: {flow1_730[0]:.2f}")
print(f"支路2: {flow2_730[0]:.2f}")
print(f"支路3: {flow3_730[0]:.2f}")
print(f"支路4: {flow4_730[0]:.2f}")

print("\n8:30时刻各支路车流量：")
print(f"支路1: {flow1_830[0]:.2f}")
print(f"支路2: {flow2_830[0]:.2f}")
print(f"支路3: {flow3_830[0]:.2f}")
print(f"支路4: {flow4_830[0]:.2f}")

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(t, flow, 'bo-', label='主路5实际车流量')
plt.plot(t, flow_pred, 'r--', label='主路5预测车流量')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题2：主路5车流量预测结果')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, flow1, 'g-', label='支路1车流量')
plt.plot(t, flow2, 'm-', label='支路2车流量')
plt.plot(t, flow3, 'c-', label='支路3车流量')
plt.plot(t, flow4, 'y-', label='支路4车流量')
plt.axvline(x=params[4], color='k', linestyle='--', label=f'支路2第一个转折点 (t={params[4]:.1f})')
plt.axvline(x=params[5], color='k', linestyle='--', label=f'支路2第二个转折点 (t={params[5]:.1f})')
plt.axvline(x=params[9], color='b', linestyle='--', label=f'支路3稳定开始时刻 (t={params[9]:.1f})')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题2：各支路车流量')
plt.grid(True)
plt.legend()

plt.tight_layout()

# 保存第一个图表
plt.savefig(OUTPUT_DIR / 'Q-2_主路5和各支路车流量.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(14, 8))
plt.plot(t, flow, 'bo-', label='主路5实际车流量', alpha=0.5)
plt.plot(t, flow_pred, 'r--', label='主路5预测车流量')
flow1_delayed = np.zeros_like(t, dtype=float)
flow2_delayed = np.zeros_like(t, dtype=float)
flow1_delayed[t >= delay] = flow1[np.where(t >= delay)[0] - delay]
flow2_delayed[t >= delay] = flow2[np.where(t >= delay)[0] - delay]
flow1_delayed[t < delay] = flow1[0]
flow2_delayed[t < delay] = flow2[0]
plt.fill_between(t, 0, flow1_delayed, alpha=0.3, label='支路1车流量(延迟后)')
plt.fill_between(t, flow1_delayed, flow1_delayed + flow2_delayed, alpha=0.3, label='支路2车流量(延迟后)')
plt.fill_between(t, flow1_delayed + flow2_delayed, flow1_delayed + flow2_delayed + flow3, alpha=0.3, label='支路3车流量')
plt.fill_between(t, flow1_delayed + flow2_delayed + flow3, flow1_delayed + flow2_delayed + flow3 + flow4, alpha=0.3, label='支路4车流量')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题2：支路车流量叠加及主路车流量比较')
plt.grid(True)
plt.legend()

plt.tight_layout()

# 保存第二个图表
plt.savefig(OUTPUT_DIR / 'Q-2_支路车流量叠加.png', dpi=300, bbox_inches='tight')

plt.show()

a, b1, b2, b3, t_break1, t_break2, c1, c2, c3, t_break3, d1, d2, d3, d4 = params
print("\n函数表达式：")
print(f"支路1: f1(t) = {a:.4f}")
print(f"支路2:")
print(f"  当 t <= {t_break1:.1f} 时: f2(t) = {b1:.4f}*t + {b2:.4f}")
print(f"  当 {t_break1:.1f} < t <= {t_break2:.1f} 时: f2(t) = {b3:.4f}")
print(f"  当 t > {t_break2:.1f} 时: f2(t) = {b1:.4f}*(t-{t_break2:.1f}) + {b3:.4f}")
print(f"支路3:")
print(f"  当 t <= {t_break3:.1f} 时: f3(t) = {c1:.4f}*t + {c2:.4f}")
print(f"  当 t > {t_break3:.1f} 时: f3(t) = {c3:.4f}")
print(f"支路4: f4(t) = {d1:.4f}*sin({d2:.4f}*t + {d3:.4f}) + {d4:.4f}")

with open(OUTPUT_DIR / 'Q-2_result.txt', 'w', encoding='utf-8') as f:
    f.write("优化结果：\n")
    f.write(f"支路1稳定流量 (a): {params[0]:.4f}\n")
    f.write(f"支路2参数 (b1, b2, b3): ({params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f})\n")
    f.write(f"支路2转折点 (t_break1, t_break2): ({params[4]:.4f}, {params[5]:.4f})\n")
    f.write(f"支路3参数 (c1, c2, c3): ({params[6]:.4f}, {params[7]:.4f}, {params[8]:.4f})\n")
    f.write(f"支路3稳定开始时刻 (t_break3): {params[9]:.4f}\n")
    f.write(f"支路4参数 (d1, d2, d3, d4): ({params[10]:.4f}, {params[11]:.4f}, {params[12]:.4f}, {params[13]:.4f})\n")
    f.write(f"\nRMSE: {rmse:.6f}\n")
    f.write("\n7:30时刻各支路车流量：\n")
    f.write(f"支路1: {flow1_730[0]:.2f}\n")
    f.write(f"支路2: {flow2_730[0]:.2f}\n")
    f.write(f"支路3: {flow3_730[0]:.2f}\n")
    f.write(f"支路4: {flow4_730[0]:.2f}\n")
    f.write("\n8:30时刻各支路车流量：\n")
    f.write(f"支路1: {flow1_830[0]:.2f}\n")
    f.write(f"支路2: {flow2_830[0]:.2f}\n")
    f.write(f"支路3: {flow3_830[0]:.2f}\n")
    f.write(f"支路4: {flow4_830[0]:.2f}\n")
    f.write("\n函数表达式：\n")
    f.write(f"支路1: f1(t) = {a:.4f}\n")
    f.write(f"支路2:\n")
    f.write(f"  当 t <= {t_break1:.1f} 时: f2(t) = {b1:.4f}*t + {b2:.4f}\n")
    f.write(f"  当 {t_break1:.1f} < t <= {t_break2:.1f} 时: f2(t) = {b3:.4f}\n")
    f.write(f"  当 t > {t_break2:.1f} 时: f2(t) = {b1:.4f}*(t-{t_break2:.1f}) + {b3:.4f}\n")
    f.write(f"支路3:\n")
    f.write(f"  当 t <= {t_break3:.1f} 时: f3(t) = {c1:.4f}*t + {c2:.4f}\n")
    f.write(f"  当 t > {t_break3:.1f} 时: f3(t) = {c3:.4f}\n")
    f.write(f"支路4: f4(t) = {d1:.4f}*sin({d2:.4f}*t + {d3:.4f}) + {d4:.4f}\n")
