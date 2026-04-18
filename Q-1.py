import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import pandas as pd

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
sns.set_palette("husl")

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取表1数据
data = pd.DataFrame({
    '时刻': ['07:00', '07:02', '07:04', '07:06', '07:08', '07:10', '07:12', '07:14', '07:16', '07:18', '07:20', '07:22',
             '07:24', '07:26', '07:28', '07:30', '07:32', '07:34', '07:36', '07:38', '07:40', '07:42', '07:44', '07:46',
             '07:48', '07:50', '07:52', '07:54', '07:56', '07:58', '08:00', '08:02', '08:04', '08:06', '08:08', '08:10',
             '08:12', '08:14', '08:16', '08:18', '08:20', '08:22', '08:24', '08:26', '08:28', '08:30', '08:32', '08:34',
             '08:36', '08:38', '08:40', '08:42', '08:44', '08:46', '08:48', '08:50', '08:52', '08:54', '08:56',
             '08:58'],
    '时间t': list(range(60)),
    '主路3车流量': [7.00, 8.50, 10.00, 11.50, 13.00, 14.50, 16.00, 17.50, 19.00, 20.50, 22.00, 23.50, 25.00, 26.50,
                    28.00, 29.50, 31.00, 32.50, 34.00, 35.50, 37.00, 38.50, 40.00, 41.50, 43.00, 44.50, 46.00, 47.50,
                    49.00, 50.50, 52.00, 51.50, 51.00, 50.50, 50.00, 49.50, 49.00, 48.50, 48.00, 47.50, 47.00, 46.50,
                    46.00, 45.50, 45.00, 44.50, 44.00, 43.50, 43.00, 42.50, 42.00, 41.50, 41.00, 40.50, 40.00, 39.50,
                    39.00, 38.50, 38.00, 37.50]
})

# 提取时间和车流量数据
t = data['时间t'].values
flow = data['主路3车流量'].values

# 观察数据趋势，确定拐点
t_breakpoint = 30  # 对应时刻为8:00

def objective(params):
    a, b, c, d, e, g = params
    f1 = a * t + b
    f2 = np.zeros_like(t, dtype=float)
    f2[t <= t_breakpoint] = c * t[t <= t_breakpoint] + d
    f2[t > t_breakpoint] = e * t[t > t_breakpoint] + g
    f3_pred = f1 + f2
    error = np.sum((f3_pred - flow) ** 2)
    penalty = 0
    # 确保支路1斜率为正且较大
    if a <= 0.3:  # Changed from 0 to 0.3 to encourage steeper slope
        penalty += 1000 * np.abs(a - 0.3)
    if np.any(f1 < 0):
        penalty += 1000 * np.sum(np.abs(f1[f1 < 0]))
    if np.any(f2 < 0):
        penalty += 1000 * np.sum(np.abs(f2[f2 < 0]))
    continuity_error = np.abs(c * t_breakpoint + d - (e * t_breakpoint + g))
    penalty += 1000 * continuity_error
    return error + penalty

# 设定初始参数，确保a为正
initial_params = [0.5, 3.0, 0.5, 2.0, -0.5, 47.0]  # Increased a from 0.2 to 0.5

# 优化求解
result = minimize(objective, initial_params, method='Nelder-Mead')
a, b, c, d, e, g = result.x

# 计算支路1和支路2的流量
f1 = a * t + b
f2 = np.zeros_like(t, dtype=float)
f2[t <= t_breakpoint] = c * t[t <= t_breakpoint] + d
f2[t > t_breakpoint] = e * t[t > t_breakpoint] + g

# 计算主路3的预测流量
f3_pred = f1 + f2

# 计算均方根误差
rmse = np.sqrt(np.mean((f3_pred - flow) ** 2))

# 绘制结果 - 使用seaborn增强效果
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.lineplot(x=t, y=flow, marker='o', label='主路3实际车流量', linewidth=2)
sns.lineplot(x=t, y=f3_pred, label='主路3预测车流量', linewidth=2, linestyle='--')
plt.xlabel('时间t (相对于7:00的分钟数/2)', fontsize=12)
plt.ylabel('车流量', fontsize=12)
plt.title('问题1：主路3车流量预测结果', fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.subplot(2, 1, 2)
sns.lineplot(x=t, y=f1, label='支路1车流量', linewidth=2)
sns.lineplot(x=t, y=f2, label='支路2车流量', linewidth=2)
plt.axvline(x=t_breakpoint, color='k', linestyle='--', label=f'支路2拐点 (t={t_breakpoint})')
plt.xlabel('时间t (相对于7:00的分钟数/2)', fontsize=12)
plt.ylabel('车流量', fontsize=12)
plt.title('问题1：各支路车流量', fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

# 保存第一个图表
plt.savefig('Q-1_预测结果.png', dpi=300, bbox_inches='tight')

# 堆叠图 - 使用seaborn增强效果
plt.figure(figsize=(14, 8))
sns.lineplot(x=t, y=flow, marker='o', label='主路3实际车流量', alpha=0.7, linewidth=2)
sns.lineplot(x=t, y=f3_pred, label='主路3预测车流量', linewidth=2, linestyle='--')
plt.fill_between(t, 0, f1, alpha=0.3, label='支路1车流量', color='green')
plt.fill_between(t, f1, f1 + f2, alpha=0.3, label='支路2车流量', color='purple')
plt.xlabel('时间t (相对于7:00的分钟数/2)', fontsize=12)
plt.ylabel('车流量', fontsize=12)
plt.title('问题1：支路车流量叠加及主路车流量比较', fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

# 保存第二个图表
plt.savefig('Q-1_堆叠图.png', dpi=300, bbox_inches='tight')

# 创建交互式图表
fig = go.Figure()

# 添加实际数据
fig.add_trace(go.Scatter(
    x=t,
    y=flow,
    mode='lines+markers',
    name='主路3实际车流量',
    line=dict(color='blue', width=2),
    marker=dict(size=6)
))

# 添加预测数据
fig.add_trace(go.Scatter(
    x=t,
    y=f3_pred,
    mode='lines',
    name='主路3预测车流量',
    line=dict(color='red', width=2, dash='dash')
))

# 添加支路1数据
fig.add_trace(go.Scatter(
    x=t,
    y=f1,
    mode='lines',
    name='支路1车流量',
    line=dict(color='green', width=2),
    fill='tozeroy',
    fillcolor='rgba(0,255,0,0.2)'
))

# 添加支路2数据
fig.add_trace(go.Scatter(
    x=t,
    y=f2,
    mode='lines',
    name='支路2车流量',
    line=dict(color='purple', width=2),
    fill='tonexty',
    fillcolor='rgba(128,0,128,0.2)'
))

# 更新布局
fig.update_layout(
    title='问题1：车流量预测与分解',
    xaxis_title='时间t (相对于7:00的分钟数/2)',
    yaxis_title='车流量',
    hovermode='x unified',
    showlegend=True,
    template='plotly_white',
    height=600,
    width=1000
)

# 保存交互式图表
fig.write_html('Q-1_交互式图表.html')

# 保存结果到文件
with open('Q-1_result.txt', 'w', encoding='utf-8') as f:
    f.write("优化结果：\n")
    f.write(f"支路1函数参数 (a, b): ({a:.6f}, {b:.6f})\n")
    f.write(f"支路2函数参数 (c, d): ({c:.6f}, {d:.6f})\n")
    f.write(f"支路2函数参数 (e, g): ({e:.6f}, {g:.6f})\n")
    f.write(f"支路2拐点 (t_breakpoint): {t_breakpoint}\n")
    f.write(f"\nRMSE: {rmse:.6f}\n")
    f.write("\n函数表达式：\n")
    f.write(f"支路1: f1(t) = {a:.6f}*t + {b:.6f}\n")
    f.write(f"支路2:\n")
    f.write(f"  当 t <= {t_breakpoint} 时: f2(t) = {c:.6f}*t + {d:.6f}\n")
    f.write(f"  当 t > {t_breakpoint} 时: f2(t) = {e:.6f}*t + {g:.6f}\n")

# 打印结果
print("优化结果：")
print(f"支路1函数参数 (a, b): ({a:.6f}, {b:.6f})")
print(f"支路2函数参数 (c, d): ({c:.6f}, {d:.6f})")
print(f"支路2函数参数 (e, g): ({e:.6f}, {g:.6f})")
print(f"RMSE: {rmse:.6f}")

# 展示结果表格
result_df = pd.DataFrame({
    '时刻': data['时刻'],
    '时间t': t,
    '主路3实际车流量': flow,
    '主路3预测车流量': f3_pred,
    '支路1车流量': f1,
    '支路2车流量': f2
})
print("\n结果表格（部分展示）：")
print(result_df.head(10))

# 打印函数表达式
print("\n函数表达式：")
print(f"支路1: f1(t) = {a:.6f}*t + {b:.6f}")
print(f"支路2: f2(t) = {c:.6f}*t + {d:.6f} (t <= {t_breakpoint})")
print(f"支路2: f2(t) = {e:.6f}*t + {g:.6f} (t > {t_breakpoint})")

# 显示图表
plt.show()