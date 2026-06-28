import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import pinv
import itertools
from functools import lru_cache
from joblib import Parallel, delayed
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ------------------------- 通用工具函数 -------------------------
def format_time(t):
    """将时间索引转换为HH:MM格式"""
    hour = 7 + t // 30
    minute = (t * 2) % 60
    return f"{hour:02d}:{minute:02d}"


# ------------------------- 问题2优化模型 -------------------------
@lru_cache(maxsize=100)
def cached_branch1(t, param):
    """带缓存的分支1计算"""
    return param[0] if isinstance(t, (int, float)) else np.full_like(t, param[0])


def branch2_problem2(t, params):
    """向量化支路2计算"""
    t_stable_start, t_stable_end, growth_rate, stable_value = params
    t = np.asarray(t)
    flow = np.zeros_like(t, dtype=float)

    mask1 = t <= t_stable_start
    flow[mask1] = growth_rate * t[mask1]

    mask2 = (t > t_stable_start) & (t <= t_stable_end)
    flow[mask2] = stable_value

    mask3 = t > t_stable_end
    flow[mask3] = stable_value + growth_rate * (t[mask3] - t_stable_end)

    return flow


def predict_main_flow_problem2(t, delay, params):
    """向量化主流量预测"""
    params_1 = params[0:1]
    params_2 = params[1:5]
    params_3 = params[5:8]
    params_4 = params[8:12]

    t = np.asarray(t)
    t_delayed = t - delay

    flow1 = cached_branch1(tuple(t_delayed), tuple(params_1)) * (t_delayed >= 0)
    flow2 = branch2_problem2(t_delayed, params_2) * (t_delayed >= 0)
    flow3 = params_5[2] * (t >= params_5[0]) * (t <= params_5[1])  # 支路3优化计算
    flow4 = params_6[0] * np.sin(2 * np.pi * t / params_6[1] + params_6[2]) + params_6[3]  # 支路4

    return flow1 + flow2 + flow3 + flow4


# ------------------------- 问题3优化模型 -------------------------
def branch1_problem3(t, params):
    """向量化支路1计算"""
    t = np.asarray(t)
    t_start, t_peak1, t_valley, t_peak2, t_end, max_flow = params
    flow = np.zeros_like(t, dtype=float)

    # 各阶段掩码
    mask1 = (t >= t_start) & (t <= t_peak1)
    mask2 = (t > t_peak1) & (t <= t_valley)
    mask3 = (t > t_valley) & (t <= t_peak2)
    mask4 = (t > t_peak2) & (t <= t_end)
    mask5 = (t > t_end)

    flow[mask1] = max_flow * (t[mask1] - t_start) / (t_peak1 - t_start)
    flow[mask2] = max_flow - (max_flow / 2) * (t[mask2] - t_peak1) / (t_valley - t_peak1)
    flow[mask3] = max_flow / 2 + (max_flow / 2) * (t[mask3] - t_valley) / (t_peak2 - t_valley)
    flow[mask4] = max_flow
    flow[mask5] = max_flow * (1 - (t[mask5] - t_end) / (t_end - t_peak2))

    return np.clip(flow, 0, None)


# ------------------------- 观测点优化算法 -------------------------
def evaluate_subset_error(time_points, problem_type):
    """并行化误差评估"""
    time_points = sorted(time_points)
    max_gap = max((b - a for a, b in zip(time_points, time_points[1:])), default=0)

    # 关键点检查逻辑
    key_moments = {
        "problem2": [24, 37],
        "problem3": [3, 35, 47]
    }.get(problem_type, [])

    penalty = sum(10 for m in key_moments
                  if not any(abs(t - m) <= 1 for t in time_points))

    return max_gap + penalty


def find_optimal_observations(problem_type, max_points=12):
    """并行化贪心算法优化观测点选择"""
    all_points = list(range(60))
    optimal = [0, 59]

    with Parallel(n_jobs=4, prefer="threads") as parallel:
        while len(optimal) < max_points:
            candidates = [t for t in all_points if t not in optimal]

            # 并行评估候选点
            errors = parallel(
                delayed(evaluate_subset_error)(optimal + [t], problem_type)
                for t in candidates
            )

            best_idx = np.argmin(errors)
            optimal.append(candidates[best_idx])

    return sorted(optimal)


# ------------------------- 矩阵分析核心算法 -------------------------
def compute_gradient(t, problem_type):
    """参数梯度计算（示例实现）"""
    if problem_type == "problem2":
        return [1, t, t ** 2, np.sin(t)]  # 示例梯度向量
    else:
        return [np.exp(-0.1 * t), t // 10, np.log(t + 1)]


def select_by_sensitivity(problem_type, params_size):
    """基于参数敏感性的观测点选择"""
    t_candidates = np.arange(60)
    J = np.array([compute_gradient(t, problem_type) for t in t_candidates])

    # 奇异值分解选择关键点
    _, s, vt = np.linalg.svd(J, full_matrices=False)
    rank = np.sum(s > 1e-6)

    selected = []
    for i in range(min(rank, params_size)):
        max_idx = np.argmax(np.abs(vt[i, :]))
        selected.append(t_candidates[max_idx])

    return sorted(selected)


# ------------------------- 可视化与输出 -------------------------
# ------------------------- 增强版可视化函数 -------------------------
# ------------------------- 增强版可视化函数（修复布局问题）-----------------------
# ------------------------- 实心三角形标记的可视化函数 -------------------------
# ------------------------- 专业级美观可视化方案 -------------------------
# ------------------------- 修正后的可视化方案 -------------------------
# ------------------------- 高级感可视化方案 -------------------------
def visualize_results(problem2_points, problem3_points):
    """现代极简风格的可视化方案"""
    try:
        # 尝试设置现代中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用微软雅黑
    except:
        print("警告: 系统缺失推荐字体，已启用默认字体")

    # 自定义高级配色方案
    DARK_GRAY = '#3C3C3C'
    ACCENT_RED = '#B71C1C'
    ACCENT_BLUE = '#0D47A1'
    LIGHT_GRAY = '#E0E0E0'

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.4})

    # ------------------------- 问题2图表 -------------------------
    # 主元素
    ax1.scatter(problem2_points, [1] * len(problem2_points),
                marker='^',
                s=120,  # 标记大小
                c=ACCENT_RED,
                edgecolors=DARK_GRAY,
                linewidths=0.8,
                zorder=3)

    # 装饰性时间轴
    ax1.axhline(1, xmin=0.03, xmax=0.97, color=DARK_GRAY, linewidth=1.5, zorder=1)

    # 标签标注
    for t in problem2_points:
        ax1.text(t, 1.08, format_time(t),
                 ha='center', va='bottom',
                 fontsize=10, color=DARK_GRAY,
                 rotation=45,
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor=LIGHT_GRAY,
                           edgecolor='none',
                           alpha=0.8))

    # ------------------------- 问题3图表 -------------------------
    ax2.scatter(problem3_points, [1] * len(problem3_points),
                marker='^',
                s=120,
                c=ACCENT_BLUE,
                edgecolors=DARK_GRAY,
                linewidths=0.8,
                zorder=3)

    ax2.axhline(1, xmin=0.03, xmax=0.97, color=DARK_GRAY, linewidth=1.5, zorder=1)

    for t in problem3_points:
        ax2.text(t, 1.08, format_time(t),
                 ha='center', va='bottom',
                 fontsize=10, color=DARK_GRAY,
                 rotation=45,
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor=LIGHT_GRAY,
                           edgecolor='none',
                           alpha=0.8))

    # ------------------------- 统一美化设置 -------------------------
    time_ticks = np.arange(0, 60, 15)
    time_labels = [format_time(t) for t in time_ticks]

    for ax in (ax1, ax2):
        # 坐标轴设置
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, rotation=30, ha='right')
        ax.set_xlim(-2, 59)
        ax.set_ylim(0.85, 1.15)

        # Y轴设置
        ax.set_yticks([])  # 移除所有刻度
        ax.spines['left'].set_visible(False)  # 隐藏左边线

        # 添加"观测点"标签
        ax.text(-0.02, 1.02,  # 调整位置参数
                '观\n测\n点',
                transform=ax.transAxes,
                va='top',
                ha='right',
                fontsize=12,
                color=DARK_GRAY,
                linespacing=1.8,
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          edgecolor='none',
                          pad=0.3))

        # 边框优化
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(DARK_GRAY)

        # 网格系统
        ax.grid(which='major', axis='x', linestyle='--', color=LIGHT_GRAY, alpha=0.8)
        ax.grid(which='minor', axis='x', linestyle=':', color=LIGHT_GRAY, alpha=0.4)

        # 背景处理
        ax.set_facecolor('#FAFAFA')

    # 标题设置
    fig.suptitle('交通观测点优化分布',
                 y=0.95,
                 fontsize=16,
                 color=DARK_GRAY,
                 fontweight='bold')

    ax1.set_title("问题二：主路5 - 稳态流量监测",
                  pad=12,
                  fontsize=12,
                  color=ACCENT_RED,
                  loc='left')
    ax2.set_title("问题三：主路4 - 动态控制分析",
                  pad=12,
                  fontsize=12,
                  color=ACCENT_BLUE,
                  loc='left')

    # 保存输出
    plt.savefig(OUTPUT_DIR / 'Premium_Visualization.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.show()
# ------------------------- 主执行流程 -------------------------
if __name__ == "__main__":
    # 问题2观测点优化
    problem2_points = find_optimal_observations("problem2", 12)

    # 问题3观测点优化
    problem3_points = find_optimal_observations("problem3", 11)

    # 矩阵分析方法验证
    problem2_svd_points = select_by_sensitivity("problem2", 12)
    problem3_svd_points = select_by_sensitivity("problem3", 11)

    # 结果可视化
    visualize_results(problem2_points, problem3_points)

    # 结果输出
    print("问题2最优观测时刻:", [format_time(t) for t in problem2_points])
    print("问题3最优观测时刻:", [format_time(t) for t in problem3_points])
