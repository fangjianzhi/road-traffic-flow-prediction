import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 读取表4数据
def load_data():
    times = []
    t_values = []
    flows = []
    for i in range(60):
        hour = 7 + i // 30
        minute = (i * 2) % 60
        times.append(f"{hour:02d}:{minute:02d}")
        t_values.append(i)
    flows = [20.1178, 21.6463, 31.0159, 40.5070, 41.4589, 40.3803, 45.0280, 23.5751,
             27.6601, 28.5191, 22.8609, 54.4147, 56.3573, 54.6330, 59.2022, 55.9851,
             37.2741, 43.2759, 44.3377, 46.9259, 74.7459, 69.1357, 74.9930, 75.0219,
             72.3299, 52.8119, 53.9451, 52.3878, 55.7438, 81.1195, 86.6484, 81.2547,
             83.2154, 81.3694, 53.7771, 59.5877, 58.5586, 54.5368, 80.8018, 71.8756,
             73.5100, 68.9525, 67.9241, 44.8616, 36.4950, 35.4181, 31.5647, 71.6705,
             68.6749, 65.5375, 58.2082, 60.4376, 17.2408, 12.3009, 10.9520, 12.9869,
             36.8780, 31.1231, 32.2821, 28.5905]
    df = pd.DataFrame({
        'time': times,
        't': t_values,
        'flow': flows
    })
    return df


# 支路1：分段线性函数
def branch1_flow(t, params):
    t_start, t_peak, t_end, max_flow = params
    t = np.asarray(t)
    flow = np.zeros_like(t, dtype=float)
    mask1 = (t >= t_start) & (t <= t_peak)
    flow[mask1] = max_flow * (t[mask1] - t_start) / (t_peak - t_start)
    mask2 = (t > t_peak) & (t <= t_end)
    flow[mask2] = max_flow
    mask3 = (t > t_end) & (t < t_end + (t_end - t_peak))
    flow[mask3] = max_flow * (1 - (t[mask3] - t_end) / (t_end - t_peak))
    return np.maximum(flow, 0)


# 支路2：分段线性函数
def branch2_flow(t, params):
    t_peak, max_flow = params
    t = np.asarray(t)
    flow = np.zeros_like(t, dtype=float)
    mask1 = (t <= 36)
    flow[mask1] = max_flow * t[mask1] / 36
    mask2 = (t > 36) & (t <= 54)
    flow[mask2] = max_flow
    mask3 = (t > 54) & (t <= 60)
    flow[mask3] = max_flow * (1 - (t[mask3] - 54) / 6)
    return np.maximum(flow, 0)


# 支路3：交通信号灯控制
def branch3_flow(t, params):
    first_green, flow_rate = params
    cycle_length = 9  # 18 minutes = 9 t units (2 min per t)
    green_duration = 5  # 10 minutes = 5 t units
    t = np.asarray(t)
    t_in_cycle = (t - first_green) % cycle_length
    flow = np.where((t_in_cycle >= 0) & (t_in_cycle < green_duration), flow_rate, 0)
    return np.maximum(flow, 0)


# 计算主路预测流量
def predict_main_flow(t, params_1, params_2, params_3):
    t = np.asarray(t)
    flow1_delayed = np.zeros_like(t, dtype=float)
    flow2_delayed = np.zeros_like(t, dtype=float)
    mask = t >= 1
    flow1_delayed[mask] = branch1_flow(t[mask] - 1, params_1)
    flow2_delayed[mask] = branch2_flow(t[mask] - 1, params_2)
    flow3 = branch3_flow(t, params_3)
    return flow1_delayed + flow2_delayed + flow3


# 优化目标函数
def objective_function(params, data):
    params_1 = params[0:4]
    params_2 = params[4:6]
    params_3 = params[6:8]
    t = data['t'].values
    actual_flow = data['flow'].values
    predicted_flow = predict_main_flow(t, params_1, params_2, params_3)
    mse = np.mean((predicted_flow - actual_flow) ** 2)

    # 更严格的惩罚项
    penalty = 0
    flow1 = branch1_flow(t, params_1)
    flow2 = branch2_flow(t, params_2)
    flow3 = branch3_flow(t, params_3)

    # 负流量惩罚
    if np.any(flow1 < 0):
        penalty += 10000 * np.sum(np.abs(flow1[flow1 < 0]))
    if np.any(flow2 < 0):
        penalty += 10000 * np.sum(np.abs(flow2[flow2 < 0]))
    if np.any(flow3 < 0):
        penalty += 10000 * np.sum(np.abs(flow3[flow3 < 0]))

    # 参数合理性惩罚
    t_start, t_peak, t_end, max_flow_1 = params_1
    if t_start >= t_peak:
        penalty += 10000 * (t_start - t_peak + 1)
    if t_peak >= t_end:
        penalty += 10000 * (t_peak - t_end + 1)

    return mse + penalty


# 主流程
def solve_problem4():
    data = load_data()

    # 参数边界 - 调整边界以获得更好的优化结果
    param_bounds = [
        (0, 15), (10, 30), (25, 45), (20, 80),  # 支路1参数
        (20, 50), (20, 70),  # 支路2参数
        (0, 9), (20, 70)  # 支路3参数
    ]

    # 初始参数 - 基于数据特征设置更好的初始值
    initial_params = [5, 15, 30, 50, 36, 40, 3, 45]

    # 第一阶段：差分进化全局优化 - 增加迭代次数和种群大小
    print("开始差分进化优化...")
    result_de = differential_evolution(
        objective_function,
        bounds=param_bounds,
        args=(data,),
        strategy='best1bin',
        maxiter=2000,  # 增加迭代次数
        popsize=50,  # 增加种群大小
        tol=1e-7,  # 更严格的收敛条件
        mutation=(0.5, 1.5),  # 调整变异范围
        recombination=0.8,  # 增加重组概率
        seed=42,
        disp=True,
        polish=False  # 不立即进行局部优化
    )

    # 第二阶段：局部优化 - 使用更强大的优化方法
    print("\n开始局部优化...")
    result_local = minimize(
        objective_function,
        result_de.x,
        args=(data,),
        bounds=param_bounds,
        method='SLSQP',  # 使用SLSQP方法
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    # 第三阶段：进一步优化 - 在最优解附近进行精细搜索
    print("\n开始精细优化...")
    final_params = minimize(
        objective_function,
        result_local.x,
        args=(data,),
        method='Nelder-Mead',  # 使用不需要梯度的Nelder-Mead方法
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
    ).x

    optimal_params = final_params
    params_1 = optimal_params[0:4]
    params_2 = optimal_params[4:6]
    params_3 = optimal_params[6:8]

    t = data['t'].values
    actual_flow = data['flow'].values
    predictions = predict_main_flow(t, params_1, params_2, params_3)
    rmse = np.sqrt(np.mean((predictions - actual_flow) ** 2))

    # 计算特定时刻的流量
    t_730 = 15
    t_830 = 45
    flow1_730 = branch1_flow(t_730, params_1)
    flow2_730 = branch2_flow(t_730, params_2)
    flow3_730 = branch3_flow(t_730, params_3)
    flow1_830 = branch1_flow(t_830, params_1)
    flow2_830 = branch2_flow(t_830, params_2)
    flow3_830 = branch3_flow(t_830, params_3)

    return data, params_1, params_2, params_3, predictions, rmse, flow1_730, flow2_730, flow3_730, flow1_830, flow2_830, flow3_830


# 可视化结果
def visualize_results(data, params_1, params_2, params_3, predictions, rmse, flow1_730, flow2_730, flow3_730, flow1_830,
                      flow2_830, flow3_830):
    t = data['t'].values
    actual_flows = data['flow'].values
    t_start, t_peak, t_end, max_flow_1 = params_1
    t_peak_2, max_flow_2 = params_2
    first_green, flow_rate = params_3

    # 计算各支路流量
    branch1_flows = branch1_flow(t, params_1)
    branch2_flows = branch2_flow(t, params_2)
    branch3_flows = branch3_flow(t, params_3)

    # 计算延迟后的支路流量
    flow1_delayed = np.zeros_like(t, dtype=float)
    flow2_delayed = np.zeros_like(t, dtype=float)
    mask = t >= 1
    flow1_delayed[mask] = branch1_flow(t[mask] - 1, params_1)
    flow2_delayed[mask] = branch2_flow(t[mask] - 1, params_2)

    # 第一幅图：主路和支路流量
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, actual_flows, 'bo-', label='主路实际车流量')
    plt.plot(t, predictions, 'r--', label=f'主路预测车流量 (RMSE={rmse:.4f})')
    plt.xlabel('时间t (相对于7:00的分钟数/2)')
    plt.ylabel('车流量')
    plt.title('问题4：主路车流量预测结果')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, branch1_flows, 'g-', label='支路1车流量')
    plt.plot(t, branch2_flows, 'm-', label='支路2车流量')
    plt.plot(t, branch3_flows, 'c-', label='支路3车流量')
    plt.axvline(x=t_start, color='g', linestyle='--', alpha=0.5, label='支路1转折点')
    plt.axvline(x=t_peak, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=t_end, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=t_peak_2, color='m', linestyle='--', alpha=0.5, label='支路2转折点')

    # 标记绿灯时段
    cycle_length = 9
    green_duration = 5
    for i in range(-1, 7):
        cycle_start = first_green + i * cycle_length
        if cycle_start >= 0:
            plt.axvspan(cycle_start, cycle_start + green_duration, color='green', alpha=0.1)

    plt.xlabel('时间t (相对于7:00的分钟数/2)')
    plt.ylabel('车流量')
    plt.title('问题4：各支路车流量')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Q-4_主路和支路流量_优化后.png', dpi=300, bbox_inches='tight')

    # 第二幅图：堆叠图
    plt.figure(figsize=(14, 8))
    plt.plot(t, actual_flows, 'bo-', label='主路实际车流量', alpha=0.5)
    plt.plot(t, predictions, 'r--', label=f'主路预测车流量 (RMSE={rmse:.4f})')

    # 标记绿灯时段
    for i in range(-1, 7):
        cycle_start = first_green + i * cycle_length
        if cycle_start >= 0:
            plt.axvspan(cycle_start, cycle_start + green_duration, color='green', alpha=0.1)

    # 绘制堆叠区域
    plt.fill_between(t, 0, flow1_delayed, alpha=0.3, label='支路1车流量(延迟后)')
    plt.fill_between(t, flow1_delayed, flow1_delayed + flow2_delayed, alpha=0.3, label='支路2车流量(延迟后)')
    plt.fill_between(t, flow1_delayed + flow2_delayed, flow1_delayed + flow2_delayed + branch3_flows, alpha=0.3,
                     label='支路3车流量')

    plt.xlabel('时间t (相对于7:00的分钟数/2)')
    plt.ylabel('车流量')
    plt.title('问题4：支路车流量叠加及主路车流量比较')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Q-4_支路车流量叠加_优化后.png', dpi=300, bbox_inches='tight')

    # 保存结果到文件
    with open('Q-4_result_优化后.txt', 'w', encoding='utf-8') as f:
        f.write("优化结果：\n")
        f.write(
            f"支路1参数 (t_start, t_peak, t_end, max_flow): ({t_start:.4f}, {t_peak:.4f}, {t_end:.4f}, {max_flow_1:.4f})\n")
        f.write(f"支路2参数 (t_peak, max_flow): ({t_peak_2:.4f}, {max_flow_2:.4f})\n")
        f.write(f"支路3参数 (first_green, flow_rate): ({first_green:.4f}, {flow_rate:.4f})\n")
        f.write(f"\nRMSE: {rmse:.6f}\n")
        f.write("\n7:30时刻各支路车流量：\n")
        f.write(f"支路1: {flow1_730:.2f}\n")
        f.write(f"支路2: {flow2_730:.2f}\n")
        f.write(f"支路3: {flow3_730:.2f}\n")
        f.write("\n8:30时刻各支路车流量：\n")
        f.write(f"支路1: {flow1_830:.2f}\n")
        f.write(f"支路2: {flow2_830:.2f}\n")
        f.write(f"支路3: {flow3_830:.2f}\n")
        f.write("\n函数表达式：\n")
        f.write(f"支路1:\n")
        f.write(f"  当 t < {t_start:.2f} 时: f1(t) = 0\n")
        f.write(
            f"  当 {t_start:.2f} <= t <= {t_peak:.2f} 时: f1(t) = {max_flow_1:.4f} * (t - {t_start:.2f}) / {t_peak - t_start:.2f}\n")
        f.write(f"  当 {t_peak:.2f} < t <= {t_end:.2f} 时: f1(t) = {max_flow_1:.4f}\n")
        f.write(
            f"  当 {t_end:.2f} < t < {t_end + (t_end - t_peak):.2f} 时: f1(t) = {max_flow_1:.4f} * (1 - (t - {t_end:.2f}) / {t_end - t_peak:.2f})\n")
        f.write(f"  当 t >= {t_end + (t_end - t_peak):.2f} 时: f1(t) = 0\n")
        f.write(f"\n支路2:\n")
        f.write(f"  当 t <= 36 时: f2(t) = {max_flow_2:.4f} * t / 36\n")
        f.write(f"  当 36 < t <= 54 时: f2(t) = {max_flow_2:.4f}\n")
        f.write(f"  当 54 < t <= 60 时: f2(t) = {max_flow_2:.4f} * (1 - (t - 54) / 6)\n")
        f.write(f"  当 t > 60 时: f2(t) = 0\n")
        f.write(f"\n支路3:\n")
        f.write(
            f"  当 t 在绿灯时段 (e.g., [{first_green:.2f}, {first_green + 5:.2f}), [{first_green + 9:.2f}, {first_green + 14:.2f}), ...) 时: f3(t) = {flow_rate:.4f}\n")
        f.write(f"  其他时间(红灯时段) 时: f3(t) = 0\n")

    # 打印结果
    print("\n最终优化结果：")
    print(f"支路1参数 (t_start, t_peak, t_end, max_flow): ({t_start:.4f}, {t_peak:.4f}, {t_end:.4f}, {max_flow_1:.4f})")
    print(f"支路2参数 (t_peak, max_flow): ({t_peak_2:.4f}, {max_flow_2:.4f})")
    print(f"支路3参数 (first_green, flow_rate): ({first_green:.4f}, {flow_rate:.4f})")
    print(f"RMSE: {rmse:.6f}")
    print("\n7:30时刻各支路车流量：")
    print(f"支路1: {flow1_730:.2f}")
    print(f"支路2: {flow2_730:.2f}")
    print(f"支路3: {flow3_730:.2f}")
    print("\n8:30时刻各支路车流量：")
    print(f"支路1: {flow1_830:.2f}")
    print(f"支路2: {flow2_830:.2f}")
    print(f"支路3: {flow3_830:.2f}")
    print("\n表4 问题4支路车流量数值")
    print("时刻    支路1    支路2    支路3")
    print(f"7:30    {flow1_730:.2f}    {flow2_730:.2f}    {flow3_730:.2f}")
    print(f"8:30    {flow1_830:.2f}    {flow2_830:.2f}    {flow3_830:.2f}")

    # 显示图形
    plt.show()


# 运行主函数
if __name__ == "__main__":
    data, params_1, params_2, params_3, predictions, rmse, flow1_730, flow2_730, flow3_730, flow1_830, flow2_830, flow3_830 = solve_problem4()
    visualize_results(data, params_1, params_2, params_3, predictions, rmse, flow1_730, flow2_730, flow3_730, flow1_830,
                      flow2_830, flow3_830)