import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from datetime import datetime, timedelta
import sys
from pathlib import Path

# ========== 系统配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示设置
plt.rcParams['axes.unicode_minus'] = False

# ========== 文件路径配置 ==========
REPO_ROOT = Path(__file__).resolve().parents[1]
file_path = REPO_ROOT / "data" / "competition_attachment.xlsx"
output_dir = REPO_ROOT / "outputs" / "generated"


# ========== 系统检查 ==========
def check_system():
    """系统兼容性检查"""
    if sys.platform.startswith('win'):
        print("检测到Windows系统，自动设置workers=1")
        return 1
    else:
        print("检测到Linux/MacOS系统，启用多核并行计算")
        return -1


# ========== 数据加载模块 ==========
def load_data():
    """加载并验证数据"""
    if not os.path.exists(file_path):
        print(f"错误：文件不存在于指定路径: {file_path}")
        print("请检查：1.路径正确性 2.文件名准确性 3.文件扩展名")
        sys.exit(1)

    try:
        # 读取数据
        data_table3 = pd.read_excel(file_path, sheet_name='表3 (Table 3)', engine='openpyxl')

        # 数据验证
        required_columns = ['时间 t (Time t)', '主路4的车流量 (Traffic flow on the Main road 4)']
        missing_cols = [col for col in required_columns if col not in data_table3.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}\n现有列: {list(data_table3.columns)}")

        return data_table3

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        sys.exit(1)


# ========== 交通流量模型模块 ==========
class TrafficModels:
    @staticmethod
    def traffic_light_status(t_min):
        """信号灯状态判断"""
        cycle = 18  # 完整周期(分钟)
        green_start = 6  # 第一个绿灯开始时间(7:06)
        pos_in_cycle = (t_min - green_start) % cycle
        return 'green' if 0 <= pos_in_cycle < 10 else 'red'

    @classmethod
    def branch3_model(cls, t, a, b, c, d):
        """支路3车流量模型"""
        t_min = t * 2
        if cls.traffic_light_status(t_min) == 'red':
            return 0
        if t_min < 16:
            return max(0, a + b * t_min)
        elif 16 <= t_min < 34:
            return c
        else:
            return max(0, c - d * (t_min - 34))

    @classmethod
    def branch2_model(cls, t, a, b, c, d):
        """支路2车流量模型"""
        t_min = t * 2
        if t_min <= 70:
            return max(0, a + b * t_min)
        elif 70 < t_min < 94:
            return c
        else:
            return max(0, c - d * (t_min - 94))

    @classmethod
    def branch1_model(cls, t, a, b, c, d, e, f, g, h, i):
        """支路1车流量模型"""
        t_min = t * 2
        if t_min < 10:
            return max(0, a + b * t_min)
        elif 10 <= t_min < 30:
            return max(0, c + d * t_min)
        elif 30 <= t_min < 50:
            return max(0, e - f * (t_min - 30))
        elif 50 <= t_min < 70:
            return g
        else:
            return max(0, h - i * (t_min - 70))


# ========== 优化计算模块 ==========
class Optimizer:
    def __init__(self, data):
        """初始化优化器"""
        self.time_t = data['时间 t (Time t)'].values
        self.main_road4 = data['主路4的车流量 (Traffic flow on the Main road 4)'].values
        self.workers = check_system()

        # 参数边界设置
        self.bounds = [
            # 支路1参数
            (0, 50), (0, 5),  # a, b
            (0, 50), (0, 5),  # c, d
            (0, 100), (0, 5),  # e, f
            (0, 100),  # g
            (0, 100), (0, 5),  # h, i

            # 支路2参数
            (0, 50), (0, 5),  # a, b
            (0, 100),  # c
            (0, 5),  # d

            # 支路3参数
            (0, 50), (0, 5),  # a, b
            (0, 100),  # c
            (0, 5)  # d
        ]

    def full_model(self, t, *params):
        """完整车流量计算模型"""
        b1_params = params[:9]
        b2_params = params[9:13]
        b3_params = params[13:]

        b1 = TrafficModels.branch1_model(t, *b1_params)
        b2 = TrafficModels.branch2_model(t, *b2_params)
        b3 = TrafficModels.branch3_model(t, *b3_params)

        # 考虑前一时段流量
        b1_prev = TrafficModels.branch1_model(t - 1, *b1_params) if t > 0 else 0
        b2_prev = TrafficModels.branch2_model(t - 1, *b2_params) if t > 0 else 0

        return b1_prev + b2_prev + b3

    def loss_function(self, params):
        """损失函数计算"""
        try:
            pred = np.array([self.full_model(t, *params) for t in self.time_t])
            return np.sum((pred - self.main_road4) ** 2)
        except Exception as e:
            print(f"参数计算错误: {str(e)}")
            return np.inf

    def optimize(self):
        """执行优化过程"""
        print("\n正在进行差分进化优化...")
        try:
            result = differential_evolution(
                self.loss_function,
                self.bounds,
                strategy='best1bin',
                maxiter=1000,
                popsize=50,
                tol=0.001,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=True,
                workers=self.workers,
                updating='deferred'
            )

            if not result.success:
                print("\n警告：优化未完全收敛！")
                print(f"终止原因: {result.message}")

            return result

        except Exception as e:
            print(f"\n优化过程发生严重错误: {str(e)}")
            sys.exit(1)


# ========== 结果处理模块 ==========
class ResultProcessor:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.time_points = [datetime.strptime('07:00', '%H:%M') + timedelta(minutes=2 * i) for i in range(len(data))]

        # 计算流量数据
        self.branch1, self.branch2, self.branch3, self.predicted, self.error = self.calculate_flows()

    def calculate_flows(self):
        """计算各支路流量"""
        branch1 = np.array([TrafficModels.branch1_model(t, *self.params[:9])
                            for t in self.data['时间 t (Time t)']])
        branch2 = np.array([TrafficModels.branch2_model(t, *self.params[9:13])
                            for t in self.data['时间 t (Time t)']])
        branch3 = np.array([TrafficModels.branch3_model(t, *self.params[13:])
                            for t in self.data['时间 t (Time t)']])

        # 流量时移处理
        branch1_shifted = np.roll(branch1, -1)
        branch2_shifted = np.roll(branch2, -1)
        branch1_shifted[-1] = 0
        branch2_shifted[-1] = 0

        predicted = branch1_shifted + branch2_shifted + branch3
        error = self.data['主路4的车流量 (Traffic flow on the Main road 4)'] - predicted
        return branch1, branch2, branch3, predicted, error

    def visualize(self):
        """生成基础可视化图表"""
        plt.figure(figsize=(15, 12))

        # 主图对比
        plt.subplot(3, 1, 1)
        plt.plot(self.time_points, self.data['主路4的车流量 (Traffic flow on the Main road 4)'],
                 'b-', label='观测值')
        plt.plot(self.time_points, self.predicted, 'r--',
                 label=f'预测值 (RMSE={np.sqrt(np.mean(self.error ** 2)):.2f})')
        plt.title('主路4车流量对比分析')
        plt.xlabel('时间')
        plt.ylabel('车流量')
        plt.legend()
        plt.grid(True)

        # 支路分解
        plt.subplot(3, 1, 2)
        plt.plot(self.time_points, self.branch1, 'g-', label='支路1')
        plt.plot(self.time_points, self.branch2, 'b-', label='支路2')
        plt.plot(self.time_points, self.branch3, 'r-', label='支路3')
        plt.title('支路车流量分解')
        plt.xlabel('时间')
        plt.ylabel('车流量')
        plt.legend()
        plt.grid(True)

        # 误差分析
        plt.subplot(3, 1, 3)
        plt.plot(self.time_points, self.error, 'k-', label='误差')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title('预测误差分布')
        plt.xlabel('时间')
        plt.ylabel('误差值')
        plt.grid(True)

        plt.tight_layout()
        self._save_plot('1_basic_analysis.png')
        plt.close()

    def generate_additional_plots(self):
        """生成新增图表"""
        self._plot_stacked_area()
        self._plot_combined_bar()
        self._plot_horizontal_bars()
        self._plot_error_distribution()

    def _plot_stacked_area(self):
        """堆叠面积图"""
        plt.figure(figsize=(12, 6))
        plt.stackplot(
            self.time_points,
            self.branch1,
            self.branch2,
            self.branch3,
            labels=['支路1', '支路2', '支路3'],
            colors=['#FFA07A', '#87CEEB', '#98FB98'],
            alpha=0.8
        )
        plt.title('支路车流量构成（堆叠面积图）')
        plt.xlabel('时间')
        plt.ylabel('车流量 (辆/2分钟)')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        self._save_plot('2_stacked_area.png')
        plt.close()

    def _plot_combined_bar(self):
        """组合条形图"""
        selected_indices = np.linspace(0, len(self.time_points) - 1, 8, dtype=int)
        selected_times = [self.time_points[i].strftime('%H:%M') for i in selected_indices]

        x = np.arange(len(selected_times))
        width = 0.25

        plt.figure(figsize=(14, 7))
        plt.bar(x - width, self.branch1[selected_indices], width, label='支路1', color='#FFA07A')
        plt.bar(x, self.branch2[selected_indices], width, label='支路2', color='#87CEEB')
        plt.bar(x + width, self.branch3[selected_indices], width, label='支路3', color='#98FB98')

        plt.title('分时段车流量对比（组合条形图）')
        plt.xlabel('时间点')
        plt.ylabel('车流量 (辆/2分钟)')
        plt.xticks(x, selected_times, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        self._save_plot('3_combined_bar.png')
        plt.close()

    def _plot_horizontal_bars(self):
        """水平条形图（误差分析）"""
        error_abs = np.abs(self.error)
        error_bins = [
            (error_abs < 5).sum(),
            ((error_abs >= 5) & (error_abs < 10)).sum(),
            (error_abs >= 10).sum()
        ]

        plt.figure(figsize=(10, 4))
        plt.barh(
            ['<5', '5-10', '≥10'],
            error_bins,
            color=['#66CDAA', '#4682B4', '#CD5C5C']
        )
        plt.title('预测误差分布（水平条形图）')
        plt.xlabel('误差出现次数')
        plt.xlim(0, len(self.error))
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        for i, v in enumerate(error_bins):
            plt.text(v + 1, i, str(v), color='black', va='center')
        plt.tight_layout()
        self._save_plot('4_error_distribution.png')
        plt.close()

    def _plot_error_distribution(self):
        """误差分布直方图"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.error, bins=15, color='#4B9CD3', edgecolor='black', alpha=0.7)
        plt.title('预测误差频率分布')
        plt.xlabel('误差值')
        plt.ylabel('出现次数')
        plt.grid(True, axis='y', alpha=0.5)
        self._save_plot('5_error_histogram.png')
        plt.close()

    def _save_plot(self, filename):
        """统一保存图表"""
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

    def save_results(self):
        """保存结果到文件"""
        results_df = pd.DataFrame({
            '时间': [t.strftime('%H:%M') for t in self.time_points],
            '观测值': self.data['主路4的车流量 (Traffic flow on the Main road 4)'],
            '预测值': self.predicted,
            '支路1': self.branch1,
            '支路2': self.branch2,
            '支路3': self.branch3,
            '误差': self.error
        })
        results_df.to_csv(os.path.join(output_dir, 'traffic_flow_results.csv'),
                          index=False, encoding='utf_8_sig')


# ========== 主程序 ==========
if __name__ == '__main__':
    # 数据加载
    data = load_data()

    # 优化计算
    optimizer = Optimizer(data)
    result = optimizer.optimize()

    # 结果处理
    processor = ResultProcessor(data, result.x)

    # 结果展示与保存
    processor.visualize()
    processor.generate_additional_plots()
    processor.save_results()

    # 关键指标输出
    print("\n优化结果摘要:")
    print(f"最终损失值: {result.fun:.2f}")
    print(f"迭代次数: {result.nit}")
    print(f"RMSE: {np.sqrt(np.mean(processor.error ** 2)):.2f}")

    # 关键时间点输出
    print("\n关键时间点车流量:")
    print("时间\t支路1\t支路2\t支路3")
    for t_min in [30, 90]:
        t = t_min // 2
        flow1 = TrafficModels.branch1_model(t, *result.x[:9])
        flow2 = TrafficModels.branch2_model(t, *result.x[9:13])
        flow3 = TrafficModels.branch3_model(t, *result.x[13:])
        print(f"7:{t_min:02d}\t{flow1:.1f}\t{flow2:.1f}\t{flow3:.1f}")

    print("\n所有图表和结果已保存至指定目录。")
