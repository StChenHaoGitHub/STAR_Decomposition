# 导入所需库
from statistics import median  # 提供中位数计算函数
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from statsmodels.nonparametric.smoothers_lowess import lowess  # LOWESS平滑算法

# 设置中文显示（适用于Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载与预处理
data = pd.read_excel("data.xlsx")  # 加载Excel数据集
power = data["实际发电功率"].values  # 提取功率数据为numpy数组
x = np.arange(len(power))  # 创建时间索引序列（0,1,2,...n）

# 趋势分量提取（LOWESS平滑）
smoothed = lowess(power, x, frac=0.3, it=3)  # 执行局部加权回归
tk = smoothed[:, 1]  # 提取趋势分量（第二列为拟合值）

# 去趋势处理（乘法分解模型）
data_d_tk = power / tk  # 原始信号/趋势分量 = 季节分量 × 残差分量

# 周期特征提取（假设每日96个15分钟间隔）
son_sequence = data_d_tk.copy().reshape(96, -1)  # 重塑为(96,天数)矩阵
sequence_len = son_sequence.shape[-1]  # 获取周期天数（列数）

# 季节分量计算（注意此处实际使用均值，建议修改为np.median）
column_medians = np.mean(son_sequence, axis=1)  # 计算每个时间点的均值
column_medians = column_medians[:, np.newaxis]  # 转换为列向量

# 季节分量重构
sk = np.tile(column_medians, (1, sequence_len)).T.reshape(-1)  # 延展为完整序列

# 残差分量计算
rk = data_d_tk / sk  # 残差 = 去趋势信号 / 季节分量

# 异常检测算法 --------------------------------------------------
# 计算残差中位数
rk_median = np.median(rk)  # 获取残差的中位数值

# 计算绝对离差（Median Absolute Deviation, MAD）
rou_abs = np.abs(rk - rk_median)  # 每个残差与中位数的绝对距离

# 计算标准化离差（类似Z-score标准化）
rou_k = rou_abs / np.sqrt(np.sum(rou_abs)/(len(rou_abs)-1))  # 自定义标准化方法

# 确定异常阈值（取95%分位数）
sorted_data = sorted(rou_k, reverse=True)  # 降序排列离差值
rou_c = sorted_data[int(0.05*len(sorted_data))-1]  # 取前5%作为异常阈值

# 生成异常指示向量
ak = rou_k.copy()  # 创建副本
ak[np.where(rou_k < rou_c)] = 1  # 正常点标记为1
rk[np.where(rou_k >= rou_c)] = 1  # 异常点残差设为1（需验证逻辑合理性）

# 可视化设置 ---------------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(20, 14))  # 创建5行子图布局

# 子图1：原始功率数据
axes[0].plot(power,
            label='原始功率数据',
            color='blue')
axes[0].set_ylabel('原始功率数据')
axes[0].legend()
axes[0].grid(True)

# 子图2：趋势分量
axes[1].plot(tk,
            label='趋势信息 (tk)',
            color='orange')
axes[1].set_ylabel('趋势成分')
axes[1].legend()
axes[1].grid(True)

# 子图3：季节分量
axes[2].plot(sk,
            label='季节信息分量 (sk)',
            color='green')
axes[2].set_ylabel('季节成分')
axes[2].legend()
axes[2].grid(True)

# 子图4：异常分量（需注意标记逻辑）
axes[3].plot(ak,
            label='异常分量 (ak)',
            color='red')
axes[3].set_ylabel('异常成分')
axes[3].legend()
axes[3].grid(True)

# 子图5：修正后的残差分量
axes[4].plot(rk,
            label='残差分量 (rk)',
            color='purple')
axes[4].set_ylabel('残差成分')
axes[4].legend()
axes[4].grid(True)

# 布局调整
plt.tight_layout()
plt.show()