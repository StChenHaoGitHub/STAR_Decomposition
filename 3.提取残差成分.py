# 导入必要的库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from statsmodels.nonparametric.smoothers_lowess import lowess  # LOWESS平滑算法

# 设置中文显示（适用于Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

# 数据加载与预处理
data = pd.read_excel("data.xlsx")  # 从Excel文件加载数据集
power = data["实际发电功率"].values  # 提取功率列数据为numpy数组
x = np.arange(len(power))  # 创建时间索引序列（0,1,2,...n）

# 趋势分量提取（LOWESS平滑）
smoothed = lowess(power, x, frac=0.3, it=3)  # 执行局部加权回归
tk = smoothed[:, 1]  # 提取趋势分量（第二列为拟合值）

# 去趋势处理（乘法分解模型）
data_d_tk = power / tk  # 原始信号/趋势分量 = 季节分量 × 残差分量

# 周期特征提取（假设每日96个15分钟间隔）
son_sequence = data_d_tk.copy().reshape(96, -1)  # 重塑为(96,天数)矩阵
sequence_len = son_sequence.shape[-1]  # 获取周期天数（列数）

# 季节分量计算（行中位数）
column_medians = np.mean(son_sequence, axis=1)  # 计算每个时间点的平均值
column_medians = column_medians[:, np.newaxis]  # 转换为列向量便于矩阵运算

# 季节分量重构
sk = np.tile(column_medians, (1, sequence_len)).T.reshape(-1)  # 延展中位数序列
# 操作分解：
# 1. np.tile复制列向量为(96,天数)矩阵
# 2. .T转置为(天数,96)
# 3. reshape(-1)展平为原始长度

# 残差分量计算
rk = data_d_tk / sk  # 残差 = 去趋势信号 / 季节分量

# 可视化设置（创建4行1列的子图布局）
plt.figure(figsize=(20, 10))  # 设置画布尺寸（宽20英寸，高10英寸）

# ---------------------------
# 子图1：原始功率信号
# ---------------------------
plt.subplot(4, 1, 1)  # 定位到第1个子图
plt.plot(x, power,
         label="原始功率信号",  # 图例标签
         color='blue',  # 信号颜色
         linewidth=2)  # 线宽设置
plt.title("原始功率信号", fontsize=14)  # 子图标题
plt.xlabel("时间/索引", fontsize=12)  # X轴标签
plt.ylabel("功率值", fontsize=12)  # Y轴标签
plt.legend(loc='upper right', fontsize=10)  # 图例位置与字号
plt.grid(True, linestyle='--', alpha=0.6)  # 半透明虚线网格

# ---------------------------
# 子图2：趋势分量 (tk)
# ---------------------------
plt.subplot(4, 1, 2)  # 定位到第2个子图
plt.plot(x, tk,
         label="趋势分量 (tk)",  # 分量名称
         color='green',  # 趋势线颜色
         linewidth=2)
plt.title("趋势分量 (tk)", fontsize=14)
plt.xlabel("时间/索引", fontsize=12)
plt.ylabel("趋势值", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# ---------------------------
# 子图3：季节分量 (sk)
# ---------------------------
plt.subplot(4, 1, 3)  # 定位到第3个子图
plt.plot(x, sk,
         label="季节分量 (sk)",  # 周期特征
         color='red',  # 季节线颜色
         linewidth=2)
plt.title("季节分量 (sk)", fontsize=14)
plt.xlabel("时间/索引", fontsize=12)
plt.ylabel("季节值", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# ---------------------------
# 子图4：残差分量 (rk)
# ---------------------------
plt.subplot(4, 1, 4)  # 定位到第4个子图
plt.plot(x, rk,
         label="残差分量 (rk)",  # 剩余波动
         color='purple',  # 残差线颜色
         linewidth=2)
plt.title("残差分量 (rk)", fontsize=14)
plt.xlabel("时间/索引", fontsize=12)
plt.ylabel("残差值", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 全局布局调整
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 渲染显示图像