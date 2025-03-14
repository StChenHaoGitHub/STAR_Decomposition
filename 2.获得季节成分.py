# 导入所需库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from statsmodels.nonparametric.smoothers_lowess import lowess  # LOWESS平滑算法

# 设置中文字体显示（针对Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

# 数据读取与预处理
data = pd.read_excel("data.xlsx")  # 从Excel文件加载数据集
power = data["实际发电功率"].values  # 将功率列转换为numpy数组
x = np.arange(len(power))  # 创建与数据长度相同的索引序列

# LOWESS趋势提取
smoothed = lowess(power, x, frac=0.3, it=3)  # 执行局部加权回归
rt = smoothed[:, 1]  # 提取趋势分量（第二列为拟合值）

# 去趋势处理
data_d_rt = power / rt  # 通过除法去除趋势（适用于乘法分解模型）

# 周期特征提取
son_sequence = data_d_rt.copy().reshape(96, -1)  # 按日周期(15分钟间隔)重塑数据为96行
sequence_len = son_sequence.shape[-1]  # 获取每个周期的天数（列数）

# 中位数计算（注意：此处实际使用均值计算，建议修改为np.median）
column_medians = np.mean(son_sequence, axis=1)  # 计算每个时间点的均值（应使用中位数）
column_medians = column_medians[:, np.newaxis]  # 添加新轴便于矩阵运算

# 季节分量重构
st = np.tile(column_medians, (1, sequence_len)).T.reshape(-1)  # 将均值延展为完整序列
# 操作解析：
# 1. np.tile进行矩阵复制 (1列→sequence_len列)
# 2. .T转置矩阵适配后续操作
# 3. reshape(-1)展平为一维季节分量

# 可视化设置
plt.figure(figsize=(20, 4))  # 创建20英寸宽、4英寸高的画布
plt.plot(st,  # 绘制季节分量
         label="中位数延展结果",  # 图例标签
         color='blue',  # 线条颜色
         linewidth=2)  # 线宽设置

# 图表美化
plt.title("中位数延展结果可视化", fontsize=16)  # 设置标题及字号
plt.xlabel("时间/索引", fontsize=14)  # X轴标签设置
plt.ylabel("中位数延展值", fontsize=14)  # Y轴标签设置（实际为均值）
plt.legend(loc='upper right', fontsize=12)  # 右上角图例
plt.grid(True, linestyle='--', alpha=0.6)  # 半透明虚线网格
plt.tight_layout()  # 自动调整元素间距
plt.show()  # 显示图形