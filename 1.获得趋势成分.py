# 导入必要的库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from statsmodels.nonparametric.smoothers_lowess import lowess  # 非参数平滑方法

# 设置中文字体（以 Windows 的 SimHei 为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel数据文件
data = pd.read_excel("data.xlsx")  # 从Excel文件加载数据
power = data["实际发电功率"].values  # 提取"实际发电功率"列的值转为numpy数组
x = np.arange(len(power))  # 创建与功率数据长度相同的索引数组作为x轴

# 使用LOWESS进行平滑处理（非参数局部回归）
# frac: 平滑窗口比例（0.3表示使用30%的数据点进行局部拟合）
# it: 执行3次稳健拟合迭代（消除异常值影响）
smoothed = lowess(power, x, frac=0.3, it=3)

# 提取并处理平滑结果
rt = smoothed[:, 1]  # 提取平滑后的趋势分量（第二列为拟合值）
data_d_rt = power / rt  # 计算原始功率与趋势分量的比值（分解周期分量）

# 创建绘图画布（宽20英寸，高4英寸）
plt.figure(figsize=(20, 4))

# 绘制原始功率曲线
plt.plot(power,
         label="功率",
         color='blue',
         linestyle='-',  # 实线
         linewidth=2)  # 线宽2磅

# 绘制趋势分量曲线
plt.plot(rt,
         label="趋势分量 $r^{(k)}$",  # LaTeX格式的数学符号
         color='green',
         linestyle='--',  # 虚线
         linewidth=2)

# 保存图形到文件
plt.savefig("2.获得趋势分量.png",
            bbox_inches="tight")  # 紧凑型边界框

# 显示图形
plt.legend(loc="upper left")  # 在左上角显示图例
plt.grid(True)  # 添加网格线
plt.show()