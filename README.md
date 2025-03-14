#  STAR 分解
---
在时序预测任务中，为了情绪化信号的各种成分，例如趋势信息季节信息等往往都需要对信号进行分解。目前熟知的分解方式有很多种，经验模态分解 EMD 变分模态分解 VMD ，还有 集合经验模态分解  EEMD，自适应噪声完备集合经验模态分解 CEEMDAN 等等，但这些和小波变换分解类似，一般都不会针对异常信号和极端事件进行分解。STAR 是一个比较冷门的分解方式，是我在Paper with Code 淘宝的时候发现的，他来源于 2022 年的 一篇叫 [AA-Forecast: Anomaly-Aware Forecast for Extreme Events](https://paperswithcode.com/paper/aa-forecast-anomaly-aware-forecast-for)。中提出的，个人认为这块是一个比较容易拼出一个创新点的内容。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a210ab3d4bc74ebe83aba4724d02dea9.png)
根据论文结构图，整个论文是分 STAR 分解，异常感知模型 Anomaly-Aware Model ，动态不确定性优化 Dynamic Uncertainty Optimization 三个部分，这里就先值关注第一个 STAR 分解的部分。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae4a59a6aa3b43d59a361ee72b206118.png)
#  1.STAR Decomposition 论文精读部分
---
对于这个分解方法的描述写在了论文的 3.1 部分，也不是很长，但我也是整体精读了下来，也从 Paper with Code 的官方链接上下载了代码进行对照，可以说如果其实有一些核心的地方是没有讲清楚的，所以导致看上去挺不错的东西，没啥人用，接下来就直接对 3.1 的部分精读一下，翻译的话是结合 DeepSeek 初步翻译，然后加上我自己的理解精调的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/20520d41420a4b3c82bcb03740bb2db5.png)
* *STAR decomposition is used as a strategy to not only extract the anomalies and sudden changes of data but also decompose the complex time series to its essential components.*
**STAR分解被用作一种策略，不仅用于提取数据的异常值和突变，而且还用于将复杂的时间序列分解为其基本组成部分。**

* *Unfortunately, widely popular decomposition method such as STL [14] does not extract anomalies.*
**不幸的是，广泛使用的 STL 方法不适用于去提取异常值**。
> STL 信号分解方法即基于局部加权回归的季节性趋势分解法（Seasonal and Trend decomposition using Loess）。

* *Although recent works such as STR [15] and RobustSTL [16] are designed to be robust to the extreme effect of anomalies in their decomposition, they are not used to explicitly extract anomalies from the residual component.*
**尽管最近的工作，如STR [15] 和 RobustSTL [16] 被设计为在其分解过程中对异常的极端效应具有鲁棒性，但它们并未被用来明确地从残差分量中提取异常。**

* *To alleviate these issues, we propose STAR decomposition that are decomposes the original time series $x^{(k)}$ in multiplicative manner to its seasonal $s^{(k)}$ ,trend $t^{(k)}$,anomalies $a^{(k)}$, and residual $r^{(k)}$ components:*
==**为了解决这个问题我们提出了 STAR 分解  ，将原始信号$x^{(k)}$ 分解为 季节成分$s^{(k)}$ ，趋势成分$t^{(k)}$，异常成分$a^{(k)}$，残差成分 $r^{(k)}$ 的相乘。**==
$$
x^{(k)} = s^{(k)}×t^{(k)}×a^{(k)}×r^{(k)}
$$
> 季节成分：是指时间序列数据中呈现出的周期性、重复性的模式，其周期通常是固定的，与季节、月份、星期、甚至一天中的不同时段等时间因素相关。

* *Such decomposition not only increases the dimensions of the original data but also supplies us withextracted anomalies.*
这种分解不仅增加了原始的数据维度，还为我们提供了提取出的异常值。

* As shown in Figure 1:
分解效果如图1所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1d65633e23af4929a5e4a6ec1caf593f.png)
* *we begin the decomposition by approximating the trend line $t^{(k)}$ with the locally weighted scatterplot smoothing (i.e. LOESS).* 
**我们通过局部加权散点点平滑来近似趋势线 $t^{(k)}$，开始进行分解。**

> LOESS 算法，即局部加权散点平滑法（Locally Weighted Scatterplot Smoothing），是一种用于数据拟合和曲线平滑的非参数统计方法。它通过对每个数据点附近的局部数据进行加权回归，来估计该点的平滑值。其优点是不需要对数据的分布形式做出先验假设，能够自适应地根据数据的局部特征进行拟合，对于具有复杂非线性关系的数据有较好的拟合效果，能有效去除数据中的噪声，展现出数据的潜在趋势。

* *Then, we divide the original data $x^{(k)}$ by the approximated trend line to derive the detrended time series.*
**接下来我门用原始信号 $x^{(k)}$除以趋势信号$t^{(k)}$，得到去除时间趋势之后的序列。**

* *We then partition the detrended time series into periods of cyclic sub-series where the cycle size is determined by the time interval of the dataset.*
**然后我们将去趋势的时间序列分割成周期性的子序列，其中周期大小由数据集的时间间隔决定。**

* *As an example, the cycle size for a monthly dataset would be 12. Then we obtain the seasonal component* 
**举一个例子，假设是一个月度数据，则周期大小将是12。**

* *Subsequently, the residual component  $r^{(k)}$  is derived by dividing the seasonal and trend segments from the original series.*
**之后，残留的残差成分 $r^{(k)}$ 是原始信号$x^{(k)}$ 除以趋势成分 $t^{(k)}$ 和季节成分 $s^{(k)}$ 得到的。**
* *Note that the anomaly component $a^{(k)}$ can be considered as the oddities of the dataset, which do not follow the extracted trend or seasonal components.*
**值得注意的是 $a^{(k)}$ 可以被视为数据集中的异常情况，这个异常情况不随季节和趋势的变化而变化。**

*  *Intuitively, anomalies are spread out through residual components, which also contain noise and other real-world effects.*
**直观上，异常值通常在分布残差成分中，这些异常值包含噪声和其他显示的现实世界的影响。**

---
残差成分 Residuals 和异常成分的区别 Anomalies 的区别。

|          | 残差 (Residuals)             | **异常 (Anomalies)**     |
| -------- | ---------------------------- | ------------------------ |
| **定义** | 模型无法解释的所有剩余波动   | 显著偏离正常模式的极端值 |
| **组成** | 噪声 + 未建模因素 + 潜在异常 | 残差中的特殊子集         |
| **性质** | 必然存在（模型不可能完美）   | 偶然出现（需要特别检测） |
| **处理** | 分析分布特征                 | 需要单独识别和调查       |
| **示例** | 日销售额波动 ±5%             | 某天销售额突然暴涨200%   |

- **残差** = 浪花（包含正常涟漪、小鱼跃出水面、偶尔的垃圾漂浮）
- **异常** = 突然出现的巨型漩涡（需要立即关注的特殊现象）
---
* *To distinguish the anomalies from residual components, statistical metrics such as mean and variance are not the appropriate measure as they are highly sensitive to the severity level of anomaly values.*
**为了区分异常值和残差分量，均值和方差等统计度量不是适当的衡量方法，因为他们对异常值的异常程度非常敏感。**
---

**假设你是班主任，要找出考试作弊的学生（异常检测）：**

- **班级平均分（mean）** = 所有学生分数的平均值
- **分数波动（variance）** = 学生们分数的分散程度

**现在有两个作弊情况：**

1. 张三：偷偷抄答案考了100分（极端异常值）
2. 李四：偶尔偷看考了75分（轻度异常值）

**如果只用均值和方差判断：**

- 张三的100分会大幅拉高班级平均分（比如从70→78）
- 方差也会急剧增大（分数范围从50-80变成50-100）

==通过均值和方差判则无法找出李四的异常==

---

* *As one expects, the severity of the anomalies can change the mean and variance values which are unwanted.*
**正如所设想的内样，异常的严重程度会以我们不期望的形式改变均值和方式。**

* *To resolve this issue, we leverage the median of the residuals, which is immune to the severity of the outliers in the residual components.*
**为了解决这个问题，我们利用残差的中位数，它对残差分量中的异常值的严重程度具有免疫力。**

* *Next,we define robustness score $\rho_t^{(k)}$ for each observation at time $t$ as：*
**接下来我们为在各个 $t$ 时刻的每一个观察值定义了鲁棒性因子 $\rho_t^{(k)}$ 公式如下：**
$$
\rho_t^{(k)} = \frac{\left| r_t^{(k)} - \dot{r}^{(k)} \right|}{\sqrt{\frac{\sum_{t = 1}^{T} \left| r_t^{(k)} - \dot{r}^{(k)} \right|}{T - 1}}}
$$
* *where $\rho_t^{(k)}$ stand for the strength of the anomalies,  $r_t^{(k)}$ is the residual at time step $t$ and $\dot{r}^{(k)}$ is the median of the residuals.* 
**其中 $\rho_t^{(k)}$ 代表异常强度，$r_t^{(k)}$ 时间步  $t$ 时刻的残差值，$\dot{r}^{(k)}$ 是残差成分的中位数。**


* *Note that the larger $\rho_t$ indicates that a drastic change has occurred in the trend and seasonal components. We then extract the anomalies from residuals as below:*
  **$\rho_t$  较大表明趋势和季节性成分发生了剧烈的变化，我们从残差中提取异常如下。**
$$
\mathbf{a}_t^{(k)} = 
\begin{cases}
1, & \rho_t^{(k)} < \rho_c^{(k)} \\
r_t^{(k)}, & \rho_t^{(k)} \geq \rho_c^{(k)}
\end{cases}
$$

*  *where $\rho_c^{(k)}$ is the constant threshold given by the value of a robustness score ranked in the $p-$value 0.05 while the values of elements in $\rho^{(k)}$ are ranked in descending order from large to small.*
**其中 $\rho_c^{(k)}$ 是在降序排列的 $\rho^{(k)}$ 序列中分位值为 0.05 的时候确定的。**

> 假设有 $\rho^{(k)}$ 有100个数，阈值 $\rho_c^{(k)}$ 就是第5（100×0.05）大的数字

* *Notably, when the value of the anomaly component ($a^{(k)}$)  deviates further from the value 1, it indicates an abrupt change in the trend and the seasonal component (no sign of anomalies).*
**这里要明确的是，当异常值$a^{(k)}$ 的值距离 1 很远时候，它表面确实和季节性成分发生了突然的变化，但不一定是异常。**

*  *On the contrary, when both anomaly and residual values are equal 1 ($r_t^{(k)}$ = 1 and  $a_t^{(k)}$ = 1)， it indicates that the observed signal at time $t$* *explicitly follows the trend and seasonal component.* 
**可以对比来看的是，当异常值和残差值都等于1时，表明在时间 $t$ 观察到的信号明确地遵循了趋势和季节成分。**

* *Note that such important information might not be automatically inferred when additive decomposition methods are being used. This due to the fact that the values of residual components can differ from one dataset to another which requires manual effort in their detection.*
**当使用加法分解方法时，这样的重要信息可能不会被自动推断出来，这是因为残差分量的值可能因数据集而异，因此需要手动筛选出异常。**（个人补充：加法分解一般只能直接通过幅度大小判定异常，但由于各个数据集的数据情况不同，无法直接通过对幅度加上阈值进行直接筛选）

*  *A sample result of anomaly decomposition is shown in the left-most part of Figure 1, where the observed time series data is decomposed into tis seasonal, trend, anomalies, and residual* .
**异常分解的一个示例结果显示在图最左侧的部分，其中观察到的时间序列数据被分别分解为季节性、趋势、异常和残差分量。**（还是开头内个图不放来他又说了一遍）

* *Each of these components holds essential information about the characteristics of the time and will be leveraged to train the forecast model.*
**这些成分中的每一个都包含了关于时间序列特征的重要信息，并将被我们使用来预测模型。**
* *To this end, we concatenate the derived decomposed vector of time series with the input, which includes the observed time series and its labeled extreme event.*
最终，我们将这些时间序列进行拼接作为一个输入模型的训练的特征样本，这个样本包含观测到的时间序列及标记的极端事件还有我们分解出的各个成分。


读完之后其实我是带着疑惑的，就是实际上 $x^{(k)} = s^{(k)}×t^{(k)}×a^{(k)}×r^{(k)}$ 在这个乘法项的分解里 $a^{(k)}$ 这一项是怎么乘进去的其实是没有清晰给出的，如果不看代码我觉得但从这段的写作上来看，有故意隐藏核心细节的嫌疑。好在代码虽然没有注释但我也复现出来了。

#  2.STAR 代码复现 手搓版
---

为了保证复现结果的可验证性，我这里用了一个阿里的开源功率数据集。
[https://tianchi.aliyun.com/dataset/159885](https://tianchi.aliyun.com/dataset/159885)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9ba24fda71c249fc95dde40e828e152d.png)
下载之后数据如下一个非常标准的风电数据集，之后我们就分解和使用实际发电功率这一项。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0bb52c141df94d61bd905fa0f7822ff8.png)


##  2.1 数据读取
 ---
![请添加图片描述](https://i-blog.csdnimg.cn/direct/f40817647e2d4f40ad865adc088df935.png)
```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("data.xlsx")
print(data.head())

power = data["实际发电功率"].values

plt.plot(power)
plt.show()
```

##  2.2 提取趋势成分
---
![请添加图片描述](https://i-blog.csdnimg.cn/direct/5c94b82e10a54fd997acbef4e526ea00.png)
```python
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
```

##  2.3 提取季节成分
---
在季节分量这里，我们选择的周期的以天为周期，由于数据本身颗粒度为15分钟，==所以就是96个一循环==。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe35231c3b904ffba29e0a3a59aaea22.png)
```python
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
```
## 2.4 提取残差成分
---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/627d86f45d904f01a1576d415c0bf3f1.png)

```python
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
column_medians = np.median(son_sequence, axis=1)  # 计算每个时间点的中位数
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
```

##  2.5 提取异常成分
---
这里要说一下的是，在原文中没有给出的残差成分和提取异常成分相乘的关系，在我阅读了他的代码之后，其实是通过赋1值实现的，也就是说异常成分就是残差成分中鲁棒因子大于等于阈值的点组成的，其他地方都是1，然后残差成分中的鲁棒因子小于阈值的点给变成了1，这样的话，经过这一步之后，得到的异常成分和残差成分相乘，其实就是上一步的残差成分，这部分关键逻辑在原文中被掩盖了。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d46860d196d427e8327024f0a2b5b4b.png)


```python
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
```
#  结束
---
这篇内容写的也是比较累的，确实难度也不小，而且实用性也存疑，但是我要是用不了给师弟拿去整个花活凑个不实用的创新点也是可以的。这个就先这样吧。
