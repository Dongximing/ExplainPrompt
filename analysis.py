import numpy as np
from scipy.stats import spearmanr

# 假设 data 是你的观测数据，已正确格式化（每列是一个变量）
# 这里我们直接用转置的示例数据
data = np.array([
    [1, 2, 3, 4, 5],     # Variable 1
    [2, 3, 2, 1, 0],     # Variable 2
    [3.5, 2.5, 0.5, 1.5, 3.0]  # Variable 3
]).T

# 计算斯皮尔曼相关系数和 p 值
correlation, p_value = spearmanr(data)

# 打印相关系数矩阵
print("Spearman correlation coefficient matrix:\n", correlation)

# 计算整个矩阵的平均值
overall_mean = np.mean(correlation)
print("Overall mean of the correlation matrix:", overall_mean)

# 计算非对角线元素的平均值
# 我们可以通过创建一个不包括对角线的布尔掩码来实现
mask = np.ones(correlation.shape, dtype=bool)
np.fill_diagonal(mask, False)
non_diagonal_mean = np.mean(correlation[mask])
print("Mean of non-diagonal elements:", non_diagonal_mean)

