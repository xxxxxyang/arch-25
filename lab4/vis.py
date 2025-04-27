import matplotlib.pyplot as plt
import numpy as np
import os

# 数据
functions = ['daxpy', 'daxpy_unroll', 'daxsbxpxy', 'daxsbxpxy_unroll', 'stencil', 'stencil_unroll']
cpi_no_fu_o1 = [1.777764, 2.004756, 2.09647, 2.26635, 1.962359, 1.886007]
cpi_with_fu_o1 = [1.777739, 2.004756, 2.013156, 2.186898, 1.962359, 1.859714]
cpi_no_fu_o3 = [1.823947, 2.033501, 2.062701, 1.552187, 2.239337, 3.414568]
cpi_with_fu_o3 = [1.823947, 1.995404, 2.062701, 1.539059, 2.208287, 3.201089]

# 条形图参数
x = np.arange(len(functions))  # 函数索引
width = 0.2  # 条形宽度

# 绘制条形图
plt.bar(x - 1.5 * width, cpi_no_fu_o1, width, label='no FU/O1', color='#00BFFF')
plt.bar(x - 0.5 * width, cpi_with_fu_o1, width, label='FU/O1', color='#FFD700')
plt.bar(x + 0.5 * width, cpi_no_fu_o3, width, label='no FU/O3', color='#228B22')
plt.bar(x + 1.5 * width, cpi_with_fu_o3, width, label='FU/O3', color='#FF3030')

# 添加标签和标题
plt.xlabel('functions')
plt.ylabel('CPI')
plt.title('CPI Comparison')
plt.xticks(x, functions, rotation=45)
plt.legend()

# 显示图表
plt.tight_layout()
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'fu_cpi_comparison.png')
plt.savefig(img_path)