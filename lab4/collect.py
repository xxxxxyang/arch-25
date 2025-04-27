import re
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_and_renumber(file_path):
    # 用于存储提取的值
    cpi_values = []
    numInsts_values = []
    simseconds_values = []

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 匹配 system.cpu_cluster.cpus.cpi 和 system.cpu_cluster.cpus.numInsts
            cpi_match = re.match(r'system\.cpu_cluster\.cpus\.cpi\s+([\d\.]+)', line)
            numInsts_match = re.match(r'system\.cpu_cluster\.cpus\.numInsts\s+([\d\.]+)', line)
            simseconds_match = re.match(r'simSeconds\s+([\d\.]+)', line)

            if cpi_match:
                cpi_values.append(float(cpi_match.group(1)))
            if numInsts_match:
                numInsts_values.append(int(numInsts_match.group(1)))
            if simseconds_match:
                simseconds_values.append(float(simseconds_match.group(1)))

    # 舍弃第一和最后一个值
    cpi_values = cpi_values[1:-1]
    numInsts_values = numInsts_values[1:-1]
    simseconds_values = simseconds_values[1:-1]

    # 重新编号并打印结果
    print("CPI Values:")
    for i, value in enumerate(cpi_values, start=1):
        print(f"CPI_{i}: {value}")

    print("\nNumInsts Values:")
    for i, value in enumerate(numInsts_values, start=1):
        print(f"NumInsts_{i}: {value}")

    print("\nSimSeconds Values:")
    for i, value in enumerate(simseconds_values, start=1):
        print(f"SimSeconds_{i}: {value}")

    return cpi_values, numInsts_values, simseconds_values

def plot_values(values, name='CPI'):
    # 创建索引
    indices = np.arange(1, len(values) + 1)
    roll = [indices[i]+0.2 for i in range(len(indices)) if i % 2 == 0]
    unroll = [indices[i]-0.2 for i in range(len(indices)) if i % 2 != 0]
    # update indices
    indices = [roll[i//2] if i % 2 == 0 else unroll[i//2] for i in range(len(values))]

    # 设置颜色：1,3,5 用一种颜色，2,4,6 用另一种颜色
    color_roll = '#1f77b4'  # 蓝色
    color_unroll = '#ff7f0e'  # 橙色
    # colors = ['#1f77b4' if i % 2 != 0 else '#ff7f0e' for i in indices]

    # 创建图形
    plt.figure(figsize=(10, 6))
    bar_width = 0.5

    # 绘制 CPI 的条形图
    plt.bar(roll, [values[i] for i in range(len(values)) if i % 2 == 0], width=bar_width, color=color_roll, label='Roll')
    plt.bar(unroll, [values[i] for i in range(len(values)) if i % 2 != 0], width=bar_width, color=color_unroll, label='Unroll')

    # # 绘制 NumInsts 的条形图
    # plt.bar(indices + (bar_width / 2 + offset), numInsts_values, width=bar_width, color=colors, alpha=0.6, label='NumInsts')

    # 添加标签和标题
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(name + 'Unroll vs Roll')
    tables = ['daxpy', 'daxpy_unroll','daxsbxpxy', 'daxsbxpxy_unroll','stencil', 'stencil_unroll']
    plt.xticks(indices, tables, rotation=45)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    # 图片保存地址
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', name + '.png')
    plt.savefig(img_path)

if __name__ == "__main__":
    # 文件路径
    # file_path = '/home/xyang/arch/gem5-stable/m5out/stats.txt'
    file_path = '/home/xyang/arch/lab4/log/stats_O3.txt'
    
    # 提取数据
    cpi_values, numInsts_values, simseconds_values = extract_and_renumber(file_path)

    # 绘制图形
    plot_values(cpi_values, name='CPI')
    plot_values(numInsts_values, name='NumInsts')
    plot_values(simseconds_values, name='SimSeconds')