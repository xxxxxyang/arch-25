import os
import matplotlib.pyplot as plt
import pandas as pd

# 定义新的输出目录路径
output_dir = "/home/xyang/arch/lab3/output"

# 定义需要提取的关键指标
keywords = {
    "ExecutionTime": [
        "simSeconds",
        "system.cpu.numCycles"
    ],
    "CPI": [
        "system.cpu.cpi"
    ],
    "L1DCache": [
        "system.cpu.dcache.overallMissRate::total",
        "system.cpu.dcache.WriteReq.missRate::total",
        "system.cpu.dcache.replacements",
        "system.cpu.dcache.writebacks::total",
        "system.cpu.dcache.overallAvgMissLatency::total"
    ],
    "L2Cache": [
        "system.l2.overallMissRate::total",
        "system.l2.replacements",
        "system.l2.overallAvgMissLatency::total"
    ]
}

# 结果存储
results = {}

# 遍历新的目录结构
for config in os.listdir(output_dir):
    config_path = os.path.join(output_dir, config, "m5out")
    if not os.path.isdir(config_path):
        continue

    results[config] = {}

    # 遍历每个 cfg 子目录
    stats_file = os.path.join(config_path, "stats.txt")

    # 检查 stats.txt 是否存在
    if not os.path.isfile(stats_file):
        print(f"Warning: {stats_file} not found!")
        continue

    # 初始化存储当前 cfg 的结果
    results[config] = {category: {} for category in keywords}

    # 读取 stats.txt 文件并提取关键指标
    with open(stats_file, "r") as f:
        for line in f:
            for category, keys in keywords.items():
                for key in keys:
                    if line.startswith(key):
                        # 提取关键字对应的值
                        value = line.split()[1]
                        results[config][category][key] = value


# print("Results:")
# for config, data in results.items():
#     print(f"Configuration: {config}")
#     for category, values in data.items():
#         print(f"  {category}:")
#         for key, value in values.items():
#             print(f"    {key}: {value}")


# 要提取的指标 key 映射
metrics = {
    "CPI": "system.cpu.cpi",
    "L1 Miss Rate": "system.cpu.dcache.overallMissRate::total",
    "L2 Miss Rate": "system.l2.overallMissRate::total",
    "L1 Avg Miss Latency": "system.cpu.dcache.overallAvgMissLatency::total",
    "L2 Avg Miss Latency": "system.l2.overallAvgMissLatency::total",
    "L1 Writebacks": "system.cpu.dcache.writebacks::total",
}

# 策略颜色映射
strategy_colors = {
    "lip": "#1f77b4",      # 蓝色
    "nmru": "#ff7f0e",     # 橙色
    "random": "#2ca02c",   # 绿色
}

# 提取策略名
def get_strategy(config_name):
    if "lip" in config_name:
        return "lip"
    elif "nmru" in config_name:
        return "nmru"
    elif "random" in config_name:
        return "random"
    else:
        return "unknown"


def plot_compare(results):
    # 构造 DataFrame
    data = []
    for config_name, stat_groups in results.items():
        row = {"Config": config_name}
        row["Strategy"] = get_strategy(config_name)
        for label, key in metrics.items():
            for group in stat_groups.values():
                if key in group:
                    row[label] = float(group[key])
                    break
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values(by="Config")

    # 画图
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    for i, (label, _) in enumerate(metrics.items()):
        ax = axs[i]
        bars = []
        for idx, row in df.iterrows():
            bars.append(ax.bar(
                row["Config"],
                row[label],
                color=strategy_colors[row["Strategy"]],
                label=row["Strategy"]
            ))
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Config"], rotation=45, ha='right')

    # 去重图例并统一显示
    handles = []
    labels_seen = set()
    for strat, color in strategy_colors.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=strat))

    fig.legend(handles=handles, loc="upper right", title="Replacement Strategy")
    plt.suptitle("Cache & Performance Comparison for mm.c", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/cache_comparison.png", dpi=300)


plot_compare(results)