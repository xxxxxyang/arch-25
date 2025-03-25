import os
import matplotlib.pyplot as plt

# 定义输出目录路径
output_dir = "/home/xyang/arch/lab2/output"

# 定义需要提取的关键指标
keywords = {
    "ISA": [
        "hostSeconds",
        "system.cpu.numInsts",
        "system.cpu.numOps",
        "system.cpu.numBranches",
        "system.cpu.numLoadInsts",
        "system.cpu.numStoreInsts",
        "system.cpu.commit.integer",
        "system.cpu.commit.floating"
    ],
    "Microarchitecture": [
        "system.cpu.ipc",
        "system.cpu.cpi",
        "system.cpu.numCycles",
        "system.cpu.branchPred.condIncorrect",
        "system.cpu.branchPred.condPredicted",  # 分支预测
        "system.cpu.icache.overallMissRate::total",
        "system.cpu.dcache.overallMissRate::total",
        "system.l2.overallMissRate::total", # cache 命中率
        "system.cpu.rename.ROBFullEvents"
    ]
}

# 定义结果存储
results = {}

# 遍历 1 到 7 的配置目录
for config in range(1, 8):
    config_path = os.path.join(output_dir, str(config))
    results[config] = {}

    # 遍历每个 benchmark 子目录
    for benchmark in os.listdir(config_path):
        benchmark_path = os.path.join(config_path, benchmark)
        stats_file = os.path.join(benchmark_path, "stats.txt")

        # 检查 stats.txt 是否存在
        if not os.path.isfile(stats_file):
            print(f"Warning: {stats_file} not found!")
            continue

        # 初始化存储当前 benchmark 的结果
        results[config][benchmark] = {category: {} for category in keywords}

        # 读取 stats.txt 文件并提取关键指标
        with open(stats_file, "r") as f:
            for line in f:
                for category, keys in keywords.items():
                    for key in keys:
                        if line.startswith(key):
                            # 提取关键字对应的值
                            value = line.split()[1]
                            results[config][benchmark][category][key] = value

# # 打印提取结果
# for config, benchmarks in results.items():
#     print(f"Configuration {config}:")
#     for benchmark, categories in benchmarks.items():
#         print(f"  Benchmark {benchmark}:")
#         for category, data in categories.items():
#             print(f"    {category}:")
#             for key, value in data.items():
#                 print(f"      {key}: {value}")
#     print()



# 定义要比较的指标和对应的 y 轴标签
metrics = [
    "system.cpu.cpi",
    "hostSeconds",
    "system.cpu.numInsts",
    "system.l2.overallMissRate::total",
    "system.cpu.icache.overallMissRate::total",
    "system.cpu.dcache.overallMissRate::total",
    "branchPred" # 分支预测准确率
]
ylabels = [
    "CPI (Cycles Per Instruction)",
    "Host Seconds",
    "Number of Instructions",
    "L2 Cache Miss Rate",
    "I-Cache Miss Rate",
    "D-Cache Miss Rate",
    "Branch Prediction Accuracy"

]

# 绘制同一benchmark下不同config的性能比较图，分别以 metrics 作为指标绘制子图
def plot_compare(results):
    for _, data in results.items():
        benchmarks = list(data.keys())
        break

    for benchmark in benchmarks:
        x = list(range(1, 8))
        fig, axs = plt.subplots(4, 2, figsize=(16, 16))  # 调整子图布局
        fig.suptitle(f"Performance Comparison of {benchmark}")
        plt.subplots_adjust(hspace=0.5)

        for i, metric in enumerate(metrics):
            ax = axs[i // 2][i % 2]
            ax.set_title(f"{ylabels[i]} of {benchmark}")
            ax.set_xlabel("Configuration")
            ax.set_ylabel(ylabels[i])

            y = []
            for config, data in results.items():
                if metric == "branchPred":
                    # 计算分支预测正确率
                    cond_predicted = float(data[benchmark]["Microarchitecture"].get("system.cpu.branchPred.condPredicted", 0))
                    cond_incorrect = float(data[benchmark]["Microarchitecture"].get("system.cpu.branchPred.condIncorrect", 0))
                    if cond_predicted > 0:
                        accuracy = (cond_predicted - cond_incorrect) / cond_predicted
                    else:
                        accuracy = 0.0
                    y.append(accuracy)
                elif metric not in data[benchmark]["Microarchitecture"]:
                    y.append(float(data[benchmark]["ISA"].get(metric, 0)))
                else:
                    y.append(float(data[benchmark]["Microarchitecture"].get(metric, 0)))
            ax.bar(x, y, label=benchmark)
            ax.set_xticks(x)

            ax.legend()
        # plt.show()
        plt.savefig(f"/home/xyang/arch/lab2/image/{benchmark}.png")
        

plot_compare(results)