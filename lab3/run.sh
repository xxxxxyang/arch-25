#!/bin/bash
# filepath: /home/xyang/arch/lab2/evaluate_policy_assoc.sh

# 进入 gem5 所在目录
cd /home/xyang/arch/gem5-stable

# 定义路径
BINARY_PATH="/home/xyang/arch/lab2-benchmark/bin"
OUTPUT_DIR="/home/xyang/arch/lab3/output"
mkdir -p $OUTPUT_DIR

# 设置固定参数
CPU_TYPE="O3CPU"
CPU_CLOCK="2GHz"
L1D_SIZE="64kB"
L1I_SIZE="64kB"
L2_SIZE="2MB"
L1I_ASSOC=4
L2_ASSOC=8

# 可变参数：三种替换策略 + 三种相联度
REPLACEMENTS=("random" "nmru" "lip")
ASSOCS=(4 8 16)

# 定义需要运行的 benchmark 程序
BINARIES=("mm")

# 遍历组合
for REPL in "${REPLACEMENTS[@]}"; do
    for ASSOC in "${ASSOCS[@]}"; do
        for BINARY in "${BINARIES[@]}"; do

            # 构造输出文件夹
            OUT_PATH="${OUTPUT_DIR}/${BINARY}_${REPL}_assoc${ASSOC}"
            CONFIG_FILE="${OUT_PATH}/m5out/config.json"
            mkdir -p "$OUT_PATH"

            # 检查是否已经存在结果
            if [ -f "$CONFIG_FILE" ]; then
                echo "Skipping $BINARY with replacement=$REPL, L1D_assoc=$ASSOC (already exists)"
                continue
            fi

            echo "Running $BINARY with replacement=$REPL, L1D_assoc=$ASSOC"

            CMD="build/X86/gem5.debug configs/example/se.py \
                --cpu-type=${CPU_TYPE} \
                --cpu-clock=${CPU_CLOCK} \
                --cmd=${BINARY_PATH}/${BINARY} \
                --caches --l2cache \
                --l1d_size=${L1D_SIZE} --l1i_size=${L1I_SIZE} \
                --l1d_assoc=${ASSOC} --l1i_assoc=${L1I_ASSOC} \
                --l2_size=${L2_SIZE} --l2_assoc=${L2_ASSOC} \
                --replace=${REPL} \
                --param=system.cpu[0].issueWidth=8"

            eval $CMD
            mv m5out "$OUT_PATH"
        done
    done
done

echo "All cache replacement experiments completed."
