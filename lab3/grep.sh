#!/bin/bash

# 定义输入文件和输出文件
INPUT_FILE="/home/xyang/arch/gem5-stable/m5out/cache_trace.txt"  # 替换为你的文件名
OUTPUT_FILE="state_lines.txt"
OUTPUT="cache_trace.txt"

# 删除含有 "icache" 的行，并将含有 "state" 的行输出到另一个文件
grep -v "icache" "$INPUT_FILE" > "$OUTPUT"
grep -v "icache" "$INPUT_FILE" | tee >(grep "state" > "$OUTPUT_FILE") > /dev/null