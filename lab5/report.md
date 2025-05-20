# CUDA 矩阵乘法优化

## 代码实现

```cpp
__global__ void Matmul3(float *A,float *B,float *C,unsigned N){// A,B with padding
    __shared__ float B_shared[U][S];    // B block 转置共享内存
    unsigned brow=blockIdx.y, bcol=blockIdx.x;
    unsigned tid=threadIdx.y;
    // 坐标
    unsigned row=brow*T + tid;
    unsigned basecol=bcol*U;
    // 寄存器分配
    float A_reg[S] = {0}, C_reg[U] = {0};
    unsigned kk, u, s;
    for (kk = 0; kk < N; kk += S) {
        // 将 B 读入共享内存
        s = tid / U;  // 行索引
        u = tid % U;  // 列索引
        B_shared[u][s] = B[(kk + s) * N + basecol + u];
        __syncthreads();  // 所有线程等共享内存加载完
        // 将 A 读入寄存器
        #pragma unroll
        for (s = 0; s < S; s++) {
            A_reg[s] = A[row * N + kk + s];
        }
        // 计算 C 的值
        #pragma unroll
        for (u = 0; u < U; u++) {
            #pragma unroll
            for (s = 0; s < S; s++) {
                C_reg[u] += A_reg[s] * B_shared[u][s];
            }
        }
        __syncthreads();
    }
    // 写回 C
    #pragma unroll
    for (u = 0; u < U; u++) {
        unsigned global_col = basecol + u;
        C[row * N + global_col] = C_reg[u];
    }
}
```

关键点解析：
- **内核布局**：
- **线程坐标计算**：
- **同步**：
- **转置共享内存**：


## Profile 结果

```bash
Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
-------  --------------  ----------  --------------  --------------  --------------  ------------------------
   51.6       992778000          11      90252545.5         1873821       125631219  Matmul1                                                                         
   33.9       652039251          11      59276295.5         1369597        65555717  Matmul2                                                                         
   14.5       278879506          11      25352682.4          891263        27921068  Matmul3                                                                         
```

**不同 `T`, `U` 和 `S` 参数结果分析：**

| T   | U   | time1 (Matmul1) | time2 (Matmul2) | time3 (Matmul3) | 分析总结                                |
| --- | --- | --------------- | --------------- | --------------- | ----------------------------------- |
| 64  | 16  | 100.9           | 69.4            | 90.2            | 寄存器还没显著优势，可能是线程数太少，寄存器加速不明显          |
| 64  | 32  | 101.6           | 67.7            | 55.6            | 提升明显：寄存器输出 $U=32$ 降低访存压力，提高并行效率     |
| 64  | 64  | 102.1           | 69.0            | 38.7            | 继续变快，寄存器发挥更大作用，每线程计算更多              |
| 128 | 64  | 101.9           | 69.5            | 37.2            | 增加线程数提升并行度，但 shared memory/寄存器使用也上升 |
| 128 | 128 | 101.8           | 68.4            | 29.3            | 寄存器利用率接近饱和（每线程 128 输出），性能最好之一       |
| 256 | 128 | 101.3           | 69.0            | 34.2            | 太多线程导致 block 占用资源多，调度下降，性能反而下降      |
| 256 | 256 | 101.8           | 68.9            | 39.0            | 寄存器/共享内存资源耗尽，block 数减少，调度压力上升，性能下降  |

在 T=64 时，相比 Matmul1 和 Matmul2，Matmul3 的线程数更少，这是增大 U 可以使每个线程计算更多的元素，合并访存，加快计算速度。

## 问题回答

### 对存储器的访问次数

考虑每个线程分别对 global memory、shared memory 和寄存器的访问次数。

- **Matmul1**

每个线程计算 `C[row * N + col]` 的一个元素

```cpp
for (k = 0; k < N; k++) {
    sum += A[row * N + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

每线程访存次数：
* 每轮访问：
  * A：`A[row * N + k]`  1 次 global memory
  * B：`B[k * N + col]`  1 次 global memory
* 共循环 `N` 次 → 共 `2N` 次 global memory 读取
* 写回 C：1 次
* **总 global memory 访问：`2N + 1`**
Shared memory：无
Register：仅 2N+2 个（`sum`）考虑到读写均算访问


- **Matmul2**

```cpp
for (kk = 0; kk < N; kk += Blocksize) {
    Asub[ty][tx] = A[row * N + (kk + tx)];
    Bsub[ty][tx] = B[(kk + ty) * N + col];
    __syncthreads();
    for (k = 0; k < Blocksize; k++) {
        sum += Asub[ty][k] * Bsub[k][tx];
    }
    __syncthreads();
}
C[row * N + col] = sum;
```

每线程访存次数：
* 外层循环：`N / Blocksize` 次
* 每轮：
  * 加载 A\[row \* N + (kk + tx)]：1 次 global memory
  * 加载 B\[(kk + ty) \* N + col]：1 次 global memory
* 总读取：
  * `2 * (N / Blocksize)` 次 global memory
* 写回 C：1 次
* **总 global memory 访问：`2(N / Blocksize) + 1`**
Shared memory：
* 每个外层循环：
  * 每线程从共享内存读取 `Blocksize` 个 Asub 和 Bsub
* 总访问次数：
  * `2 × Blocksize × (N / Blocksize) = 2N`（共享内存读）
Register：
* `sum` 2N+2 寄存器


- **Matmul3：寄存器 + 转置共享内存**

```cpp
float A_reg[S] = {0}, C_reg[U] = {0};

for (kk = 0; kk < N; kk += S) {
    // Load B_shared[u][s]
    B_shared[u][s] = B[(kk + s) * N + basecol + u];
    __syncthreads();
    // Load A_reg[s]
    for (s = 0; s < S; s++) {
        A_reg[s] = A[row * N + kk + s];
    }
    // Compute: C_reg[u] += A_reg[s] * B_shared[u][s]
    ...
}
```

每线程访存次数

global memory：
外层循环：`N / S` 次：
* 每轮外层循环：
  * 读 A 到寄存器：
    * `S` 次 global memory
  * 读 B 到共享内存：
    * 每线程参与加载 1 次: `1` 次 global memory
* 写回 C：
  * 每线程写 `U` 个元素: `U` 次 global memory
总 global memory：
$$
\text{global memory} = \left( S + 1 \right) \cdot \frac{N}{S} + U = N + \frac{N}{S} + U
$$
* **总 global memory：`N + N/S + U`**

Shared memory：
* `B_shared[u][s]` 被转置加载，每轮复用
* 每线程每轮读取 `S × U` 次（用于计算 `C_reg[U]`）
* 总共享内存访问次数：
`(1 + SU) * (N/S)`


Register：
* 每线程寄存器：
  * `A_reg[S]`：共 S 个
  * `C_reg[U]`：共 U 个
  * 总共：**`S + U` 个寄存器变量**
* `2U + N/S(S + SU + 2SU)`

| Kernel  | Global Memory          | Shared Memory | Register 使用 |
| ------- | ---------------------- | ------------- | ----------- |
| Matmul1 | `2N + 1`               | 无             | `2N + 2`         |
| Matmul2 | `2(N / Blocksize) + 1` | `2N + 2N/Blocksize`    | `2N + 2`         |
| Matmul3 | `N + N/S + U`          | `N/S *(1 + SU)`       | `3NU + N + 2U`   |

matmul3 总共有 N^2/U 个线程，最后需要乘 (N^2)/U

### 寄存器矩阵乘法的好处

优点：
- warp 数量减少，便于调度
- 减少 bank conflict 释放 shared memory
- 寄存器高访存效率，计算局部化，更快速的访问
- global, shared memory 和 register 访问次数更平均，更有利于存储器利用效率

缺点：
- 线程数量减少，不利于 pipeline 填充，降低 pipeline 吞吐量，导致无法通过切换隐藏住访存延迟

### 继续优化

考虑使用线程合并访问相邻元素

**Matmul1:**
在访问 A 矩阵的时候，按行访问 A 矩阵，导致 warp 内不同线程内存访问不连续，降低访存合并效率

可以考虑将 A 转置存储为 A_T，便于一个 warp 内的线程访存合并

**Matmul2:**
由于已经载入 shared memory，warp 内线程访问共享内存时已经是连续，故转置没什么用

**Matmul3:**
放入寄存器同理，作用不大