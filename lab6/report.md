# Flash Attention 实验报告

## 代码实现

```c++
__global__ void Single_head_flash_atten(float *Q,float *K,float *V,float *O,unsigned l,float softmax_scale) {
    __shared__ float v_shared[AT][AD];
    __shared__ float k_shared[AT][AD];
    unsigned tid = threadIdx.x;
    unsigned blk_row = blockIdx.x;
    unsigned row_start = blk_row * AT;
    float q_reg[AD] = {0};
    for (int i = 0; i < AD; i++) {
        q_reg[i] = Q[(row_start + tid) * AD + i];
    }
    __syncthreads();
    float s_reg[AT] = {0};
    float s_sum = 0.0f;
    float o_reg[AD] = {0};
    for (int blk_col = 0; blk_col < l / AT; blk_col++) {
        unsigned kv_row = blk_col * AT + tid;
        for (int i = 0; i < AD; i++) {
            k_shared[tid][i] = K[kv_row * AD + i];
            v_shared[tid][i] = V[kv_row * AD + i];
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < AT; j++) {
            float s_val = 0;
            for (int k = 0; k < AD; k++) {
                s_val += q_reg[k] * k_shared[j][k];
            }
            s_reg[j] = expf(s_val * softmax_scale);
            s_sum += s_reg[j];
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < AD; i++) {
            for (int j = 0; j < AT; j++) {
                o_reg[i] += s_reg[j] * v_shared[j][i];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < AD; i++) {
        o_reg[i] /= s_sum;
    }
    unsigned global_row = row_start + tid;
    for (int i = 0; i < AD; i++) {
        O[global_row * AD + i] = o_reg[i];
    }
}
```


**思路**：
- 每个线程块处理 Q 的一个行块（大小为 AT × AD）；
- 每个线程加载一行 Q 与 K 的一列；
- 每个 Q 块要与所有的 K 块进行计算（列方向），即 loop over bc（= l / AT） 一个线程计算 S 的一行和 O 的一行
- 将 K V 块读入共享内存 每个线程计算一行
- 最后统一进行 softmax 归一化和输出 O 的计算。

## 思考题

### 请问原始flash attention 算法中能否将内层循环改为对K矩阵操作？请说明你变动后算法Q、K、V、O矩阵的搬运过程及这样做的好处与坏处（提示：考虑softmax）

正确性上可以，但是改动后会导致计算效率下降。
- 好处：对于长度更长的 Q，Q 的搬运更频繁，固定 Q 块能减少数据搬运（提升缓存效率）
- 坏处：原始 softmax 不能直接累加，需要暂存更多的中间值，增大显存占用；无法有效利用共享内存缓存 K/V

### 原始flash attention算法中Q、K、V、O小块大小能如(Br,d/2)这样吗？请分析可能遇到的问题

不行，这样会导致计算出来的每个 S 分块中的元素不是最终的结果，需要进行线性合并；但 softmax 没有线性性，导致要先计算出完整的 S，效率低下

### Single head Flash attention启动N个线程相比原始flash attention 算法对矩阵分块的复用有什么影响？请详细分析

- 可能重复搬运，原始 Flash Attention 中，K/V 小块在多个 Q 块上重复使用，具有高复用率
- 线程间竞争共享内存，需更多带宽，原来是 Br 个线程竞争，现在是 N 个线程竞争。

| 项目      | 原始算法             | 你的实现              |
| ------- | ---------------- | ----------------- |
| K/V 复用率 | 高（每次加载后用于多个 Q 块） | 低（每次只用于当前 Q 块）    |
| 共享内存利用  | 高（K/V 放共享内存）     | 中（共享内存用于当前块的 K/V） |
| 全局内存访问  | 较少（一次加载复用多次）     | 多（每个线程块都加载全量 K/V） |
| 计算并行性   | 按 Q 行块划分并行       | 每行一个线程并行，高并行度     |
| 可优化空间   | 有，通过线程协作/缓存      | 少，除非引入更多协作与复用策略   |


### 你实现的Single head Flash attention运行速度比利用传统矩阵乘法实现的Single head Flash attention慢。请分析原因并给出改进方案及方案面临的困难。

可能的原因：
- 每个线程块都要重新从 global memory 加载 K/V，小块复用率低
- softmax 过程中对 s_sum 的累积可能未并行优化，降低吞吐量。

| 方向              | 改进措施                                                   | 面临的困难                         |
| --------------- | ------------------------------------------------------ | ----------------------------- |
| 提高 K/V 复用       | 将 K/V 缓存在全局或共享内存中，多个 Q 块使用相同 K/V                       | 需要跨线程协作，设计复杂                  |
| 使用 warp shuffle | softmax 求和/归一化用 warp shuffle 提速                        | 需了解 warp 内通信，增加代码复杂度          |
| 分层并行            | 每个 block 内的线程协作计算多个 Q 行，减少线程数但增加共享内存复用                 | 控制好并行度和线程调度                   |
| 流水线分块计算         | 改为 outer loop over Q，inner loop over K，设计 softmax 累积变量 | 累加 softmax 分母、最大值等中间统计量需要额外逻辑 |
