#include <cstdio>
#include <random>

#include <gem5/m5ops.h>

// Y = alpha * X + Y
void daxpy(double *X, double *Y, double alpha, const int N)
{
    for (int i = 0; i < N; i++)
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

// Y = alpha * X^2 + beta * X + X * Y
void daxsbxpxy(double *X, double *Y, double alpha, double beta, const int N)
{
    for (int i = 0; i < N; i++)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
    }
}

// 一维模板操作 Y[i] = alpha * Y[i-1] + Y[i] + alpha * Y[i+1]
void stencil(double *Y, double alpha, const int N)
{
    for (int i = 1; i < N-1; i++)
    {
        Y[i] = alpha * Y[i-1] + Y[i] + alpha * Y[i+1];
    }
}

// // Y = alpha * X + Y
// void daxpy_unroll8(double *X, double *Y, double alpha, const int N)
// {
//     int i = 0;
//     // 循环展开，每次处理 8 个元素
//     for (; i <= N - 8; i += 8)
//     {
//         Y[i] = alpha * X[i] + Y[i];
//         Y[i + 1] = alpha * X[i + 1] + Y[i + 1];
//         Y[i + 2] = alpha * X[i + 2] + Y[i + 2];
//         Y[i + 3] = alpha * X[i + 3] + Y[i + 3];
//         Y[i + 4] = alpha * X[i + 4] + Y[i + 4];
//         Y[i + 5] = alpha * X[i + 5] + Y[i + 5];
//         Y[i + 6] = alpha * X[i + 6] + Y[i + 6];
//         Y[i + 7] = alpha * X[i + 7] + Y[i + 7];
//     }
//     // 处理剩余的元素
//     for (; i < N; i++)
//     {
//         Y[i] = alpha * X[i] + Y[i];
//     }
// }

// // Y = alpha * X^2 + beta * X + X * Y
// void daxsbxpxy_unroll8(double *X, double *Y, double alpha, double beta, const int N)
// {
//     int i = 0;
//     // 循环展开，每次处理 8 个元素
//     for (; i <= N - 8; i += 8)
//     {
//         Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
//         Y[i + 1] = alpha * X[i + 1] * X[i + 1] + beta * X[i + 1] + X[i + 1] * Y[i + 1];
//         Y[i + 2] = alpha * X[i + 2] * X[i + 2] + beta * X[i + 2] + X[i + 2] * Y[i + 2];
//         Y[i + 3] = alpha * X[i + 3] * X[i + 3] + beta * X[i + 3] + X[i + 3] * Y[i + 3];
//         Y[i + 4] = alpha * X[i + 4] * X[i + 4] + beta * X[i + 4] + X[i + 4] * Y[i + 4];
//         Y[i + 5] = alpha * X[i + 5] * X[i + 5] + beta * X[i + 5] + X[i + 5] * Y[i + 5];
//         Y[i + 6] = alpha * X[i + 6] * X[i + 6] + beta * X[i + 6] + X[i + 6] * Y[i + 6];
//         Y[i + 7] = alpha * X[i + 7] * X[i + 7] + beta * X[i + 7] + X[i + 7] * Y[i + 7];
//     }
//     // 处理剩余的元素
//     for (; i < N; i++)
//     {
//         Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
//     }
// }

// // 一维模板操作 Y[i] = alpha * Y[i-1] + Y[i] + alpha * Y[i+1]
// void stencil_unroll8(double *Y, double alpha, const int N)
// {
//     int i = 1;
//     // 循环展开，每次处理 8 个元素
//     for (; i <= N - 9; i += 8)
//     {
//         Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
//         Y[i + 1] = alpha * Y[i] + Y[i + 1] + alpha * Y[i + 2];
//         Y[i + 2] = alpha * Y[i + 1] + Y[i + 2] + alpha * Y[i + 3];
//         Y[i + 3] = alpha * Y[i + 2] + Y[i + 3] + alpha * Y[i + 4];
//         Y[i + 4] = alpha * Y[i + 3] + Y[i + 4] + alpha * Y[i + 5];
//         Y[i + 5] = alpha * Y[i + 4] + Y[i + 5] + alpha * Y[i + 6];
//         Y[i + 6] = alpha * Y[i + 5] + Y[i + 6] + alpha * Y[i + 7];
//         Y[i + 7] = alpha * Y[i + 6] + Y[i + 7] + alpha * Y[i + 8];
//     }
//     // 处理剩余的元素
//     for (; i < N - 1; i++)
//     {
//         Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
//     }
// }

// // Y = alpha * X + Y
// void daxpy_unroll2(double *X, double *Y, double alpha, const int N)
// {
//     int i = 0;
//     // 循环展开，每次处理 2 个元素
//     for (; i <= N - 2; i += 2)
//     {
//         Y[i] = alpha * X[i] + Y[i];
//         Y[i + 1] = alpha * X[i + 1] + Y[i + 1];
//     }
//     // 处理剩余的元素
//     for (; i < N; i++)
//     {
//         Y[i] = alpha * X[i] + Y[i];
//     }
// }

// // Y = alpha * X^2 + beta * X + X * Y
// void daxsbxpxy_unroll2(double *X, double *Y, double alpha, double beta, const int N)
// {
//     int i = 0;
//     // 循环展开，每次处理 2 个元素
//     for (; i <= N - 2; i += 2)
//     {
//         Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
//         Y[i + 1] = alpha * X[i + 1] * X[i + 1] + beta * X[i + 1] + X[i + 1] * Y[i + 1];
//     }
//     // 处理剩余的元素
//     for (; i < N; i++)
//     {
//         Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
//     }
// }

// // 一维模板操作 Y[i] = alpha * Y[i-1] + Y[i] + alpha * Y[i+1]
// void stencil_unroll2(double *Y, double alpha, const int N)
// {
//     int i = 1;
//     // 循环展开，每次处理 2 个元素
//     for (; i <= N - 3; i += 2)
//     {
//         Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
//         Y[i + 1] = alpha * Y[i] + Y[i + 1] + alpha * Y[i + 2];
//     }
//     // 处理剩余的元素
//     for (; i < N - 1; i++)
//     {
//         Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
//     }
// }

// Y = alpha * X + Y
void daxpy_unroll(double *X, double *Y, double alpha, const int N)
{
    int i = 0;
    // 循环展开，每次处理 4 个元素
    for (; i <= N - 4; i += 4)
    {
        Y[i] = alpha * X[i] + Y[i];
        Y[i + 1] = alpha * X[i + 1] + Y[i + 1];
        Y[i + 2] = alpha * X[i + 2] + Y[i + 2];
        Y[i + 3] = alpha * X[i + 3] + Y[i + 3];
    }
    // 处理剩余的元素
    for (; i < N; i++)
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

// Y = alpha * X^2 + beta * X + X * Y
void daxsbxpxy_unroll(double *X, double *Y, double alpha, double beta, const int N)
{
    int i = 0;
    // 循环展开，每次处理 4 个元素
    for (; i <= N - 4; i += 4)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
        Y[i + 1] = alpha * X[i + 1] * X[i + 1] + beta * X[i + 1] + X[i + 1] * Y[i + 1];
        Y[i + 2] = alpha * X[i + 2] * X[i + 2] + beta * X[i + 2] + X[i + 2] * Y[i + 2];
        Y[i + 3] = alpha * X[i + 3] * X[i + 3] + beta * X[i + 3] + X[i + 3] * Y[i + 3];
    }
    // 处理剩余的元素
    for (; i < N; i++)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
    }
}

// 一维模板操作 Y[i] = alpha * Y[i-1] + Y[i] + alpha * Y[i+1]
void stencil_unroll(double *Y, double alpha, const int N)
{
    int i = 1;
    // 循环展开，每次处理 4 个元素
    for (; i <= N - 5; i += 4)
    {
        Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
        Y[i + 1] = alpha * Y[i] + Y[i + 1] + alpha * Y[i + 2];
        Y[i + 2] = alpha * Y[i + 1] + Y[i + 2] + alpha * Y[i + 3];
        Y[i + 3] = alpha * Y[i + 2] + Y[i + 3] + alpha * Y[i + 4];
    }
    // 处理剩余的元素
    for (; i < N - 1; i++)
    {
        Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
    }
}


int main()
{
    const int N = 10000;    // 数据大小
    double *X = new double[N], *Y = new double[N], alpha = 0.5, beta = 0.1;

    // 初始化
    //std::random_device rd;
    std::mt19937 gen(0);    // 固定随机数种子
    std::uniform_real_distribution<> dis(1, 2);
    for (int i = 0; i < N; ++i)
    {
        X[i] = dis(gen);
        Y[i] = dis(gen);
    }

    // 统计
    m5_dump_reset_stats(0, 0);  // 重置统计
    daxpy(X, Y, alpha, N);
    m5_dump_reset_stats(0, 0);
    daxpy_unroll(X, Y, alpha, N);
    m5_dump_reset_stats(0, 0);
    daxsbxpxy(X, Y, alpha, beta, N);
    m5_dump_reset_stats(0, 0);
    daxsbxpxy_unroll(X, Y, alpha, beta, N);
    m5_dump_reset_stats(0, 0);
    stencil(Y, alpha, N);
    m5_dump_reset_stats(0, 0);
    stencil_unroll(Y, alpha, N);
    m5_dump_reset_stats(0, 0);

    // 计算数组和
    double sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += Y[i];
    }
    printf("%lf\n", sum);
    return 0;
}
