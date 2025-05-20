#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#define smalloc(type,ptr,num) if(!(ptr=(type *)malloc(sizeof(type)*(num)))) exit(1)
#define Blocksize (32)
#define Matsize (4096)
#define Verifysize (1024)
#define T (2*64)
#define U (2*64)
#define S (T/U)

__global__ void Matmul1(float *A,float *B,float *C,unsigned N){
    unsigned row=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned col=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned k;
    float sum=0;
    for(k=0;k<N;k++){
        sum+=A[row*N+k]*B[k*N+col];
    }
    C[row*N+col]=sum;
}

__global__ void Matmul2(float *A,float *B,float *C,unsigned N){// A,B with padding
    unsigned tx=threadIdx.x,ty=threadIdx.y;
    unsigned bx=blockIdx.x,by=blockIdx.y;
    unsigned row=by*blockDim.y+ty;
    unsigned col=bx*blockDim.x+tx;
    __shared__ float Asub[Blocksize][Blocksize],Bsub[Blocksize][Blocksize];
    float sum=0;   
    unsigned kk,k;
    for(kk=0;kk<N;kk+=Blocksize){
        Asub[ty][tx]=A[row*N+(kk+tx)];
        Bsub[ty][tx]=B[(kk+ty)*N+col];
        __syncthreads();
        for(k=0;k<Blocksize;k++){
            sum+=Asub[ty][k]*Bsub[k][tx];
        }
        __syncthreads();
    }
    C[row*N+col]=sum;
}


// 使用寄存器共享加速
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


__host__ void matmubase(float *A,float *B,float *C,unsigned N){
    unsigned i,j,k;
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            C[i*N+j]=0;
            for(k=0;k<N;k++){
                C[i*N+j]+=A[i*N+k]*B[k*N+j];
            }
        }
    }
}

__host__ void gen_mat(float **pA,float **pB,unsigned N){
    float *A,*B;
    smalloc(float,A,N*N);
    smalloc(float,B,N*N);
    unsigned i;
    for (i = 0; i < N*N; i++){
        A[i] = 1.0*rand()/RAND_MAX;
        B[i] = 1.0*rand()/RAND_MAX;
    }
    *pA=A;*pB=B;
}

__host__ unsigned compare(float *pred_,float *true_, unsigned n){
    unsigned i;
    float relative_error;
    for(i=0;i<n;i++){
        relative_error=fabs((pred_[i]-true_[i])/true_[i]);
        if(relative_error>=1e-6){
            printf("not equal! relative error: %12.9lf pred: %12.9f true: %12.9f\n",
                relative_error,pred_[i],true_[i]);
            return 1;
        }
    }
    printf("equal!\n");
    return 0;
}


int main(void){
    const unsigned PN=Matsize,VN=Verifysize;
    float *hA,*hB,*hC1,*hC2,*dA,*dB,*dC1,*dC2,*Cbase;
    float *hc3, *dC3;
    gen_mat(&hA,&hB,VN);
    smalloc(float,Cbase,sizeof(float)*VN*VN);
    smalloc(float,hC1,sizeof(float)*VN*VN);
    smalloc(float,hC2,sizeof(float)*VN*VN);
    smalloc(float,hc3,sizeof(float)*VN*VN);
    cudaMalloc(&dA,sizeof(float)*VN*VN);
    cudaMalloc(&dB,sizeof(float)*VN*VN);
    cudaMalloc(&dC1,sizeof(float)*VN*VN);
    cudaMalloc(&dC2,sizeof(float)*VN*VN);
    cudaMalloc(&dC3,sizeof(float)*VN*VN);
    cudaMemcpy(dA, hA, sizeof(float)*VN*VN, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*VN*VN, cudaMemcpyHostToDevice);

    dim3 gridsize(VN/Blocksize,VN/Blocksize),blocksize(Blocksize,Blocksize);
    dim3 gridsize3(VN/U, VN/T), blocksize3(1, T);  // T 个线程，每个处理一行的 U 个元素
    Matmul1 <<<gridsize ,blocksize >>>(dA,dB,dC1,VN);
    Matmul2 <<<gridsize ,blocksize >>>(dA,dB,dC2,VN);
    Matmul3 <<<gridsize3,blocksize3>>>(dA,dB,dC3,VN);
    cudaMemcpy(hC1, dC1, sizeof(float)*VN*VN, cudaMemcpyDeviceToHost);
    cudaMemcpy(hC2, dC2, sizeof(float)*VN*VN, cudaMemcpyDeviceToHost);
    cudaMemcpy(hc3, dC3, sizeof(float)*VN*VN, cudaMemcpyDeviceToHost);
    matmubase(hA,hB,Cbase,VN);
    cudaDeviceSynchronize();


    int flag=0;
    flag|=compare(hC1,Cbase,VN*VN);
    flag|=compare(hC2,Cbase,VN*VN);
    flag|=compare(hc3,Cbase,VN*VN);
    if(flag){
        printf("error!\n");
        exit(1);
    }
    printf("pass!\n");
    free(hA);free(hB);free(hC1);free(hC2);free(Cbase);free(hc3);
    cudaFree(dA);cudaFree(dB);cudaFree(dC1);cudaFree(dC2);cudaFree(dC3);


    gen_mat(&hA,&hB,PN);
    cudaMalloc(&dA,sizeof(float)*PN*PN);
    cudaMalloc(&dB,sizeof(float)*PN*PN);
    cudaMalloc(&dC1,sizeof(float)*PN*PN);
    cudaMalloc(&dC2,sizeof(float)*PN*PN);
    cudaMalloc(&dC3,sizeof(float)*PN*PN);
    cudaMemcpy(dA, hA, sizeof(float)*PN*PN, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*PN*PN, cudaMemcpyHostToDevice);

    gridsize={PN/Blocksize,PN/Blocksize};blocksize={Blocksize,Blocksize};
    gridsize3={PN/U,PN/T};blocksize3={1,T};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float Time1 = 0.0,Time2=0.0,Time3=0.0,temp=0;
    const unsigned loopnum=10;
    unsigned i;
    for(i=0;i<loopnum;i++){

        cudaEventRecord(start, 0);
        Matmul1 <<<gridsize,blocksize>>>(dA,dB,dC1,PN);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        Time1+=temp;temp=0;

        cudaDeviceSynchronize();
        

        cudaEventRecord(start, 0);
        Matmul2 <<<gridsize,blocksize>>>(dA,dB,dC2,PN);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        Time2+=temp;temp=0;

        cudaDeviceSynchronize();
        
        cudaEventRecord(start, 0);
        Matmul3 <<<gridsize3,blocksize3>>>(dA,dB,dC3,PN);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        Time3+=temp;temp=0;

        cudaDeviceSynchronize();
    }
    
    printf("N: %5.d  time1: %12.9f  time2: %12.9f time3: %12.9f\n"
            ,PN,Time1/loopnum,Time2/loopnum,Time3/loopnum);
    free(hA);free(hB);
    cudaFree(dA);cudaFree(dB);cudaFree(dC1);cudaFree(dC2);cudaFree(dC3);

}