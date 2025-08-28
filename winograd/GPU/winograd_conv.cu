#include "winograd.cuh"

__global__
void winograd_conv_kernel(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW)
{
    const int n = blockIdx.x / K;
    const int k = blockIdx.x % K;
    const int idx = threadIdx.x + blockIdx.y * blockDim.x;
    const int row = idx / (outW / 2) * 2;
    const int col = idx % (outW / 2) * 2;

    extern __shared__ float smem[];
    float* shared_f = smem;
    float m[16] = {0.0f};
    float im_tile[16], v_ncp[16];

    for(int c = threadIdx.x; ; c += blockDim.x){
        if(c >= C) break;
        const float* f = filter + (k * C + c) * 9;
        float* out = shared_f + c * 16;
        float g[9] = {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]};

        out[0] = g[0];
        out[1] = 0.5f * (g[0] + g[1] + g[2]);
        out[2] = 0.5f * (g[0] - g[1] + g[2]);
        out[3] = g[2];
        out[4] = 0.5f * (g[0] + g[3] + g[6]);
        out[5] = 0.25f * (g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8]);
        out[6] = 0.25f * (g[0] - g[1] + g[2] + g[3] - g[4] + g[5] + g[6] - g[7] + g[8]);
        out[7] = 0.5f * (g[2] + g[5] + g[8]);
        out[8] = 0.5f * (g[0] - g[3] + g[6]);
        out[9] = 0.25f * (g[0] + g[1] + g[2] - g[3] - g[4] - g[5] + g[6] + g[7] + g[8]);
        out[10] = 0.25f * (g[0] - g[1] + g[2] - g[3] + g[4] - g[5] + g[6] - g[7] + g[8]);
        out[11] = 0.5f * (g[2] - g[5] + g[8]);
        out[12] = g[6];
        out[13] = 0.5f * (g[6] + g[7] + g[8]);
        out[14] = 0.5f * (g[6] - g[7] + g[8]);
        out[15] = g[8];
    }
    __syncthreads();
    if(idx >= (outH / 2) * (outW / 2)) return;

    for(int c = 0; c < C; c++){
        for(int i = 0; i < 16; i++)
            im_tile[i] = image[((n * C + c) * H + (row + i / 4)) * W + (col + i % 4)];

        v_ncp[0] = im_tile[0] - im_tile[2] - im_tile[8] + im_tile[10];
        v_ncp[1] = im_tile[1] + im_tile[2] - im_tile[9] - im_tile[10];
        v_ncp[2] = -im_tile[1] + im_tile[2] + im_tile[9] - im_tile[10];
        v_ncp[3] = im_tile[1] - im_tile[3] - im_tile[9] + im_tile[11];
        v_ncp[4] = im_tile[4] - im_tile[6] + im_tile[8] - im_tile[10];
        v_ncp[5] = im_tile[5] + im_tile[6] + im_tile[9] + im_tile[10];
        v_ncp[6] = -im_tile[5] + im_tile[6] - im_tile[9] + im_tile[10];
        v_ncp[7] = im_tile[5] - im_tile[7] + im_tile[9] - im_tile[11];
        v_ncp[8] = -im_tile[4] + im_tile[6] + im_tile[8] - im_tile[10];
        v_ncp[9] = -im_tile[5] - im_tile[6] + im_tile[9] + im_tile[10];
        v_ncp[10] = im_tile[5] - im_tile[6] - im_tile[9] + im_tile[10];
        v_ncp[11] = -im_tile[5] + im_tile[7] + im_tile[9] - im_tile[11];
        v_ncp[12] = im_tile[4] - im_tile[6] - im_tile[12] + im_tile[14];
        v_ncp[13] = im_tile[5] + im_tile[6] - im_tile[13] - im_tile[14];
        v_ncp[14] = -im_tile[5] + im_tile[6] + im_tile[13] - im_tile[14];
        v_ncp[15] = im_tile[5] - im_tile[7] - im_tile[13] + im_tile[15];

        for(int a = 0; a < 16; a++)
            m[a] += v_ncp[a] * shared_f[c * 16 + a];
    }

    output[blockIdx.x * outH * outW + row * outW + col] = m[0] + m[1] + m[2] + m[4] + m[5] + m[6] + m[8] + m[9] + m[10];
    output[blockIdx.x * outH * outW + row * outW + (col + 1)] = m[1] - m[2] - m[3] + m[5] - m[6] - m[7] + m[9] - m[10] - m[11];
    output[blockIdx.x * outH * outW + (row + 1) * outW + col] = m[4] + m[5] + m[6] - m[8] - m[9] - m[10] - m[12] - m[13] - m[14];
    output[blockIdx.x * outH * outW + (row + 1) * outW + (col + 1)] = m[5] - m[6] - m[7] - m[9] + m[10] + m[11] - m[13] + m[14] + m[15];
}

void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N)
{
    const int outH = H - 2;
    const int outW = W - 2;
    const int threads_per_block = 256;

    int num_threads = (outH / 2) * (outW / 2);
    dim3 grid_size = dim3(N * K, (num_threads + threads_per_block - 1) / threads_per_block);
    size_t smem_size = C * 4 * 4 * sizeof(float);
    winograd_conv_kernel<<<grid_size, threads_per_block, smem_size>>>(
        image.data().get(), filter.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}