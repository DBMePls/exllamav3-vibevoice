// --- START OF FILE exllamav3/exllamav3_ext/vibevoice_kernels.cu ---
#include <cuda_fp16.h>
#include "vibevoice_kernels.cuh"
#include <curand_kernel.h>
#include "util.h"
#include "util.cuh"

#define NUM_THREADS 256

__global__ void randn_f16_kernel(half* out, int size, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    out[idx] = __float2half(curand_normal(&state));
}

void vibevoice_randn_f16(at::Tensor& out, uint32_t seed) {
    int size = out.numel();
    randn_f16_kernel<<<CEIL_DIVIDE(size, NUM_THREADS), NUM_THREADS>>>(
        (half*)out.data_ptr(), size, seed
    );
}

__global__ void modulate_kernel(
    const half* __restrict__ xn, const half* __restrict__ adaln_out,
    half* __restrict__ mod_out, half* __restrict__ gate_out,
    int size, int dim, int num_chunks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int b_s = idx / dim;
    int d = idx % dim;

    float x_norm = __half2float(xn[idx]);
    float shift = __half2float(adaln_out[b_s * num_chunks * dim + d]);
    float scale = __half2float(adaln_out[b_s * num_chunks * dim + dim + d]);
    
    if (gate_out != nullptr && num_chunks == 3) {
        float gate  = __half2float(adaln_out[b_s * num_chunks * dim + 2 * dim + d]);
        gate_out[idx] = __float2half(gate);
    }

    mod_out[idx] = __float2half(x_norm * (1.0f + scale) + shift);
}

void vibevoice_modulate(at::Tensor& xn, const at::Tensor& adaln_out, at::Tensor& mod_out, at::Tensor& gate_out, int num_chunks) {
    int dim = xn.size(-1);
    int size = xn.numel();
    modulate_kernel<<<CEIL_DIVIDE(size, NUM_THREADS), NUM_THREADS>>>(
        (half*)xn.data_ptr(), (half*)adaln_out.data_ptr(),
        (half*)mod_out.data_ptr(), 
        gate_out.defined() && gate_out.numel() > 0 ? (half*)gate_out.data_ptr() : nullptr,
        size, dim, num_chunks
    );
}

__global__ void gated_residual_kernel(half* x, const half* f, const half* gate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x_val = __half2float(x[idx]);
    float f_val = __half2float(f[idx]);
    float g_val = __half2float(gate[idx]);
    x[idx] = __float2half(x_val + g_val * f_val);
}

void vibevoice_gated_residual(at::Tensor& x, const at::Tensor& f, const at::Tensor& gate) {
    int size = x.numel();
    gated_residual_kernel<<<CEIL_DIVIDE(size, NUM_THREADS), NUM_THREADS>>>(
        (half*)x.data_ptr(), (half*)f.data_ptr(), (half*)gate.data_ptr(), size
    );
}

__global__ void dpm_step_kernel(
    half* x, const half* v_pos, const half* v_neg, half* prev_x0,
    float cfg_scale, float a_t, float sg_t, float l_t,
    float a_s, float sg_s, float l_s, float l_prev,
    int order, bool is_first, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float vp = __half2float(v_pos[idx]);
    float vn = __half2float(v_neg[idx]);
    float v = vn + cfg_scale * (vp - vn);

    float cx = __half2float(x[idx]);
    float x0 = a_t * cx - sg_t * v;

    if (a_s == 1.0f && sg_s == 0.0f) { 
        x[idx] = __float2half(x0);
    } else if (order == 1) {
        float h = l_s - l_t;
        float A = sg_s / sg_t;
        float B_val = a_s * (expf(-h) - 1.0f);
        x[idx] = __float2half(A * cx - B_val * x0);
    } else {
        float h = l_s - l_t;
        float h_0 = l_t - l_prev;
        float r0 = h_0 / h;
        float A = sg_s / sg_t;
        float B_val = a_s * (expf(-h) - 1.0f);
        float Bd1 = 0.5f * B_val;

        float D0 = x0;
        float D1 = (1.0f / r0) * (x0 - __half2float(prev_x0[idx]));
        x[idx] = __float2half(A * cx - B_val * D0 - Bd1 * D1);
    }
    prev_x0[idx] = __float2half(x0);
}

void vibevoice_dpm_step(
    at::Tensor& x, const at::Tensor& v_pos, const at::Tensor& v_neg, at::Tensor& prev_x0,
    float cfg_scale, float a_t, float sg_t, float l_t, float a_s, float sg_s, float l_s, float l_prev,
    int order, bool is_first
) {
    int size = x.numel();
    dpm_step_kernel<<<CEIL_DIVIDE(size, NUM_THREADS), NUM_THREADS>>>(
        (half*)x.data_ptr(), (half*)v_pos.data_ptr(), (half*)v_neg.data_ptr(), (half*)prev_x0.data_ptr(),
        cfg_scale, a_t, sg_t, l_t, a_s, sg_s, l_s, l_prev, order, is_first, size
    );
}