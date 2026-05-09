// --- START OF FILE exllamav3/exllamav3_ext/vibevoice_kernels.cuh ---
#pragma once
#include <ATen/Tensor.h>

void vibevoice_randn_f16(at::Tensor& out, uint32_t seed);

void vibevoice_modulate(
    at::Tensor& xn, 
    const at::Tensor& adaln_out, 
    at::Tensor& mod_out, 
    at::Tensor& gate_out,
    int num_chunks
);

void vibevoice_gated_residual(
    at::Tensor& x,
    const at::Tensor& f,
    const at::Tensor& gate
);

void vibevoice_dpm_step(
    at::Tensor& x,
    const at::Tensor& v_pos,
    const at::Tensor& v_neg,
    at::Tensor& prev_x0,
    float cfg_scale,
    float a_t, float sg_t, float l_t,
    float a_s, float sg_s, float l_s,
    float l_prev,
    int order,
    bool is_first
);