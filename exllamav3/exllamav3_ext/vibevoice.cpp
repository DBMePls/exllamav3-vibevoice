#include "vibevoice.h"
#include "vibevoice_kernels.cuh"
#include "hgemm.cuh"
#include "activation.cuh"
#include "norm.cuh"
#include <ATen/cuda/CUDAContext.h>

VibeVoiceWorker::VibeVoiceWorker(
    std::map<std::string, at::Tensor> weights,
    std::vector<int> timesteps,
    at::Tensor alpha_t, at::Tensor sigma_t, at::Tensor lambda_t,
    int num_layers, int num_vae_upsamples, std::vector<int> vae_depths
) : weights(weights), timesteps(timesteps), alpha_t(alpha_t), sigma_t(sigma_t), lambda_t(lambda_t), 
    num_layers(num_layers), num_vae_upsamples(num_vae_upsamples), vae_depths(vae_depths) {}

void VibeVoiceWorker::head_forward(at::Tensor x, at::Tensor cond, float t_val, DiTWorkspace& ws) {
    int B = x.size(0);
    int T = x.size(1);
    
    int dit_dim = weights["noisy_proj"].size(1);
    int ffn_dim = weights["layer_0_gate"].size(1);
    
    at::Tensor t_tensor = torch::full({B}, t_val, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    int half_dim = 128;
    auto freqs = torch::exp(-std::log(10000.0f) * torch::arange(0, half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())) / half_dim);
    auto args = t_tensor.unsqueeze(1) * freqs.unsqueeze(0);
    auto t_sin = torch::cat({torch::cos(args), torch::sin(args)}, -1).to(torch::kHalf);

    at::Tensor x_flat = x.view({B * T, x.size(-1)});
    at::Tensor cond_flat = cond.view({B * T, cond.size(-1)});

    hgemm(x_flat, weights["noisy_proj"], ws.h_x);

    hgemm(t_sin, weights["t_mlp0"], ws.t_emb);
    torch::silu_out(ws.t_emb, ws.t_emb); 
    hgemm(ws.t_emb, weights["t_mlp2"], ws.t_emb2);
    
    hgemm(cond_flat, weights["cond_proj"], ws.c);
    
    ws.c.view({B, T, ws.c.size(-1)}).add_(ws.t_emb2.unsqueeze(1));
    torch::silu_out(ws.c_silu, ws.c);

    for (int i = 0; i < num_layers; i++) {
        std::string pfx = "layer_" + std::to_string(i) + "_";

        hgemm(ws.c_silu, weights[pfx + "adaln"], ws.adaln_out);

        rms_norm(ws.h_x, weights[pfx + "norm"], ws.mod_out, 1e-6f, 0.0f);
        vibevoice_modulate(ws.mod_out, ws.adaln_out, ws.mod_out, ws.gate_out_param, 3);
        
        hgemm(ws.mod_out, weights[pfx + "gate"], ws.ffn_gate);
        hgemm(ws.mod_out, weights[pfx + "up"], ws.ffn_up);
        silu_mul(ws.ffn_gate, ws.ffn_up, ws.ffn_gate);
        
        hgemm(ws.ffn_gate, weights[pfx + "down"], ws.f);

        vibevoice_gated_residual(ws.h_x, ws.f, ws.gate_out_param);
    }

    hgemm(ws.c_silu, weights["final_adaln"], ws.final_adaln_out);
    
    rms_norm(ws.h_x, ws.dummy_norm, ws.mod_out, 1e-6f, 0.0f);
    
    at::Tensor empty_gate;
    vibevoice_modulate(ws.mod_out, ws.final_adaln_out, ws.mod_out, empty_gate, 2); 
    
    hgemm(ws.mod_out, weights["final_proj"], ws.out);
}

at::Tensor VibeVoiceWorker::sample_latent(at::Tensor cond_pos, at::Tensor cond_neg, float cfg_scale, int seed, bool increase_cfg) {
    torch::NoGradGuard no_grad;
    int B = cond_pos.size(0);
    int T = cond_pos.size(1);
    auto device = cond_pos.device();

    at::Tensor x = torch::empty({B, T, 64}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    vibevoice_randn_f16(x, seed);
    at::Tensor prev_x0 = torch::zeros_like(x);

    int dit_dim = weights["noisy_proj"].size(1);
    int ffn_dim = weights["layer_0_gate"].size(1);

    DiTWorkspace ws;
    ws.h_x = torch::empty({B * T, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.t_emb = torch::empty({B, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.t_emb2 = torch::empty({B, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.c = torch::empty({B * T, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.c_silu = torch::empty({B * T, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.adaln_out = torch::empty({B * T, 3 * dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.final_adaln_out = torch::empty({B * T, 2 * dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.mod_out = torch::empty({B * T, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.gate_out_param = torch::empty({B * T, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.ffn_gate = torch::empty({B * T, ffn_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.ffn_up = torch::empty({B * T, ffn_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.f = torch::empty({B * T, dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.out = torch::empty({B * T, 64}, torch::TensorOptions().dtype(torch::kHalf).device(device));
    ws.dummy_norm = torch::ones({dit_dim}, torch::TensorOptions().dtype(torch::kHalf).device(device));

    at::Tensor v_pos = torch::empty_like(x);
    at::Tensor v_neg = torch::empty_like(x);

    int M = timesteps.size() - 1;
    for (int i = 0; i < M; ++i) {
        int t = timesteps[i];
        int s = timesteps[i + 1];

        float a_t = alpha_t[t].item<float>();
        float sg_t = sigma_t[t].item<float>();
        float l_t = lambda_t[t].item<float>();
        float l_prev = (i > 0) ? lambda_t[timesteps[i - 1]].item<float>() : 0.0f;

        float a_s = 1.0f, sg_s = 0.0f, l_s = 0.0f;
        if (s != -1) {
            a_s = alpha_t[s].item<float>();
            sg_s = sigma_t[s].item<float>();
            l_s = lambda_t[s].item<float>();
        }

        head_forward(x, cond_pos, (float)t, ws);
        v_pos.copy_(ws.out.view({B, T, 64}));
        
        if (cfg_scale > 1.0f) {
            head_forward(x, cond_neg, (float)t, ws);
            v_neg.copy_(ws.out.view({B, T, 64}));
        } else {
            v_neg.copy_(v_pos);
        }

        bool is_first = (i == 0);
        bool is_last = (i == M - 1);
        int order = (is_first || is_last) ? 1 : 2;
        
        float current_cfg = cfg_scale;
        if (increase_cfg && ((float)i / M) < 0.5f) {
            current_cfg *= 1.5f;
        }

        vibevoice_dpm_step(x, v_pos, v_neg, prev_x0, current_cfg, a_t, sg_t, l_t, a_s, sg_s, l_s, l_prev, order, is_first);
    }
    
    return x;
}

// ... Keep VibeVoiceWorker::head_forward & VibeVoiceWorker::sample_latent ...

// --- VAE ENCODER HELPERS ---

at::Tensor VibeVoiceWorker::causal_conv1d_enc(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, int stride, int dilation, int groups) {
    int k = weight.size(2);
    int pad_total = (k - 1) * dilation - (stride - 1);
    int T_in = x.size(2);
    float n_frames = (T_in - k + pad_total) / (float)stride + 1.0f;
    int ideal_length = (std::ceil(n_frames) - 1) * stride + (k - pad_total);
    int extra = std::max(0, ideal_length - T_in);
    
    at::Tensor x_pad = torch::constant_pad_nd(x, {pad_total, extra}, 0.0);
    std::vector<int64_t> stride_vec = {stride};
    std::vector<int64_t> pad_vec = {0};
    std::vector<int64_t> dil_vec = {dilation};
    return torch::conv1d(x_pad, weight, bias, stride_vec, pad_vec, dil_vec, groups);
}

at::Tensor VibeVoiceWorker::enc_block(at::Tensor x, const std::string& bpfx) {
    at::Tensor res = x;
    at::Tensor h = x.transpose(1, 2);
    h = h * torch::rsqrt(h.pow(2).mean(-1, /*keepdim=*/true) + 1e-5) * weights[bpfx + "norm.weight"];
    h = h.transpose(1, 2);
    
    c10::optional<at::Tensor> conv_b = weights.count(bpfx + "mixer.conv.conv.conv.bias") ? c10::make_optional(weights[bpfx + "mixer.conv.conv.conv.bias"]) : c10::nullopt;
    h = causal_conv1d_enc(h, weights[bpfx + "mixer.conv.conv.conv.weight"], conv_b, 1, 1, h.size(1));
    
    if (weights.count(bpfx + "gamma")) {
        h = h * weights[bpfx + "gamma"].view({1, -1, 1});
    }
    h = res + h;
    
    res = h;
    at::Tensor f = h.transpose(1, 2);
    f = f * torch::rsqrt(f.pow(2).mean(-1, /*keepdim=*/true) + 1e-5) * weights[bpfx + "ffn_norm.weight"];
    
    at::Tensor ffn_1_b = weights.count(bpfx + "ffn.linear1.bias") ? weights[bpfx + "ffn.linear1.bias"] : at::Tensor();
    f = torch::nn::functional::linear(f, weights[bpfx + "ffn.linear1.weight"], ffn_1_b);
    
    auto gelu_options = torch::nn::functional::GELUFuncOptions().approximate("tanh");
    f = torch::nn::functional::gelu(f, gelu_options);
    
    at::Tensor ffn_2_b = weights.count(bpfx + "ffn.linear2.bias") ? weights[bpfx + "ffn.linear2.bias"] : at::Tensor();
    f = torch::nn::functional::linear(f, weights[bpfx + "ffn.linear2.weight"], ffn_2_b);
    f = f.transpose(1, 2);
    
    if (weights.count(bpfx + "ffn_gamma")) {
        f = f * weights[bpfx + "ffn_gamma"].view({1, -1, 1});
    }
    return res + f;
}

// --- FULL C++ VAE ENCODER ---

at::Tensor VibeVoiceWorker::encode_acoustic(at::Tensor audio) {
    torch::NoGradGuard no_grad;
    
    std::string pfx = "model.acoustic_tokenizer.encoder.";
    at::Tensor h = audio;
    
    c10::optional<at::Tensor> stem_b = weights.count(pfx + "downsample_layers.0.0.conv.conv.bias") ? c10::make_optional(weights[pfx + "downsample_layers.0.0.conv.conv.bias"]) : c10::nullopt;
    h = causal_conv1d_enc(h, weights[pfx + "downsample_layers.0.0.conv.conv.weight"], stem_b, 1, 1, 1);
    
    for (int bi = 0; bi < vae_depths[6]; bi++) {
        h = enc_block(h, pfx + "stages.0." + std::to_string(bi) + ".");
    }
    
    std::vector<int> rev_ratios = {2, 2, 4, 5, 5, 8};
    for (int i = 0; i < 6; i++) {
        int stride = rev_ratios[i];
        std::string ds_pfx = pfx + "downsample_layers." + std::to_string(i + 1) + ".0.conv.conv.";
        c10::optional<at::Tensor> b = weights.count(ds_pfx + "bias") ? c10::make_optional(weights[ds_pfx + "bias"]) : c10::nullopt;
        
        h = causal_conv1d_enc(h, weights[ds_pfx + "weight"], b, stride, 1, 1);
        
        for (int bi = 0; bi < vae_depths[6 - (i + 1)]; bi++) {
            h = enc_block(h, pfx + "stages." + std::to_string(i + 1) + "." + std::to_string(bi) + ".");
        }
    }
    
    if (weights.count(pfx + "norm.weight")) {
        h = h.transpose(1, 2);
        h = h * torch::rsqrt(h.pow(2).mean(-1, /*keepdim=*/true) + 1e-5) * weights[pfx + "norm.weight"];
        h = h.transpose(1, 2);
    }
    
    c10::optional<at::Tensor> head_b = weights.count(pfx + "head.conv.conv.bias") ? c10::make_optional(weights[pfx + "head.conv.conv.bias"]) : c10::nullopt;
    h = causal_conv1d_enc(h, weights[pfx + "head.conv.conv.weight"], head_b, 1, 1, 1);
    
    float scale = weights["vae_scale"].item<float>();
    float bias = weights["vae_bias"].item<float>();
    h = (h + bias) * scale;
    
    return acoustic_connector_forward(h.transpose(1, 2));
}

// --- C++ ACOUSTIC CONNECTOR ---

at::Tensor VibeVoiceWorker::acoustic_connector_forward(at::Tensor z) {
    torch::NoGradGuard no_grad;
    at::Tensor h = z.to(torch::kFloat32);
    std::string conn_pfx = "model.acoustic_connector.";
    
    at::Tensor fc1_b = weights.count(conn_pfx + "fc1.bias") ? weights[conn_pfx + "fc1.bias"] : at::Tensor();
    h = torch::nn::functional::linear(h, weights[conn_pfx + "fc1.weight"], fc1_b);
    
    h = h * torch::rsqrt(h.pow(2).mean(-1, /*keepdim=*/true) + 1e-6);
    if (weights.count(conn_pfx + "norm.weight")) {
        h = h * weights[conn_pfx + "norm.weight"];
    }
    
    at::Tensor fc2_b = weights.count(conn_pfx + "fc2.bias") ? weights[conn_pfx + "fc2.bias"] : at::Tensor();
    h = torch::nn::functional::linear(h, weights[conn_pfx + "fc2.weight"], fc2_b);
    
    return h;
}

// --- VAE DECODER HELPERS ---

at::Tensor VibeVoiceWorker::causal_conv1d_dec(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, int stride, int dilation, int groups) {
    int k = weight.size(2);
    int pad_left = std::max(0, (k - 1) * dilation - (stride - 1));
    
    at::Tensor x_pad = torch::constant_pad_nd(x, {pad_left, 0}, 0.0);
    
    int T_in = x.size(2);
    int pad_right = 0;
    if (stride > 1) {
        float n_frames = (T_in - k + pad_left) / (float)stride + 1.0f;
        int ideal_length = (std::ceil(n_frames) - 1) * stride + (k - pad_left);
        pad_right = std::max(0, ideal_length - T_in);
    }
    
    if (pad_right > 0) {
        x_pad = torch::constant_pad_nd(x_pad, {0, pad_right}, 0.0);
    }
    
    std::vector<int64_t> stride_vec = {stride};
    std::vector<int64_t> pad_vec = {0};
    std::vector<int64_t> dil_vec = {dilation};
    
    return torch::conv1d(x_pad, weight, bias, stride_vec, pad_vec, dil_vec, groups);
}

at::Tensor VibeVoiceWorker::causal_convtr1d(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, int stride) {
    int k = weight.size(2);
    std::vector<int64_t> stride_vec = {stride};
    at::Tensor y = torch::conv_transpose1d(x, weight, bias, stride_vec);
    
    int padding_total = k - stride;
    int padding_right = std::ceil(padding_total * 1.0f);
    int padding_left = padding_total - padding_right;
    
    if (padding_left + padding_right > 0) {
        int end = y.size(-1) - padding_right;
        y = y.slice(-1, padding_left, end);
    }
    return y;
}

at::Tensor VibeVoiceWorker::dec_block(at::Tensor x, const std::string& bpfx) {
    at::Tensor res = x;
    at::Tensor h = x.transpose(1, 2);
    h = h * torch::rsqrt(h.pow(2).mean(-1, /*keepdim=*/true) + 1e-5) * weights[bpfx + "norm.weight"];
    h = h.transpose(1, 2);
    
    c10::optional<at::Tensor> conv_b = weights.count(bpfx + "mixer.conv.conv.conv.bias") ? c10::make_optional(weights[bpfx + "mixer.conv.conv.conv.bias"]) : c10::nullopt;
    h = causal_conv1d_dec(h, weights[bpfx + "mixer.conv.conv.conv.weight"], conv_b, 1, 1, h.size(1));
    
    if (weights.count(bpfx + "gamma")) {
        h = h * weights[bpfx + "gamma"].view({1, -1, 1});
    }
    h = res + h;
    
    res = h;
    at::Tensor f = h.transpose(1, 2);
    f = f * torch::rsqrt(f.pow(2).mean(-1, /*keepdim=*/true) + 1e-5) * weights[bpfx + "ffn_norm.weight"];
    
    at::Tensor ffn_1_b = weights.count(bpfx + "ffn.linear1.bias") ? weights[bpfx + "ffn.linear1.bias"] : at::Tensor();
    f = torch::nn::functional::linear(f, weights[bpfx + "ffn.linear1.weight"], ffn_1_b);
    
    auto gelu_options = torch::nn::functional::GELUFuncOptions().approximate("tanh");
    f = torch::nn::functional::gelu(f, gelu_options);
    
    at::Tensor ffn_2_b = weights.count(bpfx + "ffn.linear2.bias") ? weights[bpfx + "ffn.linear2.bias"] : at::Tensor();
    f = torch::nn::functional::linear(f, weights[bpfx + "ffn.linear2.weight"], ffn_2_b);
    f = f.transpose(1, 2);
    
    if (weights.count(bpfx + "ffn_gamma")) {
        f = f * weights[bpfx + "ffn_gamma"].view({1, -1, 1});
    }
    
    return res + f;
}

// --- FULL C++ VAE DECODER ---

at::Tensor VibeVoiceWorker::decode_vae(at::Tensor latents) {
    torch::NoGradGuard no_grad;
    
    latents = (latents.to(torch::kFloat32) / weights["vae_scale"] - weights["vae_bias"]).transpose(1, 2);
    
    at::Tensor h = latents;
    std::string pfx = "model.acoustic_tokenizer.decoder.";
    
    c10::optional<at::Tensor> stem_b = weights.count(pfx + "upsample_layers.0.0.conv.conv.bias") ? c10::make_optional(weights[pfx + "upsample_layers.0.0.conv.conv.bias"]) : c10::nullopt;
    h = causal_conv1d_dec(h, weights[pfx + "upsample_layers.0.0.conv.conv.weight"], stem_b, 1, 1, 1);
    
    for (int bi = 0; bi < vae_depths[0]; bi++) {
        h = dec_block(h, pfx + "stages.0." + std::to_string(bi) + ".");
    }
    
    for (int i = 1; i <= num_vae_upsamples; i++) {
        std::string up_pfx = pfx + "upsample_layers." + std::to_string(i) + ".0.convtr.convtr.";
        at::Tensor w = weights[up_pfx + "weight"];
        c10::optional<at::Tensor> b = weights.count(up_pfx + "bias") ? c10::make_optional(weights[up_pfx + "bias"]) : c10::nullopt;
        int stride = w.size(2) / 2;
        
        h = causal_convtr1d(h, w, b, stride);
        
        for (int bi = 0; bi < vae_depths[i]; bi++) {
            h = dec_block(h, pfx + "stages." + std::to_string(i) + "." + std::to_string(bi) + ".");
        }
    }
    
    if (weights.count(pfx + "norm.weight")) {
        h = h.transpose(1, 2);
        h = h * torch::rsqrt(h.pow(2).mean(-1, /*keepdim=*/true) + 1e-6) * weights[pfx + "norm.weight"];
        h = h.transpose(1, 2);
    }
    
    c10::optional<at::Tensor> head_b = weights.count(pfx + "head.conv.conv.bias") ? c10::make_optional(weights[pfx + "head.conv.conv.bias"]) : c10::nullopt;
    h = causal_conv1d_dec(h, weights[pfx + "head.conv.conv.weight"], head_b, 1, 1, 1);
    
    return h.squeeze(0).squeeze(0);
}