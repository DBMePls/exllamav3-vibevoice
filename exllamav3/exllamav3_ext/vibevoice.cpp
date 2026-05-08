#include "vibevoice.h"
#include <cmath>
#include <limits>

VibeVoiceDiffusionWorker::VibeVoiceDiffusionWorker(
    at::Tensor noisy_proj_w, at::Tensor cond_proj_w, 
    at::Tensor t_embed_lin1_w, at::Tensor t_embed_lin2_w,
    at::Tensor final_proj_w, at::Tensor final_adaln_w,
    std::vector<at::Tensor> layer_norm_w,
    std::vector<at::Tensor> layer_ffn_gate_w,
    std::vector<at::Tensor> layer_ffn_up_w,
    std::vector<at::Tensor> layer_ffn_down_w,
    std::vector<at::Tensor> layer_adaln_w,
    at::Tensor ac_fc1_w, at::Tensor ac_fc1_b, at::Tensor ac_norm_w,
    at::Tensor ac_fc2_w, at::Tensor ac_fc2_b,
    at::Tensor alpha_t, at::Tensor sigma_t, at::Tensor lambda_t,
    std::vector<int> timesteps, float eps
) : noisy_proj_w(noisy_proj_w), cond_proj_w(cond_proj_w), 
    t_embed_lin1_w(t_embed_lin1_w), t_embed_lin2_w(t_embed_lin2_w),
    final_proj_w(final_proj_w), final_adaln_w(final_adaln_w),
    ac_fc1_w(ac_fc1_w), ac_fc1_b(ac_fc1_b), ac_norm_w(ac_norm_w),
    ac_fc2_w(ac_fc2_w), ac_fc2_b(ac_fc2_b),
    alpha_t(alpha_t), sigma_t(sigma_t), lambda_t(lambda_t),
    timesteps(timesteps), eps(eps) 
{
    for (size_t i = 0; i < layer_norm_w.size(); ++i) {
        layers.push_back({
            layer_norm_w[i], layer_ffn_gate_w[i], layer_ffn_up_w[i], 
            layer_ffn_down_w[i], layer_adaln_w[i]
        });
    }
}

at::Tensor VibeVoiceDiffusionWorker::get_timestep_embedding(at::Tensor t, int dim, torch::Device device, torch::ScalarType dtype) {
    int half = dim / 2;
    auto freqs = torch::exp(-std::log(10000.0f) * torch::arange(0, half, torch::TensorOptions().dtype(torch::kFloat32).device(device)) / half);
    auto args = t.unsqueeze(1).to(torch::kFloat32) * freqs.unsqueeze(0);
    auto emb = torch::cat({torch::cos(args), torch::sin(args)}, -1);
    if (dim % 2) {
        emb = torch::cat({emb, torch::zeros({t.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device))}, -1);
    }
    return emb.to(dtype);
}

at::Tensor VibeVoiceDiffusionWorker::rms_norm_torch(const at::Tensor& x, const at::Tensor& weight, float eps) {
    at::Tensor variance = x.pow(2).mean(-1, /*keepdim=*/true);
    return weight * (x * at::rsqrt(variance + eps));
}

at::Tensor VibeVoiceDiffusionWorker::rms_norm_no_weight(const at::Tensor& x, float eps) {
    at::Tensor variance = x.pow(2).mean(-1, /*keepdim=*/true);
    return x * at::rsqrt(variance + eps);
}

at::Tensor VibeVoiceDiffusionWorker::head_forward(at::Tensor noisy, at::Tensor cond, float t_val) {
    int B = noisy.size(0);
    at::Tensor t_tensor = torch::full({B}, t_val, torch::TensorOptions().device(noisy.device()).dtype(torch::kFloat32));
    at::Tensor t_sin = get_timestep_embedding(t_tensor, 256, noisy.device(), noisy.scalar_type());

    at::Tensor x = at::matmul(noisy, noisy_proj_w.t());

    at::Tensor t_emb = at::matmul(t_sin, t_embed_lin1_w.t());
    t_emb = at::silu(t_emb);
    t_emb = at::matmul(t_emb, t_embed_lin2_w.t());

    at::Tensor c = at::matmul(cond, cond_proj_w.t()) + t_emb;

    for (const auto& l : layers) {
        at::Tensor m = at::matmul(at::silu(c), l.adaln_w.t());
        auto chunks = m.chunk(3, -1);
        at::Tensor shift = chunks[0], scale = chunks[1], gate = chunks[2];

        at::Tensor xn = rms_norm_torch(x, l.norm_w, eps);
        at::Tensor mod = xn * (1.0f + scale) + shift;

        at::Tensor gate_out = at::matmul(mod, l.ffn_gate_w.t());
        at::Tensor up_out = at::matmul(mod, l.ffn_up_w.t());
        at::Tensor act = at::silu(gate_out) * up_out;
        at::Tensor f = at::matmul(act, l.ffn_down_w.t());

        x = x + gate * f;
    }

    at::Tensor m_final = at::matmul(at::silu(c), final_adaln_w.t());
    auto chunks_final = m_final.chunk(2, -1);
    at::Tensor shift_f = chunks_final[0], scale_f = chunks_final[1];

    at::Tensor xn_f = rms_norm_no_weight(x, eps);
    at::Tensor mod_f = xn_f * (1.0f + scale_f) + shift_f;
    return at::matmul(mod_f, final_proj_w.t());
}

at::Tensor VibeVoiceDiffusionWorker::acoustic_connector_forward(at::Tensor latent) {
    torch::NoGradGuard no_grad;
    at::Tensor h = at::matmul(latent, ac_fc1_w.t()) + ac_fc1_b;
    h = rms_norm_torch(h, ac_norm_w, 1e-6);
    h = at::matmul(h, ac_fc2_w.t()) + ac_fc2_b;
    return h;
}

at::Tensor VibeVoiceDiffusionWorker::sample(at::Tensor cond, at::Tensor cond_neg, float cfg_scale, bool use_cfg, bool increase_cfg) {
    torch::NoGradGuard no_grad; // Prevents backprop graphs from leaking memory
    
    int M = timesteps.size() - 1;
    int B = cond.size(0);
    
    // Ensure we are working with 2D tensors [B, D]
    if (cond.dim() == 3) cond = cond.squeeze(1);
    if (use_cfg && cond_neg.dim() == 3) cond_neg = cond_neg.squeeze(1);

    at::Tensor x = torch::randn({B, 64}, cond.options()); 
    at::Tensor prev_x0;

    for (int i = 0; i < M; ++i) {
        int t = timesteps[i];
        int s = timesteps[i + 1];

        float a_t = alpha_t[t].item<float>();
        float sg_t = sigma_t[t].item<float>();
        float l_t = lambda_t[t].item<float>();
        float l_prev = (i > 0) ? lambda_t[timesteps[i - 1]].item<float>() : 0.0f;

        float a_s, sg_s, l_s;
        if (s == -1) {
            a_s = 1.0f; sg_s = 0.0f; l_s = std::numeric_limits<float>::infinity();
        } else {
            a_s = alpha_t[s].item<float>();
            sg_s = sigma_t[s].item<float>();
            l_s = lambda_t[s].item<float>();
        }

        at::Tensor v;
        if (use_cfg) {
            float current_cfg = cfg_scale;
            if (increase_cfg && ((float)i / M) < 0.5f) {
                current_cfg = cfg_scale * 1.5f; 
            }
            
            // Batched CFG: we process positive and negative conditions in the same pass!
            at::Tensor x_double = torch::cat({x, x}, 0);
            at::Tensor cond_double = torch::cat({cond, cond_neg}, 0);
            
            at::Tensor v_double = head_forward(x_double, cond_double, (float)t);
            
            auto v_chunks = v_double.chunk(2, 0);
            at::Tensor v_pos = v_chunks[0];
            at::Tensor v_neg = v_chunks[1];
            
            v = v_neg + current_cfg * (v_pos - v_neg);
        } else {
            v = head_forward(x, cond, (float)t);
        }

        at::Tensor x0 = a_t * x - sg_t * v;

        bool is_first = (i == 0);
        bool is_last = (i == M - 1);
        int order = (is_first || is_last) ? 1 : 2; 

        if (s == -1) {
            x = x0;
        } else if (order == 1) {
            float h = l_s - l_t;
            float A = sg_s / sg_t;
            float B_val = a_s * (std::exp(-h) - 1.0f);
            x = A * x - B_val * x0;
        } else {
            float h = l_s - l_t;
            float h_0 = l_t - l_prev;
            float r0 = h_0 / h;
            float A = sg_s / sg_t;
            float B_val = a_s * (std::exp(-h) - 1.0f);
            float Bd1 = 0.5f * B_val;

            at::Tensor D0 = x0;
            at::Tensor D1 = (1.0f / r0) * (x0 - prev_x0);
            x = A * x - B_val * D0 - Bd1 * D1;
        }
        prev_x0 = x0.clone();
    }
    // Return with shape [B, 1, 64] as expected by ExLlama Python loop
    return x.unsqueeze(1); 
}