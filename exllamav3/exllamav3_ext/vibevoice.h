#ifndef VIBEVOICE_H
#define VIBEVOICE_H

#include <torch/extension.h>
#include <vector>

class VibeVoiceDiffusionWorker {
public:
    // Diffusion Head Weights
    at::Tensor noisy_proj_w;
    at::Tensor cond_proj_w;
    at::Tensor t_embed_lin1_w;
    at::Tensor t_embed_lin2_w;
    at::Tensor final_proj_w;
    at::Tensor final_adaln_w;

    struct Layer {
        at::Tensor norm_w;
        at::Tensor ffn_gate_w;
        at::Tensor ffn_up_w;
        at::Tensor ffn_down_w;
        at::Tensor adaln_w;
    };
    std::vector<Layer> layers;

    // Acoustic Connector Weights
    at::Tensor ac_fc1_w;
    at::Tensor ac_fc1_b;
    at::Tensor ac_norm_w;
    at::Tensor ac_fc2_w;
    at::Tensor ac_fc2_b;

    // DPM Solver Precomputed States
    at::Tensor alpha_t;
    at::Tensor sigma_t;
    at::Tensor lambda_t;
    std::vector<int> timesteps;

    float eps;

    VibeVoiceDiffusionWorker(
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
    );

    at::Tensor get_timestep_embedding(at::Tensor t, int dim, torch::Device device, torch::ScalarType dtype);
    at::Tensor rms_norm_torch(const at::Tensor& x, const at::Tensor& weight, float eps);
    at::Tensor rms_norm_no_weight(const at::Tensor& x, float eps);
    
    at::Tensor head_forward(at::Tensor noisy, at::Tensor cond, float t_val);
    at::Tensor sample(at::Tensor cond, at::Tensor cond_neg, float cfg_scale, bool use_cfg, bool increase_cfg);
    at::Tensor acoustic_connector_forward(at::Tensor latent);
};

#endif // EXT_VIBEVOICE_H