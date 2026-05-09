#pragma once
#include <torch/extension.h>
#include <vector>
#include <map>
#include <string>

struct DiTWorkspace {
    at::Tensor h_x, t_emb, t_emb2, c, c_silu, adaln_out, final_adaln_out;
    at::Tensor mod_out, gate_out_param, ffn_gate, ffn_up, f, out, dummy_norm;
};

class VibeVoiceWorker {
public:
    std::map<std::string, at::Tensor> weights;
    std::vector<int> timesteps;
    at::Tensor alpha_t, sigma_t, lambda_t;
    int num_layers;
    int num_vae_upsamples;
    std::vector<int> vae_depths;

    VibeVoiceWorker(
        std::map<std::string, at::Tensor> weights,
        std::vector<int> timesteps,
        at::Tensor alpha_t, at::Tensor sigma_t, at::Tensor lambda_t,
        int num_layers, int num_vae_upsamples, std::vector<int> vae_depths
    );

    void head_forward(at::Tensor x, at::Tensor cond, float t_val, DiTWorkspace& ws);
    at::Tensor sample_latent(at::Tensor cond_pos, at::Tensor cond_neg, float cfg_scale, int seed, bool increase_cfg);
    
    // --- VAE & Audio Processors ---
    at::Tensor encode_acoustic(at::Tensor audio);
    at::Tensor acoustic_connector_forward(at::Tensor z);
    at::Tensor decode_vae(at::Tensor latents);
    
private:
    at::Tensor causal_conv1d_enc(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, int stride, int dilation, int groups);
    at::Tensor enc_block(at::Tensor x, const std::string& bpfx);
    
    at::Tensor causal_conv1d_dec(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, int stride, int dilation, int groups);
    at::Tensor causal_convtr1d(const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, int stride);
    at::Tensor dec_block(at::Tensor x, const std::string& bpfx);
};