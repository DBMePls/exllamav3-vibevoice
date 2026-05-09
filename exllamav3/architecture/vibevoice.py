from typing_extensions import override
import torch
import math
import os
import json
from ..model.config import Config, no_default
from .qwen2 import Qwen2Model, Qwen2Config
from ..ext import exllamav3_ext as ext
from ..loader.safetensors import SafetensorsCollection

class VibeVoiceConfig(Qwen2Config):
    # Hijack the standard Qwen2 architecture string so ExLlamaV3 automatically routes to us!
    arch_string = "Qwen2ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, **kwargs)
        # Ensure we route to our custom model class
        self.model_classes = {"text": VibeVoiceModel}
        self.ddpm_num_steps = 1000
        self.ddpm_inference_steps = 20

def betas_for_alpha_bar(num_diffusion_timesteps):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        a1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
        a2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1 - a2 / a1, 0.999))
    return torch.tensor(betas, dtype=torch.float32)

class VibeVoiceModel(Qwen2Model):
    config_class = VibeVoiceConfig

    def __init__(self, config: VibeVoiceConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.worker = None

    @override
    def load(self, *args, **kwargs):
        # Extract the diffusion path before passing to ExLlamaV3's base loader
        diffusion_model_path = kwargs.pop("diffusion_model_path", None)
        
        # Load the base LLM weights using ExLlamaV3's native auto-split loader
        super().load(*args, **kwargs)
        
        if not diffusion_model_path:
            raise ValueError("VibeVoiceModel requires diffusion_model_path in load() kwargs!")
            
        print(f" -- Loading VibeVoice DiT and VAE weights from {diffusion_model_path} into C++ Worker")
        
        device = self.output_device if self.output_device is not None else torch.device("cuda:0")
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        
        dit_stc = SafetensorsCollection(diffusion_model_path)
        weights = {}
        
        def _get(k): 
            return dit_stc.get_tensor(k, device=device, allow_bf16=True).float()
            
        def _get_linear(k):
            return _get(k).T.contiguous().half()
        
        weights["noisy_proj"] = _get_linear("model.prediction_head.noisy_images_proj.weight")
        weights["cond_proj"] = _get_linear("model.prediction_head.cond_proj.weight")
        weights["t_mlp0"] = _get_linear("model.prediction_head.t_embedder.mlp.0.weight")
        weights["t_mlp2"] = _get_linear("model.prediction_head.t_embedder.mlp.2.weight")
        weights["final_adaln"] = _get_linear("model.prediction_head.final_layer.adaLN_modulation.1.weight")
        weights["final_proj"] = _get_linear("model.prediction_head.final_layer.linear.weight")
        
        num_head_layers = 4
        for i in range(num_head_layers):
            pfx = f"model.prediction_head.layers.{i}."
            weights[f"layer_{i}_adaln"] = _get_linear(pfx + "adaLN_modulation.1.weight")
            weights[f"layer_{i}_norm"] = _get(pfx + "norm.weight").half()
            weights[f"layer_{i}_gate"] = _get_linear(pfx + "ffn.gate_proj.weight")
            weights[f"layer_{i}_up"] = _get_linear(pfx + "ffn.up_proj.weight")
            weights[f"layer_{i}_down"] = _get_linear(pfx + "ffn.down_proj.weight")

        if "model.speech_scaling_factor" in dit_stc.tensor_file_map:
            weights["vae_scale"] = _get("model.speech_scaling_factor")
        if "model.speech_bias_factor" in dit_stc.tensor_file_map:
            weights["vae_bias"] = _get("model.speech_bias_factor")

        # --- NEW: Dynamically Load ALL VAE Encoder & Decoder Weights using their True Keys ---
        prefixes = [
            "model.acoustic_tokenizer.encoder.", 
            "model.acoustic_tokenizer.decoder.", 
            "model.acoustic_connector.", 
        ]
        for k in dit_stc.tensor_file_map.keys():
            if any(k.startswith(pfx) for pfx in prefixes):
                weights[k] = _get(k)
        
        with open(os.path.join(diffusion_model_path, "config.json"), "r", encoding="utf-8") as f:
            full_config = json.load(f)
        
        decoder_cfg = full_config.get("acoustic_tokenizer_config", {})
        decoder_ratios = decoder_cfg.get("decoder_ratios") or decoder_cfg.get("encoder_ratios", [8,5,5,4,2,2])
        decoder_depths_str = decoder_cfg.get("decoder_depths") or decoder_cfg.get("encoder_depths", "3-3-3-3-3-3-8")
        
        if isinstance(decoder_depths_str, str):
            encoder_depths = [int(x) for x in decoder_depths_str.split("-")]
        else:
            encoder_depths = decoder_depths_str
            
        vae_depths = list(reversed(encoder_depths))
        num_vae_upsamples = len(decoder_ratios)
        
        dit_stc.close()

        # Setup DPM Solver Arrays
        betas = betas_for_alpha_bar(self.config.ddpm_num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alpha_t = torch.sqrt(alphas_cumprod).to(device)
        sigma_t = torch.sqrt(1 - alphas_cumprod).to(device)
        lambda_t = (torch.log(alpha_t) - torch.log(sigma_t)).to(device)
        
        timesteps = torch.linspace(0, self.config.ddpm_num_steps - 1, self.config.ddpm_inference_steps + 1).round().int().tolist()[::-1]
        timesteps[-1] = -1

        # Instantiate C++ Worker
        self.worker = ext.VibeVoiceWorker(
            weights, timesteps, alpha_t, sigma_t, lambda_t, num_head_layers, num_vae_upsamples, vae_depths
        )