# ⚡ ExLlamaV3-VibeVoice

A specialized implementation of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) designed specifically to run the LLM component of the **VibeVoice** Text-to-Speech (TTS) model. 

By leveraging ExLlamaV3's extremely efficient inference engine and EXL3 quantization format, this port drastically reduces VRAM consumption and significantly accelerates audio generation speeds for VibeVoice.

## 📦 Installation

Because this library relies on custom CUDA kernels for maximum performance, you will need a proper C++ build environment to install it.

### Prerequisites
* **Python 3.10+**
* **CUDA Toolkit** (matching your PyTorch installation)
* **Windows:** Visual Studio C++ Build Tools
* **Linux:** `build-essential` or equivalent (`gcc`, `g++`)

### Install via Git

Clone this repository and install it locally:

```bash
git clone https://github.com/DBMePls/exllamav3-vibevoice.git
cd exllamav3-vibevoice

# Install build requirements
pip install -r requirements.txt

# Compile and install the package
pip install -e .
```

*Note: The initial installation may take a few minutes as the custom CUDA kernels need to compile.*

## 🚀 Usage

This repository is designed to act as a high-performance backend dependency. Once installed in your Python environment, it can be imported and utilized by VibeVoice inference APIs and wrappers to handle the LLM generation step before the diffusion process.

*(If you are installing a VibeVoice API or UI that requires this backend, simply ensuring it is installed in your active `venv` or `conda` environment is usually all that is required).*

## 🤝 Credits & Acknowledgements

This repository exists thanks to the incredible foundational work of the open-source community:

* **[Turboderp](https://github.com/turboderp-org/exllamav3)** - For creating the original **ExLlamaV3** engine, a masterpiece of local LLM optimization.
* **[Mozer](https://github.com/Mozer/exllamav3)** - For the brilliant engineering work to port and integrate VibeVoice compatibility into the EXL3 engine. You can also check out their ComfyUI implementation here: **[comfyUI-vibevoice-exl3](https://github.com/mozer/comfyUI-vibevoice-exl3)**.
