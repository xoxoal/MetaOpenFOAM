# MetaOpenFOAM: an LLM-based multi-agent framework for CFD

请选择语言 / Choose your language:

- 中文版： [README_zh.md](README_zh.md)  
- English:   [README_en.md](README_en.md)(default)

# MetaOpenFOAM One‑Click Installation and User Guide

> **Version**: v1.3.0
> **Overview**: Set up the entire development environment, dependencies, and build MetaOpenFOAM with a single script.

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [One‑Click Install](#one-click-install)  
- [Usage](#usage)  
  - [Activate Environment](#activate-environment)  
  - [Activate OpenFOAM](#activate-openfoam)  
  - [Configure Inputs](#configure-inputs)  
  - [Edit Makefile](#edit-makefile)  
  - [First Run](#first-run)  
  - [Run Main Program](#run-main-program)  
- [FAQ](#faq)  
- [Citation](#citation)

---

## Prerequisites

1. **Conda** installed (Miniconda / Anaconda)  
2. **OpenFOAM‑10** installed and sourced (`source $WM_PROJECT_DIR/etc/bashrc`)  
3. Repository root contains:  
   - `environment.yml`  
   - `requirements.txt`  
   - `MetaGPT/` (git source)  
   - `active_subspaces/` (git source)  
   - `MetaOpenFOAM/` (MetaOpenFOAM source)  
   - `install_metaopenfoam.sh` (installation script)

---

## One‑Click Install

```bash
# 1. Grant execute permission
chmod +x install_metaopenfoam.sh

# 2. Run the installer
./install_metaopenfoam.sh
```

This script will:

- Create & activate a Conda env at `./metaopenfoam_env`  
- Install all Python dependencies (including local MetaGPT & active_subspaces)  
- Add `MetaOpenFOAM/` to `PYTHONPATH`  
- Build and compile MetaOpenFOAM  

---

## Usage

### Activate Environment

```bash
conda activate ./metaopenfoam_env
```

### Activate OpenFOAM

```bash
source $WM_PROJECT_DIR/etc/bashrc
```

### Configure Inputs

Edit `inputs/config.yaml` with your case settings:

```yaml
usr_requirement: >-
  do a RANS simulation of buoyantCavity using buoyantFoam, which
  investigates natural convection in a heat cavity with a temperature
  difference of 20K between the hot and cold walls; remaining patches
  are adiabatic. Case name: Buoyant_Cavity

max_loop:    10
temperature: 0.0
batchsize:   10
searchdocs:  2
run_times:   1

MetaGPT_PATH:    "MetaGPT/"
DEEPSEEK_API_KEY: "YOUR_DEEPSEEK_KEY"
DEEPSEEK_BASE_URL:"https://api.deepseek.com"
model:           "deepseek-chat"

# —— Optional: Uncomment for OpenAI model —— 
# OPENAI_API_KEY:    "YOUR_OPENAI_KEY"
# OPENAI_PROXY:      "http://127.0.0.1:8118"
# OPENAI_BASE_URL:   "https://api.openai-proxy.com/v1"
# model:            "gpt-4o"
```

> **Note**: The default implementation uses HuggingFace embeddings for retrieval-augmented generation (RAG), which have not been fully validated and may yield different retrieval performance compared to the OpenAI embeddings used in the original paper. If you want to reproduce the paper’s results more faithfully, switch to the OpenAI embedding for dataset.

### Edit Makefile

In the project root `Makefile`, adjust:

```makefile
# Input case name (matches filename in inputs/, without extension)
Case_input = Buoyant_Cavity
```

### First Run

```bash
make
```

- Initializes the database  
- Builds the project  

### Run Main Program

```bash
make run_main
```

---

## FAQ

- **Script failed halfway—how to retry?**  
  Re-run `./install_metaopenfoam.sh`. It skips completed steps and finishes the rest.

---

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{Chen2024MetaOpenFOAM,
  title={MetaOpenFOAM: an LLM-based multi-agent framework for CFD},
  author={Yuxuan Chen and Xu Zhu and Hua Zhou and Zhuyin Ren},
  journal={Journal Name},
  year={2024},
  doi={http://arxiv.org/abs/2407.21320}
}
@article{Chen2025MetaOpenFOAM2.0,
  title={MetaOpenFOAM 2.0: Large Language Model Driven Chain of Thought for Automating CFD Simulation and Post-Processing},
  author={Yuxuan Chen and Xu Zhu and Hua Zhou and Zhuyin Ren},
  journal={Journal Name},
  year={2025},
  doi={http://arxiv.org/abs/2502.00498}
}

@article{Chen2025OptMetaOpenFOAM,
  title={OptMetaOpenFOAM: Large Language Model Driven Chain of Thought for Sensitivity Analysis and Parameter Optimization based on CFD},
  author={Yuxuan Chen and Long Zhang and Xu Zhu and Hua Zhou and Zhuyin Ren},
  journal={Journal Name},
  year={2025},
  doi={http://arxiv.org/abs/2503.01273}
}
