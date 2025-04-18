```markdown
# MetaOpenFOAM 一键安装及使用指南

> **版本**：v1.3.0  
> **概述**：通过单个脚本完成开发环境搭建、依赖安装与源码编译，快速运行 MetaOpenFOAM。

---

## 目录

- [前置条件](#前置条件)  
- [一键安装](#一键安装)  
- [使用说明](#使用说明)  
  - [激活环境](#激活环境)  
  - [激活 OpenFOAM](#激活-openfoam)  
  - [配置输入](#配置输入)  
  - [编辑 Makefile](#编辑-makefile)  
  - [首次运行](#首次运行)  
  - [运行主程序](#运行主程序)  
- [常见问题](#常见问题)  
- [引用格式](#引用格式)

---

## 前置条件

1. 已安装 **Conda**（Miniconda / Anaconda）  
2. 已安装并 `source $WM_PROJECT_DIR/etc/bashrc` 生效的 **OpenFOAM‑10**  
3. 仓库根目录包含以下文件／目录：  
   - `environment.yml`  
   - `requirements.txt`  
   - `MetaGPT/`（本地源码）  
   - `active_subspaces/`（本地源码）  
   - `MetaOpenFOAM/`（MetaOpenFOAM 源码）  
   - `install_metaopenfoam.sh`（安装脚本）

---

## 一键安装

```bash
# 1. 授予脚本执行权限
chmod +x install_metaopenfoam.sh

# 2. 运行安装脚本
./install_metaopenfoam.sh
```

脚本将自动完成：

- 在当前目录下创建并激活 Conda 环境（`./metaopenfoam_env`）  
- 安装所有 Python 依赖（包括本地 MetaGPT 与 active_subspaces）  
- 将 `MetaOpenFOAM/` 目录加入 `PYTHONPATH`  
- 构建并编译 MetaOpenFOAM  

---

## 使用说明

### 激活环境

```bash
conda activate ./metaopenfoam_env
```

### 激活 OpenFOAM

```bash
source $WM_PROJECT_DIR/etc/bashrc
```

### 配置输入

编辑 `inputs/config.yaml`，填写你的算例设置，例如：

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

# —— 可选：使用 OpenAI 模型时取消注释 —— 
# OPENAI_API_KEY:    "YOUR_OPENAI_KEY"
# OPENAI_PROXY:      "http://127.0.0.1:8118"
# OPENAI_BASE_URL:   "https://api.openai-proxy.com/v1"
# model:            "gpt-4o"
```

> **注意**：  
> 默认使用 HuggingFace Embedding（未完全验证），其检索性能可能与论文中使用的 OpenAI Embedding 存在差异。如需与论文结果一致，请切换到 OpenAI 模型并配置对应的 API Key。

### 编辑 Makefile

打开项目根目录下的 `Makefile`，修改以下内容：

```makefile
# 输入案例名称（对应 inputs/ 目录下的文件名，不含扩展名）
Case_input = Buoyant_Cavity
```

### 首次运行

```bash
make
```

- 初始化数据库  
- 构建项目  

### 运行主程序

```bash
make run_main
```

---

## 常见问题

- **脚本中途失败，如何重试？**  
  重新运行 `./install_metaopenfoam.sh`，脚本会自动跳过已完成的步骤，继续安装剩余内容。

---

## 引用格式

如果本工作对您的研究有帮助，请引用：

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
```
```
