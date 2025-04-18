以下提供两个独立的 README：  

- **README_zh.md**：中文版本  
- **README_en.md**：英文版本  

---

```markdown
<!-- README_zh.md -->

# MetaOpenFOAM 一键安装指南

> **版本**：2025‑04‑18  
> **简介**：通过一条脚本命令完成环境创建、依赖安装与源码编译，快速启动 MetaOpenFOAM 开发与运行。

---

## 目录

- [前置条件](#前置条件)  
- [一键安装](#一键安装)  
- [后续使用](#后续使用)  
  - [激活环境](#激活环境)  
  - [激活 OpenFOAM](#激活-openfoam)  
  - [配置输入文件](#配置输入文件)  
  - [修改 Makefile](#修改-makefile)  
  - [首次运行](#首次运行)  
  - [运行主程序](#运行主程序)  
- [常见问题](#常见问题)  
- [贡献 & 支持](#贡献--支持)

---

## 前置条件

1. 已安装 **Conda**（Miniconda / Anaconda）  
2. 已安装并配置 **OpenFOAM‑10**（可通过 apt / 源码 / Docker，确认 `source $WM_PROJECT_DIR/etc/bashrc` 成功）  
3. 仓库根目录包含：  
   - `environment.yml`  
   - `requirements.txt`  
   - `MetaGPT/` （本地源码）  
   - `active_subspaces/` （本地源码）  
   - `MetaOpenFOAM/` （MetaOpenFOAM 源码）  
   - `install_metaopenfoam.sh` （安装脚本）

---

## 一键安装

```bash
# 授予脚本执行权限
chmod +x install_metaopenfoam.sh

# 运行安装脚本
./install_metaopenfoam.sh
```

脚本将自动完成：

- 在当前目录下创建并激活 Conda 环境（`./metaopenfoam_env`）  
- 安装全部 Python 依赖（含本地 MetaGPT 与 active_subspaces）  
- 将 `MetaOpenFOAM/` 路径加入 `PYTHONPATH`  
- 编译并构建 MetaOpenFOAM  

---

## 后续使用

### 激活环境

```bash
conda activate ./metaopenfoam_env
```

### 激活 OpenFOAM

```bash
source $WM_PROJECT_DIR/etc/bashrc
```

### 配置输入文件

在 `inputs/config.yaml` 中填入示例或自定义配置：

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

> **说明**：  
> - 支持 `openai` 与 `deepseek` 两种模型，详情请见官网。  
> - 默认使用 HuggingFace Embedding；如需论文中 OpenAI Embedding，可切换到 OpenAI 模型。

### 修改 Makefile

打开根目录下 `Makefile`，修改：

```makefile
# Python 解释器（可替换为 python3 等）
PYTHON     = python

# 输入案例名称（对应 inputs/config.yaml 中的配置，不含扩展名）
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
  重新执行 `./install_metaopenfoam.sh`，脚本会跳过已完成步骤并继续剩余操作。  

---

## 贡献 & 支持

欢迎提交 Issues 或 Pull Requests！  
---  
© 2025 MetaOpenFOAM 项目维护团队
```

---
