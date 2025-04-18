以下提供两个独立的 README：  

- **README_zh.md**：中文版本  
- **README_en.md**：英文版本  

---

```markdown
<!-- README_zh.md -->

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

---
