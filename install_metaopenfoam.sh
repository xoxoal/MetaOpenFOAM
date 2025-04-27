#!/usr/bin/env bash
set -e

ENV_DIR=./metaopenfoam_env

# 1) 创建或跳过环境
if [ ! -d "$ENV_DIR" ]; then
  echo "🔧 [Step 1] 创建 conda 环境到 ${ENV_DIR} ..."
  conda env create -p "$ENV_DIR" -f environment.yml
else
  echo "⚡ 环境目录已存在，跳过创建"
fi

# 2) 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3) 拉取本地包源码
if [ ! -d "MetaGPT" ]; then
  echo "📥 [Step 2] 克隆 MetaGPT 源码 ..."
  git clone --depth 1 https://github.com/geekan/MetaGPT.git
else
  echo "📥 MetaGPT 已存在，跳过"
fi

if [ ! -d "active_subspaces" ]; then
  echo "📥 [Step 3] 克隆 active_subspaces 源码 ..."
  git clone --depth 1 https://github.com/paulcon/active_subspaces.git
else
  echo "📥 active_subspaces 已存在，跳过"
fi

# 4) 安装依赖
echo "📦 [Step 4] 安装公共依赖..."
pip install --upgrade -r requirements.txt

echo "📦 [Step 5] 安装源码包..."
pip install --upgrade -e ./MetaGPT
pip install --upgrade -e ./active_subspaces

echo "✅ 安装完成！下次只需：conda activate $ENV_DIR"
