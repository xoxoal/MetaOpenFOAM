#!/usr/bin/env bash
set -e

# —— 可根据需要改名 ——  
ENV_DIR=./metaopenfoam_env

# —— 1. 创建或跳过环境 ——  
if [ ! -d "$ENV_DIR" ]; then
  echo "🔧 [Step 1] 创建 conda 环境到 ${ENV_DIR} ..."
  conda env create -p "$ENV_DIR" -f environment.yml
else
  echo "⚡ 环境目录已存在，跳过创建"
fi

# —— 2. 激活环境 ——  
# 注意要先 source conda.sh，才能 activate 路径环境
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "🔑 激活环境：conda activate ${ENV_DIR}"
conda activate "$ENV_DIR"

# —— 3. 安装或更新依赖 ——  
echo "📦 [Step 2] 安装公共依赖（requirements.txt）..."
pip install --upgrade -r requirements.txt

echo "📦 [Step 3] 可编辑安装本地包 MetaGPT & active_subspaces..."
pip install --upgrade -e ./MetaGPT
pip install --upgrade -e ./active_subspaces


echo "✅ 安装完成！"
echo "   下次只需进入项目根目录，运行："
echo "     conda activate $ENV_DIR"
