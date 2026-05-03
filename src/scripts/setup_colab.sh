#!/bin/bash

set -e  # остановка при ошибке

echo "===== START SETUP ====="

# ===== 1. Клонирование репозитория =====
if [ -d "text-image-alignment" ]; then
  echo "Repo already exists, pulling updates..."
  cd "text-image-alignment"
  git pull
else
  echo "Cloning repository..."
  git clone https://github.com/trxxnk/text-image-alignment.git
  cd text-image-alignment
fi

echo "Current dir: $(pwd)"

# ===== 2. Установка зависимостей =====
echo "Installing dependencies..."
pip install -r requirements-colab.txt


# ===== 3. Подключение Google Drive =====
echo "Mounting Google Drive..."
python3 - <<EOF
from google.colab import drive
drive.mount('/content/drive')
EOF

# ===== 4. Распаковка датасета =====
DATASET_ZIP="/content/drive/MyDrive/data_generated_v3.tar.gz"
DATASET_DIR="/content/text-image-alignment/data"

echo "Preparing dataset..."

mkdir -p $DATASET_DIR

if [ -f "$DATASET_ZIP" ]; then
  echo "Extracting dataset..."
  tar -xzf $DATASET_ZIP -C $DATASET_DIR
else
  echo "Dataset tar.gz NOT FOUND at $DATASET_ZIP"
fi

# ===== 5. Проверка путей =====
python3 - <<EOF
import os
os.chdir("/content/text-image-alignment/")
from tsp_dewarp.dataset import TPSDataset
print("✅ Imports OK")
EOF

echo "===== SETUP DONE ====="
