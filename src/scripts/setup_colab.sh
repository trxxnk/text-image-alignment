# Setup Colab Notebook

set -e  # остановка при ошибке

echo "===== START SETUP COLAB ====="

# ===== 1. Установка зависимостей =====
echo "[1/2] Installing dependencies..."
pip install -q -r requirements-colab.txt

# ===== 2. Распаковка датасета =====
echo "[2/2] Download & Unzip dataset..."
DATASET_ZIP="/content/drive/MyDrive/data_generated_v3.tar.gz"
DATASET_DIR="./data/generated"

mkdir -p "$DATASET_DIR"

if [ -f "$DATASET_ZIP" ]; then
  echo "Extracting dataset..."
  tar -xzf $DATASET_ZIP -C $DATASET_DIR
  echo "Success: Dataset extracted to $DATASET_DIR"
else
  echo "!!! ERROR: Dataset tar.gz NOT FOUND at $DATASET_ZIP"
fi

echo "===== SETUP COLAB DONE ====="
