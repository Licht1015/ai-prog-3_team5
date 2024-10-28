# # 使用するPythonのベースイメージを指定
# FROM python:3.10-slim

# # 作業ディレクトリを設定
# WORKDIR /app

# # 必要なシステムパッケージをインストール
# RUN apt-get update && apt-get install -y \
#     gcc \
#     libpq-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # requirements.txtをコンテナにコピー
# COPY requirements.txt .

# # Pythonの依存関係をインストール
# RUN pip install --no-cache-dir -r requirements.txt

# # アプリのコードをコンテナにコピー
# COPY . .

# # Streamlitアプリケーションを実行するためのコマンド
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# Anacondaのベースイメージを指定
FROM continuumio/anaconda3

# 作業ディレクトリを設定
WORKDIR /app

# 環境ファイルをコンテナにコピー
COPY environment.yml .

# conda環境を作成して依存関係をインストール
RUN conda env create -f environment.yml

# アプリのコードをコンテナにコピー
COPY . .

# 環境をアクティブにして、アプリケーションを実行する
CMD ["conda", "run", "-n", "health_app", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
