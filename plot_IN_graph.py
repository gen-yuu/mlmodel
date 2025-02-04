import subprocess
import glob
import os

# `visualize_source` ディレクトリのパスを取得
IN_dir = os.path.join(os.getcwd(), "IN_visualize_source")

# ディレクトリ内のすべての .py ファイルを取得
py_files = glob.glob(os.path.join(IN_dir, "*.py"))

# 環境変数を設定して警告を抑制
env = os.environ.copy()
env["PYTHONWARNINGS"] = "ignore"

# 各ファイルを `visualize_source` に移動して実行
for py_file in py_files:
    print(f"Executing: {py_file}")
    subprocess.run(["python", os.path.basename(py_file)], cwd=IN_dir)  # `visualize_source` に移動して実行
