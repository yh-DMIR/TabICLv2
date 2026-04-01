import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from tabicl import TabICLRegressor 
#from tabicl.sklearn.regressor import TabICLRegressor

# 1. 基础配置
MODEL_PATH = "ckpt/TabICLv2/tabicl-regressor-v2-20260212.ckpt"
RESULT_DIR = "result/TabICLv2"
# 定义需要跑的三个目标文件夹
DATA_DIRS = [
    "dataset/ctr23",
    "dataset/tabarena/reg",
    "dataset/talent_reg"
]

# 确保结果目录存在
os.makedirs(RESULT_DIR, exist_ok=True)

# 2. 设置环境（针对你的 AMD 八卡服务器配置）
# 强制使用指定的临时目录，避免权限或空间问题
os.environ['TMPDIR'] = os.environ.get('TMPDIR', '/tmp') 

def run_benchmark():
    results_list = []
    
    # 初始化模型
    # 使用你下载的本地权重，关闭自动下载
    regressor = TabICLRegressor(
        model_path=MODEL_PATH,
        allow_auto_download=False,
        kv_cache="kv", # 开启缓存以加速多数据集推理
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        verbose=True
    )

    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            print(f"警告: 文件夹不存在 {base_dir}")
            continue

        print(f"\n>>> 正在进入目录: {base_dir}")
        
        # 遍历目录下的所有 CSV 文件
        for file in os.listdir(base_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(base_dir, file)
                print(f"正在处理数据集: {file}...")

                try:
                    # 读取数据
                    df = pd.read_csv(file_path)
                    # 假设最后一列是 Target
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]

                    # 82 分
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # 运行推理
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)

                    # 计算指标
                    rmse = root_mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    results_list.append({
                        "Directory": base_dir,
                        "Dataset": file,
                        "RMSE": rmse,
                        "R2": r2
                    })
                    print(f"完成! R2: {r2:.4f}")

                except Exception as e:
                    print(f"处理 {file} 时出错: {e}")

    # 3. 保存最终汇总结果
    if results_list:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(os.path.join(RESULT_DIR, "all_regression_results.csv"), index=False)
        print(f"\n所有任务完成！结果保存在 {RESULT_DIR}")
        print(final_df)

if __name__ == "__main__":
    run_benchmark()
