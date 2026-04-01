import os
import shutil
from huggingface_hub import hf_hub_download

# 你指定的监控目录
target_dir = "/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/lyh/TabICLv2/ckpt/TabICLv2"
os.makedirs(target_dir, exist_ok=True)

# 根据你提供的文档截图，精确的 v2 版本文件名
repo_id = "jingang/TabICL"  # 官方 Hugging Face 仓库名
ckpts = [
    "tabicl-classifier-v2-20260212.ckpt",
    "tabicl-regressor-v2-20260212.ckpt"
]

print(f"开始下载官方权重到: {target_dir}")

for ckpt in ckpts:
    print(f"\n正在下载: {ckpt} ...")
    try:
        # 如果你的服务器需要代理，请确保在运行此脚本前在终端 export 了 http_proxy 等环境变量
        cache_path = hf_hub_download(repo_id=repo_id, filename=ckpt)
        
        # 将缓存文件复制到你的监控目录中，以便 ckpt_wait1_v2.sh 能检测到
        final_path = os.path.join(target_dir, ckpt)
        shutil.copy(cache_path, final_path)
        print(f"✅ 成功! 已保存至: {final_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
