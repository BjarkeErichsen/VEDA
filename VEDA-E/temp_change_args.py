import pickle
import glob
import os
import argparse

def update_diffusion_type():
    # 获取所有匹配的 pickle 文件
    file_paths = glob.glob("outputs/*/args.pickle")
    
    for file_path in file_paths:
        try:
            # 读取 pickle 文件
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            # 处理 argparse.Namespace 类型
            if isinstance(data, argparse.Namespace):
                if getattr(data, "probabilistic_model", None) == "consistency":
                    setattr(data, "probabilistic_model", "elucidating_dm")
                    
                    # 保存修改后的 pickle 文件
                    with open(file_path, "wb") as f:
                        pickle.dump(data, f)
                    print(f"Updated: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    update_diffusion_type()
