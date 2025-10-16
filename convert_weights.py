import torch
from safetensors.torch import load_file
import os


def convert_safetensors_to_pth(source_path, target_dir, target_filename):
    """
    加载 .safetensors 文件并将其保存为 .pth 文件。
    """
    print(f"--- 正在加载 .safetensors 文件: {source_path} ---")

    # 确保源文件存在
    if not os.path.exists(source_path):
        print(f"[错误] 源文件未找到: {source_path}")
        print("请确认你已经下载了 model.safetensors 文件并正确设置了路径。")
        return

    # 加载 safetensors 权重
    state_dict = load_file(source_path)

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, target_filename)

    print(f"--- 正在将权重保存为 .pth 格式: {target_path} ---")

    # 保存为 pth 文件
    torch.save(state_dict, target_path)

    print("--- 转换完成！ ---")
    print(f"现在你可以重新运行你的训练命令了。")


if __name__ == '__main__':
    # --- 用户需要修改的部分 ---

    # 1. 设置你下载的 model.safetensors 文件的路径
    #    例如: "C:/Users/unirz/Downloads/model.safetensors"
    #    请使用正斜杠 '/' 或者双反斜杠 '\\'
    source_path = "C:/Users/unirz/Downloads/model.safetensors"

    # --- 以下部分通常不需要修改 ---

    # 2. torch.hub 期望的目标路径
    user_home = os.path.expanduser("~")
    target_dir = os.path.join(user_home, ".cache", "torch", "hub", "checkpoints")

    # 3. torch.hub 期望的目标文件名
    target_filename = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    convert_safetensors_to_pth(source_path, target_dir, target_filename)