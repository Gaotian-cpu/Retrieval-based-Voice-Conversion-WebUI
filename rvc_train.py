# -*- coding: utf-8 -*-
"""
一键训练的cli脚本
"""
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# 初始化路径
now_dir = os.getcwd()
sys.path.append(now_dir)

def run_command(cmd, description):
    """安全运行子进程命令"""
    print(f"\n========================================")
    print(f"正在执行：{description}")
    print(f"命令：{' '.join(cmd)}")
    print(f"========================================")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(f"错误：{description} 执行失败！")
        sys.exit(1)

def generate_training_filelist(exp_dir, use_f0=True):
    """生成训练所需的 filelist.txt"""
    print("\n[+] 生成训练文件列表...")
    wav_dir = os.path.join(exp_dir, "0_gt_wavs")
    feat_dir = os.path.join(exp_dir, "3_feature768")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    filelist_path = os.path.join(exp_dir, "filelist.txt")

    with open(filelist_path, "w", encoding="utf-8") as f:
        for name in sorted(os.listdir(wav_dir)):
            if not name.endswith(".wav"):
                continue
            wav_path = os.path.join(wav_dir, name).replace("\\", "/")
            feat_path = os.path.join(feat_dir, name.replace(".wav", ".npy")).replace("\\", "/")

            if use_f0:
                f0_path = os.path.join(f0_dir, name.replace(".wav", ".npy")).replace("\\", "/")
                f.write(f"{wav_path}|{feat_path}|{f0_path}|0\n")
            else:
                f.write(f"{wav_path}|{feat_path}|0\n")
    return filelist_path

def generate_train_config(exp_dir, sr, epoch, batch_size, use_f0, gpus, cache_data):
    """生成训练配置文件"""
    version = "v2"
    config_template = f"configs/{version}/{sr}.json"

    with open(config_template, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 覆盖关键配置
    config["model_dir"] = exp_dir
    config["data"]["training_files"] = os.path.join(exp_dir, "filelist.txt")
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epoch
    config["gpus"] = gpus
    config["if_f0"] = 1 if use_f0 else 0
    config["sample_rate"] = int(sr.replace("k", "000"))
    config["version"] = version
    config["if_cache_data_in_gpu"] = 1 if cache_data else 0
    config["pretrainG"] = f"assets/pretrained_{version}/f0G{sr}.pth"
    config["pretrainD"] = f"assets/pretrained_{version}/f0D{sr}.pth"
    config["save_every_epoch"] = 5
    config["if_latest"] = 0
    config["save_every_weights"] = "1"
    config["total_epoch"] = epoch

    # 保存临时配置
    tmp_config = os.path.join(exp_dir, "train_config.json")
    with open(tmp_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return tmp_config


def main():
    parser = argparse.ArgumentParser(description="RVC 纯命令行一键训练脚本 | 复刻 infer-web.py 逻辑")

    # 必选参数
    parser.add_argument("--train_dir", required=True, help="训练音频文件夹路径")
    parser.add_argument("--exp", required=True, help="实验名称（输出文件夹名）")

    # 可选参数
    parser.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"], help="采样率")
    parser.add_argument("--epoch", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--no_f0", action="store_true", help="禁用 F0（无法变调）")
    parser.add_argument("--f0_method", default="rmvpe", choices=["pm", "harvest", "dio", "rmvpe"], help="F0提取算法")
    parser.add_argument("--num_workers", type=int, default=4, help="预处理进程数")
    parser.add_argument("--gpus", default="0", help="使用的GPU，例如 0 或 0,1")
    parser.add_argument("--no_cache", action="store_true", help="不缓存数据到GPU")

    args = parser.parse_args()

    # 基础设置
    exp_dir = os.path.join("logs", args.exp)
    use_f0 = not args.no_f0
    py_cmd = sys.executable
    device = "cuda"

    # 打印信息
    print("=" * 60)
    print("           RVC CLI 一键训练工具")
    print("=" * 60)
    print(f"训练音频目录：{args.train_dir}")
    print(f"实验名称     ：{args.exp}")
    print(f"采样率       ：{args.sr}")
    print(f"训练轮数     ：{args.epoch}")
    print(f"批次大小     ：{args.batch_size}")
    print(f"使用F0       ：{use_f0}")
    print(f"F0提取算法   ：{args.f0_method}")
    print(f"输出目录     ：{exp_dir}")
    print("=" * 60)

    # ===================== 1. 音频预处理 =====================
    run_command([
        py_cmd, "infer/modules/train/preprocess.py",
        args.train_dir, args.sr.replace("k", "000"), str(args.num_workers),
        exp_dir, "False", "3.7"
    ], "音频预处理（切片、降噪、重采样）")

    # ===================== 2. 提取 F0 =====================
    if use_f0:
        if args.f0_method == "rmvpe":
            # 多GPU RMVPE
            gpu_list = args.gpus.split(",")
            n_gpus = len(gpu_list)
            for i, gpu_id in enumerate(gpu_list):
                run_command([
                    py_cmd, "infer/modules/train/extract/extract_f0_rmvpe.py",
                    str(n_gpus), str(i), gpu_id, exp_dir, "False"
                ], f"GPU {gpu_id} - RMVPE F0 提取")
        else:
            run_command([
                py_cmd, "infer/modules/train/extract/extract_f0_print.py",
                exp_dir, str(args.num_workers), args.f0_method
            ], f"F0 提取 ({args.f0_method})")

    # ===================== 3. 提取 Hubert 特征 =====================
    gpu_list = args.gpus.split(",")
    n_gpus = len(gpu_list)
    for i, gpu_id in enumerate(gpu_list):
        run_command([
            py_cmd, "infer/modules/train/extract/extract_feature_print.py",
            device, str(n_gpus), str(i), gpu_id, exp_dir, "v2", "False"
        ], f"GPU {gpu_id} - Hubert 特征提取")

    # ===================== 4. 生成训练列表 =====================
    generate_training_filelist(exp_dir, use_f0)

    # ===================== 5. 生成训练配置 =====================
    train_config = generate_train_config(
        exp_dir, args.sr, args.epoch, args.batch_size,
        use_f0, args.gpus, not args.no_cache
    )

    # ===================== 6. 开始训练 =====================
    run_command([
        py_cmd, "infer/modules/train/train.py",
        "-c", train_config
    ], "模型训练")

    # ===================== 完成 =====================
    print("\n🎉 训练全部完成！")
    print(f"模型文件保存在：{exp_dir}")


if __name__ == "__main__":
    main()
