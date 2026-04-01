# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("rvc_train")


def run_cmd(cmd, desc=""):
    logger.info(f"▶ {desc}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"✘ {desc} 执行失败")
        sys.exit(1)
    logger.info(f"✔ {desc} 完成\n")


def main():
    parser = argparse.ArgumentParser(description="RVC 命令行训练脚本（绝对路径模式，不污染 RVC 目录）")

    # 关键：RVC 根目录（固定不变）
    parser.add_argument("--rvc_root", type=str, required=True, help="RVC-WebUI 绝对路径")

    # 你的自定义目录（完全隔离）
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称")
    parser.add_argument("--dataset_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--log_root", type=str, required=True, help="训练日志/中间文件目录")
    parser.add_argument("--save_dir", type=str, required=True, help="最终模型保存目录")

    # 训练参数
    parser.add_argument("--sr", type=str, choices=["32k", "40k", "48k"], required=True)
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--if_f0", type=int, default=1)
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--total_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_every_epoch", type=int, default=10)
    parser.add_argument("--save_every_weights", type=int, default=1)
    parser.add_argument("--if_latest", type=int, default=0)
    parser.add_argument("--if_cache_data_in_gpu", type=int, default=1)
    parser.add_argument("--pretrainG", type=str, default="")
    parser.add_argument("--pretrainD", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")

    # 跳过开关
    parser.add_argument("--skip_process", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_index", action="store_true")

    args = parser.parse_args()

    # 实验目录 = 你指定的目录，和 RVC 无关
    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ==============================
    # 所有脚本路径 = 绝对路径（从 rvc_root 拼接）
    # ==============================
    script_preprocess = os.path.join(args.rvc_root, "infer/modules/train/preprocess/trainset_preprocess_pipeline_print.py")
    script_extract_f0 = os.path.join(args.rvc_root, "infer/modules/train/extract/extract_f0_print.py")
    script_extract_feat = os.path.join(args.rvc_root, "infer/modules/train/extract/extract_feature_print.py")
    script_train = os.path.join(args.rvc_root, "infer/modules/train/train.py")
    script_index = os.path.join(args.rvc_root, "infer/modules/train/train_index.py")

    # ==============================
    # Step 1 数据处理
    # ==============================
    if not args.skip_process:
        run_cmd([
            sys.executable, script_preprocess,
            args.dataset_dir,
            args.sr,
            exp_dir,
            "1"
        ], "数据处理（切片/重采样）")

    # ==============================
    # Step 2 特征提取
    # ==============================
    if not args.skip_feature:
        run_cmd([
            sys.executable, script_extract_f0,
            exp_dir, "1", args.f0_method
        ], f"F0 提取 ({args.f0_method})")

        run_cmd([
            sys.executable, script_extract_feat,
            exp_dir, "1", "0", args.version, exp_dir
        ], "Hubert 特征提取")

    # ==============================
    # Step 3 模型训练
    # ==============================
    run_cmd([
        sys.executable, script_train,
        "-se", str(args.save_every_epoch),
        "-te", str(args.total_epoch),
        "-pg", args.pretrainG,
        "-pd", args.pretrainD,
        "-g", args.gpus,
        "-bs", str(args.batch_size),
        "-e", args.exp_name,
        "-sr", args.sr,
        "-sw", str(args.save_every_weights),
        "-v", args.version,
        "-f0", str(args.if_f0),
        "-l", str(args.if_latest),
        "-c", str(args.if_cache_data_in_gpu),
        "-log_root", args.log_root,
        "-save_dir", args.save_dir,
    ], "模型训练")

    # ==============================
    # Step 4 索引
    # ==============================
    if not args.skip_index:
        run_cmd([
            sys.executable, script_index,
            exp_dir, args.version
        ], "训练索引")

    logger.info("✅ 训练完全结束！无任何文件写入 RVC-WebUI 目录！")
    logger.info(f"实验目录：{exp_dir}")
    logger.info(f"模型保存：{args.save_dir}")


if __name__ == "__main__":
    main()

