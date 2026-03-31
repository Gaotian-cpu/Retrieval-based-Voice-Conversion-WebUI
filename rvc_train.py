import os
import sys
import argparse
import subprocess
import logging

# 基础日志配置
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
        subprocess.check_call([sys.executable] + cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"✘ {desc} 执行失败")
        sys.exit(1)
    logger.info(f"✔ {desc} 完成\n")


def main():
    parser = argparse.ArgumentParser(description="RVC 命令行一键训练脚本（对齐 WebUI）")

    # 核心路径（你要的动态目录）
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称")
    parser.add_argument("--log_root", type=str, default="./logs", help="训练根目录")
    parser.add_argument("--save_dir", type=str, default="assets/weights", help="模型保存目录")

    # 数据配置
    parser.add_argument("--dataset_dir", type=str, required=True, help="原始音频文件夹")
    parser.add_argument("--sr", type=str, choices=["32k", "40k", "48k"], required=True, help="采样率")
    parser.add_argument("--f0_method", type=str, default="rmvpe", help="f0提取算法: crepe/pm/dio/harvest/rmvpe")
    parser.add_argument("--if_f0", type=int, default=1, choices=[0,1], help="是否使用F0")

    # 训练超参
    parser.add_argument("--total_epoch", type=int, default=50, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="单卡batch size")
    parser.add_argument("--save_every_epoch", type=int, default=10, help="每隔多少轮保存一次")
    parser.add_argument("--save_every_weights", type=int, default=1, help="是否自动导出轻量模型")
    parser.add_argument("--if_latest", type=int, default=0, help="是否只保存最新模型")
    parser.add_argument("--if_cache_data_in_gpu", type=int, default=1, help="是否缓存数据到GPU")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="模型版本")

    # 预训练模型
    parser.add_argument("--pretrainG", type=str, default="", help="G预训练模型")
    parser.add_argument("--pretrainD", type=str, default="", help="D预训练模型")
    parser.add_argument("--gpus", type=str, default="0", help="使用的GPU，如 0 或 0-1")

    # 流程开关
    parser.add_argument("--skip_process", action="store_true", help="跳过数据处理")
    parser.add_argument("--skip_feature", action="store_true", help="跳过特征提取")
    parser.add_argument("--skip_index", action="store_true", help="跳过索引训练")

    args = parser.parse_args()

    # 路径定义
    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ==============================
    # Step 1: 数据处理（process data）
    # ==============================
    if not args.skip_process:
        run_cmd([
            "infer/modules/train/preprocess/trainset_preprocess_pipeline_print.py",
            args.dataset_dir,
            args.sr,
            exp_dir,
            "1"  # 多线程
        ], "数据处理（切片/重采样）")

    # ==============================
    # Step 2: 提取特征（f0 + hubert）
    # ==============================
    if not args.skip_feature:
        # F0 提取
        run_cmd([
            "infer/modules/train/extract/extract_f0_print.py",
            exp_dir,
            "1",
            args.f0_method
        ], f"提取F0 ({args.f0_method})")

        # Hubert 特征提取
        feature_version = "v2" if args.version == "v2" else "v1"
        run_cmd([
            "infer/modules/train/extract/extract_feature_print.py",
            exp_dir,
            "1",
            "0",
            feature_version,
            exp_dir
        ], f"提取Hubert特征 ({feature_version})")

    # ==============================
    # Step 3: 训练模型
    # ==============================
    logger.info("▶ 开始模型训练")
    train_cmd = [
        "infer/modules/train/train.py",
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

        # 你要的自定义目录（关键扩展）
        "-log_root", args.log_root,
        "-save_dir", args.save_dir,
    ]
    run_cmd(train_cmd, "模型训练")

    # ==============================
    # Step 4: 训练索引（faiss index）
    # ==============================
    if not args.skip_index:
        run_cmd([
            "infer/modules/train/train_index.py",
            exp_dir,
            feature_version
        ], "训练faiss索引")

    logger.info("✅ RVC 一键训练全部完成！")
    logger.info(f"模型已保存到: {os.path.join(args.save_dir, args.exp_name)}.pth")
    logger.info(f"实验目录: {exp_dir}")


if __name__ == "__main__":
    main()
