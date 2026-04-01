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

# ==========================
# 自动识别 RVC 根目录（核心！）
# ==========================
RVC_ROOT = os.path.dirname(os.path.abspath(__file__))
logger.info(f"✅ 自动识别 RVC 根目录: {RVC_ROOT}")

def run_cmd(cmd, desc=""):
    logger.info(f"▶ {desc}")
    try:
        subprocess.check_call(cmd)
        logger.info(f"✔ {desc} 完成\n")
    except subprocess.CalledProcessError:
        logger.error(f"✘ {desc} 失败")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="RVC 干净训练脚本 - 自动路径、无污染")

    # 训练参数
    parser.add_argument("--exp_name", required=True, type=str, help="实验名称")
    parser.add_argument("--dataset_dir", required=True, type=str, help="数据集路径")
    parser.add_argument("--sr", required=True, choices=["32k", "40k", "48k"], help="采样率")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"], help="模型版本")
    parser.add_argument("--total_epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--if_f0", default=1, type=int)

    # 输出目录（你完全自定义，不会污染 RVC）
    parser.add_argument("--log_root", required=True, type=str, help="训练中间文件目录")
    parser.add_argument("--save_dir", required=True, type=str, help="模型保存目录")

    # 可选
    parser.add_argument("--f0_method", default="rmvpe", type=str)
    parser.add_argument("--save_every_epoch", default=10, type=int)
    parser.add_argument("--save_every_weights", default=1, type=int)
    parser.add_argument("--if_latest", default=0, type=int)
    parser.add_argument("--if_cache_data_in_gpu", default=1, type=int)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--pretrainG", default="", type=str)
    parser.add_argument("--pretrainD", default="", type=str)

    # 跳过开关
    parser.add_argument("--skip_process", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_index", action="store_true")

    args = parser.parse_args()

    # 实验目录
    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ==========================
    # 内部脚本路径（自动拼接，绝对正确）
    # ==========================
    preprocess = os.path.join(RVC_ROOT, "infer/modules/train/preprocess/trainset_preprocess_pipeline_print.py")
    extract_f0 = os.path.join(RVC_ROOT, "infer/modules/train/extract/extract_f0_print.py")
    extract_feat = os.path.join(RVC_ROOT, "infer/modules/train/extract/extract_feature_print.py")
    train = os.path.join(RVC_ROOT, "infer/modules/train/train.py")
    train_index = os.path.join(RVC_ROOT, "infer/modules/train/train_index.py")

    # ==========================
    # 1. 数据处理
    # ==========================
    if not args.skip_process:
        run_cmd(
            [sys.executable, preprocess, args.dataset_dir, args.sr, exp_dir, "1"],
            "数据预处理"
        )

    # ==========================
    # 2. 特征提取
    # ==========================
    if not args.skip_feature:
        run_cmd(
            [sys.executable, extract_f0, exp_dir, "1", args.f0_method],
            "F0 音高提取"
        )
        run_cmd(
            [sys.executable, extract_feat, exp_dir, "1", "0", args.version, exp_dir],
            "Hubert 特征提取"
        )

    # ==========================
    # 3. 模型训练
    # ==========================
    run_cmd(
        [
            sys.executable, train,
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
        ],
        "模型训练"
    )

    # ==========================
    # 4. 索引训练
    # ==========================
    if not args.skip_index:
        run_cmd(
            [sys.executable, train_index, exp_dir, args.version],
            "索引生成"
        )

    logger.info("🎉 所有训练完成！RVC 目录完全无污染！")
    logger.info(f"📂 中间文件: {exp_dir}")
    logger.info(f"📂 最终模型: {args.save_dir}")

if __name__ == "__main__":
    main()
