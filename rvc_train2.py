import os
import sys
import argparse
import subprocess
import logging

# ==================== 日志配置（清晰美观）====================
logging.basicConfig(
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RVC-Train")

# ==================== 自动识别 RVC 根目录（无需传参）====================
RVC_ROOT = os.path.dirname(os.path.abspath(__file__))
logger.info(f"============================================================")
logger.info(f"✅ RVC 根目录自动识别: {RVC_ROOT}")
logger.info(f"============================================================\n")

# ==================== 统一执行函数：失败立即退出 + 完整日志 ====================
def run_step(script_path, args, step_name):
    """
    执行一步训练：
    - 必须成功才能继续
    - 失败立即终止整个程序
    - 完整日志
    """
    logger.info(f"🚀 开始执行: {step_name}")

    # 拼接绝对路径命令
    cmd = [sys.executable, script_path] + args

    try:
        # 执行
        subprocess.check_call(cmd, cwd=RVC_ROOT)  # 强制在 RVC_ROOT 执行，不污染外部

        # 成功
        logger.info(f"✅ {step_name} 执行成功\n")
        return True

    except subprocess.CalledProcessError:
        logger.error(f"❌ {step_name} 执行失败！进程终止！")
        sys.exit(1)

# ==================== 主训练流程 ====================
def main():
    parser = argparse.ArgumentParser(description="RVC 官方结构 - 干净命令行训练（失败即终止）")

    # 核心参数
    parser.add_argument("--exp_name", required=True, type=str, help="实验名称")
    parser.add_argument("--dataset_dir", required=True, type=str, help="原始音频目录")
    parser.add_argument("--sr", required=True, choices=["32k", "40k", "48k"], help="采样率")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"], help="模型版本")
    parser.add_argument("--total_epoch", type=int, default=50, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--if_f0", type=int, default=1, help="是否使用F0")

    # 自定义输出目录（不污染 RVC）
    parser.add_argument("--log_root", required=True, type=str, help="训练中间文件根目录")
    parser.add_argument("--save_dir", required=True, type=str, help="模型最终保存目录")

    # 可选参数
    parser.add_argument("--f0_method", default="rmvpe", type=str)
    parser.add_argument("--num_process", default="1", type=str)
    parser.add_argument("--save_every_epoch", default=10, type=int)
    parser.add_argument("--save_every_weights", default=1, type=int)
    parser.add_argument("--if_latest", default=0, type=int)
    parser.add_argument("--if_cache_data_in_gpu", default=1, type=int)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--pretrainG", default="", type=str)
    parser.add_argument("--pretrainD", default="", type=str)

    # 跳过开关
    parser.add_argument("--skip_process", action="store_true", help="跳过数据处理")
    parser.add_argument("--skip_feature", action="store_true", help="跳过特征提取")
    parser.add_argument("--skip_index", action="store_true", help="跳过索引训练")

    args = parser.parse_args()

    # 实验目录
    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"📂 实验工作目录: {exp_dir}")
    logger.info(f"📂 模型保存目录: {args.save_dir}\n")

    # ==========================
    # 官方真实脚本路径（100%正确）
    # ==========================
    PREPROCESS  = os.path.join(RVC_ROOT, "infer/modules/train/preprocess.py")
    EXTRACT_F0  = os.path.join(RVC_ROOT, "infer/modules/train/extract_f0.py")
    EXTRACT_FEAT= os.path.join(RVC_ROOT, "infer/modules/train/extract_feature_print.py")
    TRAIN       = os.path.join(RVC_ROOT, "infer/modules/train/train.py")
    TRAIN_INDEX = os.path.join(RVC_ROOT, "infer/modules/train/train_index.py")

    # ==========================
    # 步骤 1：数据处理
    # ==========================
    if not args.skip_process:
        run_step(
            PREPROCESS,
            [args.dataset_dir, args.sr, exp_dir, args.num_process, "False"],
            "数据预处理（切片/重采样/归一化）"
        )

    # ==========================
    # 步骤 2：特征提取（F0 + Hubert）
    # ==========================
    if not args.skip_feature:
        run_step(
            EXTRACT_F0,
            [exp_dir, args.num_process, args.f0_method],
            "F0 音高提取"
        )
        run_step(
            EXTRACT_FEAT,
            [exp_dir, args.num_process, "0", args.version, exp_dir],
            "Hubert 特征提取"
        )

    # ==========================
    # 步骤 3：模型训练（必须成功）
    # ==========================
    run_step(
        TRAIN,
        [
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
        "模型训练（核心步骤）"
    )

    # ==========================
    # 步骤 4：索引训练
    # ==========================
    if not args.skip_index:
        run_step(
            TRAIN_INDEX,
            [exp_dir, args.version],
            "索引训练"
        )

    # ==========================
    # 全部成功
    # ==========================
    logger.info("")
    logger.info("============================================================")
    logger.info("🎉 所有步骤全部执行成功！训练完成！")
    logger.info(f"📦 模型输出: {args.save_dir}")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
