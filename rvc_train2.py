import os
import sys
import argparse
import subprocess
import logging

# ==================== 【关键】屏蔽 NNPACK 警告（和 WebUI 一样）====================
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ================================================================================

# ==================== 日志配置 ====================
logging.basicConfig(
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RVC-Train")

# ==================== 自动识别 RVC 根目录 ====================
RVC_ROOT = os.path.dirname(os.path.abspath(__file__))
logger.info("============================================================")
logger.info(f"✅ RVC 根目录自动识别: {RVC_ROOT}")
logger.info("============================================================\n")

# ==================== 步骤执行：失败立即终止 + 日志 ====================
def run_step(script_path, args, step_name):
    logger.info(f"🚀 开始执行: {step_name}")
    cmd = [sys.executable, script_path] + args

    try:
        subprocess.check_call(cmd, cwd=RVC_ROOT)
        logger.info(f"✅ {step_name} 执行成功\n")
    except subprocess.CalledProcessError:
        logger.error(f"❌ {step_name} 执行失败！进程终止！")
        sys.exit(1)

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="RVC 官方最终完美训练脚本")

    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--sr", required=True, choices=["32k", "40k", "48k"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--total_epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--if_f0", default=1, type=int)

    parser.add_argument("--log_root", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)

    parser.add_argument("--f0_method", default="rmvpe_gpu", type=str,
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"])
    parser.add_argument("--num_process", default="1", type=str)
    parser.add_argument("--save_every_epoch", default=10, type=int)
    parser.add_argument("--save_every_weights", default=1, type=int)
    parser.add_argument("--if_latest", default=0, type=int)
    parser.add_argument("--if_cache_data_in_gpu", default=1, type=int)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--pretrainG", default="", type=str)
    parser.add_argument("--pretrainD", default="", type=str)

    parser.add_argument("--skip_process", action="store_true")
    parser.add_argument("--skip_feature", action="store_true")
    parser.add_argument("--skip_index", action="store_true")

    args = parser.parse_args()

    # 转换采样率 48k → 48000
    sr_num = args.sr.replace("k", "000")

    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"📂 实验工作目录: {exp_dir}")
    logger.info(f"📂 模型保存目录: {args.save_dir}\n")

    # ==========================
    # 你本地真实路径（完全按你说的）
    # ==========================
    PREPROCESS = os.path.join(RVC_ROOT, "infer/modules/train/preprocess.py")
    EXTRACT_F0 = os.path.join(RVC_ROOT, "infer/modules/train/extract/extract_f0_print.py")
    EXTRACT_FEAT = os.path.join(RVC_ROOT, "infer/modules/train/extract/extract_feature_print.py")
    TRAIN = os.path.join(RVC_ROOT, "infer/modules/train/train.py")
    TRAIN_INDEX = os.path.join(RVC_ROOT, "infer/modules/train/train_index.py")

    # 1. 数据预处理
    if not args.skip_process:
        run_step(
            PREPROCESS,
            [
                args.dataset_dir,
                sr_num,
                args.num_process,
                exp_dir,
                "False",
                "0.99",
            ],
            "数据预处理（切片/重采样/归一化）"
        )

    # 2. 特征提取
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

    # 3. 模型训练
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

    # 4. 索引训练
    if not args.skip_index:
        run_step(
            TRAIN_INDEX,
            [exp_dir, args.version],
            "索引训练"
        )

    logger.info("")
    logger.info("============================================================")
    logger.info("🎉 所有步骤全部成功！训练完成！")
    logger.info(f"📦 模型输出: {args.save_dir}")
    logger.info("============================================================")

if __name__ == "__main__":
    main()