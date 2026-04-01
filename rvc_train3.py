# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import logging
import json
import pathlib
import random

# ###########################################################################
# 🔥 🔥 🔥 永久屏蔽 NNPACK 警告（和 WebUI 完全一样）
# ###########################################################################
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["PYTORCH_WARNINGS"] = "0"

# 强行禁用 NNPACK 核心警告
try:
    import torch
    torch.backends.nnpack.enabled = False
except:
    pass
# ###########################################################################

# ==================== 日志 ====================
logging.basicConfig(
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RVC-Train")

# ==================== 自动识别RVC根目录 ====================
RVC_ROOT = os.path.dirname(os.path.abspath(__file__))
logger.info("============================================================")
logger.info(f"✅ RVC 根目录自动识别: {RVC_ROOT}")
logger.info("============================================================\n")

# ==================== 执行步骤：失败立即停止 ====================
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
    parser = argparse.ArgumentParser(description="RVC 完整训练脚本（和WebUI完全一致）")

    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--sr", required=True, choices=["32k", "40k", "48k"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--total_epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--if_f0", default=1, type=int)

    parser.add_argument("--log_root", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)

    parser.add_argument("--f0_method", default="rmvpe", type=str, choices=["pm", "harvest", "dio", "rmvpe"])
    parser.add_argument("--num_process", default="1", type=str)
    parser.add_argument("--save_every_epoch", default=10, type=int)
    parser.add_argument("--save_every_weights", default=0, type=int)
    parser.add_argument("--if_latest", default=0, type=int)
    parser.add_argument("--if_cache_data_in_gpu", default=0, type=int)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--pretrainG", default="", type=str)
    parser.add_argument("--pretrainD", default="", type=str)

    args = parser.parse_args()

    ###########################################################################
    # 🔥 🔥 🔥 【新增：WebUI 自动匹配预训练模型 G/D 路径】100% 官方行为
    ###########################################################################
    logger.info("🔎 自动匹配预训练模型路径...")

    pretrained_dir = os.path.join(RVC_ROOT, "assets", "pretrained")
    pretrained_v2_dir = os.path.join(RVC_ROOT, "assets", "pretrained_v2")

    if args.version == "v1":
        if args.sr == "48k":
            args.pretrainG = os.path.join(pretrained_dir, "G_48k.pth")
            args.pretrainD = os.path.join(pretrained_dir, "D_48k.pth")
        elif args.sr == "40k":
            args.pretrainG = os.path.join(pretrained_dir, "G_40k.pth")
            args.pretrainD = os.path.join(pretrained_dir, "D_40k.pth")
        elif args.sr == "32k":
            args.pretrainG = os.path.join(pretrained_dir, "G_32k.pth")
            args.pretrainD = os.path.join(pretrained_dir, "D_32k.pth")
    else:
        if args.sr == "48k":
            args.pretrainG = os.path.join(pretrained_v2_dir, "f0G48k.pth")
            args.pretrainD = os.path.join(pretrained_v2_dir, "f0D48k.pth")
        elif args.sr == "40k":
            args.pretrainG = os.path.join(pretrained_v2_dir, "f0G40k.pth")
            args.pretrainD = os.path.join(pretrained_v2_dir, "f0D40k.pth")
        elif args.sr == "32k":
            args.pretrainG = os.path.join(pretrained_v2_dir, "f0G32k.pth")
            args.pretrainD = os.path.join(pretrained_v2_dir, "f0D32k.pth")

    logger.info(f"✅ 预训练G: {args.pretrainG}")
    logger.info(f"✅ 预训练D: {args.pretrainD}")
    ###########################################################################

    sr_num = args.sr.replace("k", "000")
    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"📂 实验目录: {exp_dir}")
    logger.info(f"📂 模型输出: {args.save_dir}\n")

    PREPROCESS = os.path.join(RVC_ROOT, "infer/modules/train/preprocess.py")
    EXTRACT_F0 = os.path.join(RVC_ROOT, "infer/modules/train/extract/extract_f0_print.py")
    EXTRACT_FEATURE = os.path.join(RVC_ROOT, "infer/modules/train/extract_feature_print.py")
    TRAIN = os.path.join(RVC_ROOT, "infer/modules/train/train.py")
    TRAIN_INDEX = os.path.join(RVC_ROOT, "infer/modules/train/train_index.py")

    # === 1 数据预处理 ===
    run_step(
        PREPROCESS,
        [args.dataset_dir, sr_num, args.num_process, exp_dir, "False", "0.99"],
        "数据预处理"
    )

    # === 2 F0 提取 ===
    run_step(
        EXTRACT_F0,
        [exp_dir, args.num_process, args.f0_method],
        "F0 音高提取"
    )

    # === 3 特征提取 ===
    run_step(
        EXTRACT_FEATURE,
        ["cuda", "1", "0", exp_dir, args.version, "False"],
        "Hubert 特征提取"
    )

    ###########################################################################
    # 🔥 ✅ ✅ ✅ 【官方原版】生成 filelist.txt（100% 从 WebUI 复制】
    ###########################################################################
    logger.info("🚀 生成官方格式 filelist.txt")
    try:
        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        feature_dir = os.path.join(exp_dir, "3_feature768")
        f0_dir = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
        spk_id = 0

        names = (
                set([n.split(".")[0] for n in os.listdir(gt_wavs_dir)])
                & set([n.split(".")[0] for n in os.listdir(feature_dir)])
                & set([n.split(".")[0] for n in os.listdir(f0_dir)])
                & set([n.split(".")[0] for n in os.listdir(f0nsf_dir)])
        )

        opt = []
        for name in names:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )

        # 加入2条静音（必须，否则训练不执行）
        for _ in range(2):
            opt.append(
                f"{RVC_ROOT}/logs/mute/0_gt_wavs/mute48k.wav|{RVC_ROOT}/logs/mute/3_feature768/mute.npy|{RVC_ROOT}/logs/mute/2a_f0/mute.wav.npy|{RVC_ROOT}/logs/mute/2b-f0nsf/mute.wav.npy|0"
            )

        random.shuffle(opt)
        filelist_path = os.path.join(exp_dir, "filelist.txt")
        with open(filelist_path, "w", encoding="utf-8") as f:
            f.write("\n".join(opt))

        logger.info(f"✅ filelist 生成完成，共 {len(opt)} 条数据\n")
    except:
        logger.exception("❌ 生成 filelist 失败")
        sys.exit(1)

    ###########################################################################
    # 🔥 生成 config.json（官方逻辑）
    ###########################################################################
    logger.info("🚀 生成 config.json")
    try:
        from configs.config import Config
        config = Config()
        if args.version == "v1":
            config_path = f"v1/{args.sr}.json"
        else:
            config_path = f"v2/{args.sr}.json"

        config_save_path = os.path.join(exp_dir, "config.json")
        if not pathlib.Path(config_save_path).exists():
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(config.json_config[config_path], f, ensure_ascii=False, indent=4)
        logger.info("✅ config 生成成功\n")
    except:
        logger.exception("❌ 生成 config 失败")
        sys.exit(1)

    # === 4 模型训练 ===
    run_step(
        TRAIN,
        [
            "-e", args.exp_name,
            "-sr", args.sr,
            "-v", args.version,
            "-g", args.gpus,
            "-bs", str(args.batch_size),
            "-te", str(args.total_epoch),
            "-se", str(args.save_every_epoch),
            "-f0", str(args.if_f0),
            "-l", str(args.if_latest),
            "-c", str(args.if_cache_data_in_gpu),
            "-sw", str(args.save_every_weights),
            "-pg", args.pretrainG,
            "-pd", args.pretrainD,
            "-log_root", args.log_root,
            "-save_dir", args.save_dir,
        ],
        "模型训练"
    )

    # === 5 索引生成 ===
    run_step(
        TRAIN_INDEX,
        [exp_dir, args.version],
        "索引生成"
    )

    logger.info("============================================================")
    logger.info("🎉 训练全部完成！完全对齐 WebUI！")
    logger.info("============================================================")

if __name__ == "__main__":
    main()