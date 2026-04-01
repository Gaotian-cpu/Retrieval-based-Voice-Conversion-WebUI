# ######################### 安全环境配置（仅关闭NNPACK）#########################
import os
import sys
import json
import logging

# 环境变量（等效sh里的export，不会屏蔽错误）
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ##############################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--sr", required=True, type=str, choices=["32k", "40k", "48k"])
    parser.add_argument("--version", default="v2", type=str, choices=["v1", "v2"])
    parser.add_argument("--f0_method", default="rmvpe", type=str)
    parser.add_argument("--num_process", default="1", type=str)
    parser.add_argument("--total_epoch", default=20, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--save_epoch", default=5, type=int)
    parser.add_argument("--if_f0", default=1, type=int)
    parser.add_argument("--if_latest", default=0, type=int)
    parser.add_argument("--if_cache_data_in_gpu", default=0, type=int)
    parser.add_argument("--if_save_every_weights", default=0, type=int)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--pretrained_G", default="", type=str)
    parser.add_argument("--pretrained_D", default="", type=str)
    parser.add_argument("--log_root", default="output_rvc/my_logs", type=str)
    parser.add_argument("--save_dir", default="output_rvc/weights", type=str)
    args = parser.parse_args()

    ###########################################################################
    # ✅ ✅ ✅ 【关键修复】强制设置为 RVC 根目录！！！
    ###########################################################################
    RVC_ROOT = "/root/Project/RVC-WebUI"  # 强制写死你的RVC真实路径
    os.chdir(RVC_ROOT)  # 直接切换工作目录
    sys.path.append(RVC_ROOT)
    ###########################################################################

    exp_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ================== 工具函数：执行子进程 ==================
    def run_process(name, cmd_list):
        logger.info(f"🚀 开始: {name}")
        logger.info(f"命令: {' '.join(cmd_list)}")

        p = os.spawnvp(os.P_WAIT, cmd_list[0], cmd_list)

        if p != 0:
            logger.error(f"❌ 失败: {name}")
            sys.exit(1)
        logger.info(f"✅ 完成: {name}\n")

    # ================== 步骤1：预处理 ==================
    sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
    actual_sr = sr_dict[args.sr]

    cmd_pre = [
        sys.executable, "infer/modules/train/preprocess.py",
        args.dataset_dir, str(actual_sr), args.num_process,
        exp_dir, "False", "0.9"
    ]
    run_process("数据预处理", cmd_pre)

    # ================== 步骤2：提取F0 ==================
    cmd_f0 = [
        sys.executable, "infer/modules/train/extract/extract_f0_print.py",
        exp_dir, args.num_process, args.f0_method
    ]
    run_process("F0提取", cmd_f0)

    # ================== 步骤3：提取Hubert特征 ==================
    cmd_feat = [
        sys.executable, "infer/modules/train/extract_feature_print.py",
        "cuda", "1", "0", exp_dir, args.version, "False"
    ]
    run_process("Hubert特征提取", cmd_feat)

    # ================== 步骤4：【官方原版】生成config.json ==================
    try:
        from configs.config import Config
        config = Config()

        if args.version == "v1" or args.sr == "40k":
            config_path = f"v1/{args.sr}.json"
        else:
            config_path = f"v2/{args.sr}.json"

        config_save_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_save_path):
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(config.json_config[config_path], f, ensure_ascii=False, indent=4)
        logger.info("✅ 生成 config.json 完成\n")
    except Exception as e:
        logger.error(f"❌ 生成config失败: {e}")
        sys.exit(1)

    # ================== 步骤5：训练模型 ==================
    cmd_train = [
        sys.executable, "infer/modules/train/train.py",
        "-e", args.exp_name,
        "-sr", args.sr,
        "-f0", str(args.if_f0),
        "-bs", str(args.batch_size),
        "-g", args.gpus,
        "-te", str(args.total_epoch),
        "-se", str(args.save_epoch),
        "-l", str(args.if_latest),
        "-c", str(args.if_cache_data_in_gpu),
        "-sw", str(args.if_save_every_weights),
        "-v", args.version,
        "-log_root", args.log_root,
        "-save_dir", args.save_dir,
    ]
    if args.pretrained_G:
        cmd_train += ["-pg", args.pretrained_G]
    if args.pretrained_D:
        cmd_train += ["-pd", args.pretrained_D]

    run_process("模型训练", cmd_train)

    logger.info("🎉 全部训练流程完成！")

if __name__ == "__main__":
    main()