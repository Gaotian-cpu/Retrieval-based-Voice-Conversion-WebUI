# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import warnings

# ###########################################################################
# 全局禁用警告
# ###########################################################################
warnings.filterwarnings("ignore")
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MKL_SERVICE_FOR_INTEL"] = "1"
os.environ["KMP_AFFINITY"] = "noverbose"

try:
    import torch
    torch.backends.nnpack.enabled = False
except:
    pass

# ###########################################################################
# 日志
# ###########################################################################
logging.basicConfig(
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RVC-INFER")

# ###########################################################################
# RVC 根目录
# ###########################################################################
RVC_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(RVC_ROOT)

logger.info("=" * 60)
logger.info("✅ RVC Inference Script (Single + Batch)")
logger.info(f"✅ RVC ROOT: {RVC_ROOT}")
logger.info("=" * 60)


# ###########################################################################
# 单音频推理（100% 匹配真实 vc_single 参数）
# ###########################################################################
def infer_single(
        model_path,
        transpose,
        input_audio,
        index_path,
        f0_method,
        resample_sr,
        rms_mix_rate,
        protect,
        filter_radius,
        index_rate
):
    from infer.modules.vc.modules import VC
    from configs.config import Config

    config = Config()
    vc = VC(config)
    vc.get_vc(model_path)

    logger.info("🚀 Start single inference...")
    result_msg, out_tuple = vc.vc_single(
        sid=0,
        input_audio_path=input_audio,
        f0_up_key=transpose,
        f0_file=None,
        f0_method=f0_method,
        file_index=index_path,
        file_index2="",
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect
    )

    if out_tuple is not None and out_tuple[1] is not None:
        logger.info(f"✅ Done: {result_msg}")
    else:
        logger.error(f"❌ Fail: {result_msg}")
    return out_tuple


# ###########################################################################
# 批量推理（100% 匹配真实 vc_multi 参数）
# ###########################################################################
def infer_batch(
        model_path,
        transpose,
        input_dir,
        output_dir,
        index_path,
        f0_method,
        resample_sr,
        rms_mix_rate,
        protect,
        filter_radius,
        index_rate,
        export_format
):
    from infer.modules.vc.modules import VC
    from configs.config import Config

    config = Config()
    vc = VC(config)
    vc.get_vc(model_path)

    logger.info("🚀 Start batch inference...")
    results = vc.vc_multi(
        sid=0,
        dir_path=input_dir,
        opt_root=output_dir,
        paths=None,
        f0_up_key=transpose,
        f0_method=f0_method,
        file_index=index_path,
        file_index2="",
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        format1=export_format
    )

    for msg in results:
        logger.info(msg)
    logger.info("✅ Batch done!")


# ###########################################################################
# 主参数
# ###########################################################################
def main():
    parser = argparse.ArgumentParser(description="RVC Inference (Single + Batch)")
    parser.add_argument("--model_path", required=True, type=str, help="Model .pth path")

    # Mode
    parser.add_argument("--mode", required=True, choices=["single", "batch"], help="single or batch")

    # Single
    parser.add_argument("--transpose", type=int, default=0, help="Pitch shift (12 up/-12 down)")
    parser.add_argument("--audio_path", type=str, help="Single audio path (for single mode)")

    # Batch
    parser.add_argument("--input_dir", type=str, help="Batch input folder")
    parser.add_argument("--output_dir", type=str, help="Batch output folder")
    parser.add_argument("--export_format", type=str, default="wav", choices=["wav", "flac", "mp3", "m4a"])

    # Common
    parser.add_argument("--index_path", type=str, default="", help="Index .index path")
    parser.add_argument("--f0_method", default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe"])
    parser.add_argument("--resample_sr", type=int, default=0, help="0=disable")
    parser.add_argument("--rms_mix_rate", type=float, default=0.25, help="Volume envelope")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect consonants/breath")
    parser.add_argument("--filter_radius", type=int, default=3, help="Median filter for f0")
    parser.add_argument("--index_rate", type=float, default=0.75, help="Search feature ratio")

    args = parser.parse_args()

    if args.mode == "single":
        if not args.audio_path:
            logger.error("❌ --audio_path required in single mode")
            return

        infer_single(
            model_path=args.model_path,
            transpose=args.transpose,
            input_audio=args.audio_path,
            index_path=args.index_path,
            f0_method=args.f0_method,
            resample_sr=args.resample_sr,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            filter_radius=args.filter_radius,
            index_rate=args.index_rate
        )

    elif args.mode == "batch":
        if not args.input_dir or not args.output_dir:
            logger.error("❌ --input_dir --output_dir required in batch mode")
            return

        infer_batch(
            model_path=args.model_path,
            transpose=args.transpose,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            index_path=args.index_path,
            f0_method=args.f0_method,
            resample_sr=args.resample_sr,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            filter_radius=args.filter_radius,
            index_rate=args.index_rate,
            export_format=args.export_format
        )


if __name__ == "__main__":
    main()