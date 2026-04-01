import os

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    # 🔥🔥🔥 【从环境变量获取 RVC 根目录，绝对路径】🔥🔥🔥
    rvc_root = os.environ.get("RVC_WEBUI_ROOT")

    hubert_path = os.path.join(rvc_root, "assets/hubert/hubert_base.pt")

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],  # <--- 修复
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
