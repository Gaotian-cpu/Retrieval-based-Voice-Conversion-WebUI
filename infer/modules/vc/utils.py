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
    # 🔥 🔥 🔥 正确拼出项目根目录（绝对不会错）
    file_path = os.path.abspath(__file__)
    # 走到项目根目录：/infer/modules/vc → 退3级
    rvc_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

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
