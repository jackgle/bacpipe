from fairseq import checkpoint_utils
import numpy as np
import torch
from bacpipe.model_specific_utils.animal2vec_nn.nn import chunk_and_normalize
from ..utils import ModelBaseClass
import yaml
from pathlib import Path
import importlib.resources as pkg_resources


SAMPLE_RATE = 8000
LENGTH_IN_SAMPLES = int(10 * SAMPLE_RATE)
# Use importlib.resources for settings.yaml
try:
    with pkg_resources.path("bacpipe", "settings.yaml") as settings_path:
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
except Exception:
    settings = {}


class Model(ModelBaseClass):
    def __init__(self, xeno_canto=False):
        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES)
        self.batch_size = 8
        if xeno_canto:
            from . import animal2vec_xc as A2VXC

            path_to_pt_file = A2VXC.PATH_TO_PT_FILE
            self.sr = A2VXC.SAMPLE_RATE
            self.segment_length = A2VXC.LENGTH_IN_SAMPLES
        else:
            # Use package-relative path for model checkpoint
            path_to_pt_file = (
                Path(__file__).parent.parent.parent.parent
                / "model_specific_utils"
                / "animal2vec_nn"
                / "nn"
                / "animal2vec_mk"
                / "animal2vec_large_finetuned_MeerKAT_240507.pt"
            )
        models, _ = checkpoint_utils.load_model_ensemble(
            [str(path_to_pt_file)]
        )
        self.model = models[0].to(self.device)
        self.model.eval()

    def preprocess(self, audio):
        return torch.stack(
            [torch.nn.functional.layer_norm(x, x.shape).squeeze() for x in audio]
        )

    @torch.inference_mode()
    def __call__(self, batch):
        res = self.model(source=batch, mode="AUDIO", features_only=True)
        embeds = res["layer_results"]
        return (
            torch.stack(embeds).mean(0).mean(1)
        )  # mean over attention heads and segment length
