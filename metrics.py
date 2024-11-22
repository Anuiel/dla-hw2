import json
import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate
from tqdm import tqdm
import torch
import torchaudio

from src.metrics.tracker import MetricTracker

warnings.filterwarnings("ignore", category=UserWarning)


def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    if path is None:
        return None
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    """
    Script for easy CER/WER evaluation over existing predictions and ground truth

    Args:
        predictions_dir: Directory with predicted text
        ground_truth_dir: Directory with ground truth text
    """
    predictions_dir = Path(config.predictions_dir)
    ground_truth_dir = Path(config.ground_truth_dir)

    if not predictions_dir.exists() or not predictions_dir.is_dir():
        print(
            f"Prediction dir {config.predictions_dir} is not existing or not a directory"
        )
        return

    if not ground_truth_dir.exists() or not ground_truth_dir.is_dir():
        print(
            f"Ground truth dir {config.ground_truth_dir} is not existing or not a directory"
        )
        return
    metrics = instantiate(config.metrics)
    metric_tracker = MetricTracker(
        *([m.name for m in metrics["inference"]] + ["PIT_SI-SNR"]),
        writer=None,
    )
    for pred_path in tqdm((predictions_dir / "s1").iterdir()):
        name = pred_path.name
        mix_path = ground_truth_dir / "audio" / "mix" / name
        sp1_target_path = ground_truth_dir / "audio" / "s1" / name
        sp2_target_path = ground_truth_dir / "audio" / "s2" / name
        sp1_pred_path = predictions_dir / "s1" / name
        sp2_pred_path = predictions_dir / "s2" / name

        for file in (mix_path, sp1_target_path, sp2_target_path, sp1_pred_path, sp1_pred_path):
            assert file.exists() and file.is_file(), f"{file} does not exist or not a file"
        
        mix_audio = load_audio(mix_path)
        targets = torch.cat([
            load_audio(path)
            for path in (sp1_target_path, sp2_target_path)
        ]).unsqueeze(0)
        preds = torch.cat([
            load_audio(path)
            for path in (sp1_pred_path, sp2_pred_path)
        ]).unsqueeze(0)

        for met in metrics["inference"]:
            metric_tracker.update(met.name, met(mix_audio=mix_audio, targets=targets, preds=preds))

    print(json.dumps(metric_tracker.result(), indent=4))


if __name__ == "__main__":
    main()