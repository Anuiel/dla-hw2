from pathlib import Path

import torch
import torchaudio
from tqdm.auto import tqdm

from src.loss import PIT_SISNR
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        save_path,
        dataloaders,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.save_path = save_path
        if self.save_path is not None:
            (save_path / "s1").mkdir(exist_ok=True, parents=True)
            (save_path / "s2").mkdir(exist_ok=True, parents=True)
        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *([m.name for m in self.metrics["inference"]] + ["PIT_SI-SNR"]),
                writer=None,
            )
        else:
            self.evaluation_metrics = None
        self.si_snr = PIT_SISNR(2).to(device)

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["preds"].shape[0]
        predictions = batch["preds"]
        if "targets" in batch:
            loss, opt_p = self.si_snr(return_p=True, **batch)
            metrics.update("PIT_SI-SNR", -loss)
        else:
            opt_p = [[0, 1] for _ in range(len(batch["preds"]))]

        ordered_predictions = []
        for i in range(batch_size):
            sp1_pred, sp2_pred = (
                self.normalize_audio(predictions[i, opt_p[i][0], :]),
                self.normalize_audio(predictions[i, opt_p[i][1], :]),
            )
            ordered_predictions.append(
                torch.cat((sp1_pred.unsqueeze(0), sp2_pred.unsqueeze(0)), dim=0)
            )
            if self.save_path is not None:
                output_path_sp1: Path = self.save_path / "s1" / (batch["id"][i] + ".wav")
                torchaudio.save(
                    output_path_sp1,
                    sp1_pred.unsqueeze(0).detach().cpu(),
                    16000,
                    format="wav",
                    buffer_size=128,
                )
                output_path_sp2: Path = self.save_path / "s2" / (batch["id"][i] + ".wav")
                torchaudio.save(
                    output_path_sp2,
                    sp2_pred.unsqueeze(0).detach().cpu(),
                    16000,
                    format="wav",
                    buffer_size=128,
                )
        ordered_predictions = torch.cat(
            [pred.unsqueeze(0) for pred in ordered_predictions]
        )
        batch["preds"] = ordered_predictions
        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        return batch

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        db_threshold = 30
        amplitude_threshold = 10 ** (db_threshold / 20.0)
        audio = audio / audio.abs().max()
        clipped_waveform = audio.clamp(
            min=-amplitude_threshold, max=amplitude_threshold
        )
        return clipped_waveform

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
