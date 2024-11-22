import json
from pathlib import Path
from typing import Any, Literal

import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.lip_dataset import LipReadingConfig, create_lipreading_index
from src.utils.io_utils import ROOT_PATH


class AVSSDataset(BaseDataset):
    """
    Audio-Video Source Separation.

    Contains audio + video of 2 speakers
    """

    def __init__(
        self,
        part: Literal["train", "val", "test"] = "train",
        data_dir: str | None = None,
        load_video: bool = False,
        video_config: LipReadingConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            part (str): partition part
            load_video (bool): load video part or not
        """
        self.data_dir = (
            ROOT_PATH / "data" / "dla_dataset" if data_dir is None else data_dir
        )

        if load_video:
            assert (
                video_config is not None
            ), "Should provide config when load_video is True"
            if create_lipreading_index(video_config):
                print(f"Succesfully created video index for {part}!")
            else:
                print("Using pre-made video index.")

        index = self._get_or_load_index(part, load_video)
        super().__init__(index, load_video=load_video, *args, **kwargs)

    @staticmethod
    def load_files(path: Path) -> tuple[Path, int]:
        info = torchaudio.info(path)
        lenght = info.num_frames / info.sample_rate
        return path, lenght

    def _get_or_load_index(self, part: str, load_video: bool) -> list[dict[str, Any]]:
        suffix = f"{part}_av_index.json" if load_video else f"{part}_index.json"
        index_path = self.data_dir / suffix
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, load_video)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part: str, load_video: bool) -> list[dict[str, str]]:
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            part (str): partition part
            load_video (bool): load video part or not
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        audio_data_path = ROOT_PATH / "data" / "dla_dataset" / "audio" / part
        video_data_path = ROOT_PATH / "data" / "dla_dataset" / "mouths_embeddings"

        assert (
            audio_data_path.exists() and audio_data_path.is_dir()
        ), f"No {audio_data_path} found!"
        if load_video:
            assert (
                video_data_path.exists() and video_data_path.is_dir()
            ), f"No {video_data_path} found!"

        print(f"Loading {part} AVSS dataset")

        for mix_path in tqdm((audio_data_path / "mix").iterdir()):
            dataset_item = {}

            item_id = mix_path.name
            mix_path, mix_lenght = self.load_files(mix_path)
            dataset_item.update(
                {
                    "mix_path": str(mix_path),
                    "audio_lenght": mix_lenght,
                    "id": item_id.rstrip(".wav"),
                }
            )

            if part != "test":
                speaker_1_path, speaker_1_lenght = self.load_files(
                    audio_data_path / "s1" / item_id
                )
                speaker_2_path, speaker_2_lenght = self.load_files(
                    audio_data_path / "s2" / item_id
                )
                assert (
                    mix_lenght == speaker_1_lenght and mix_lenght == speaker_2_lenght
                ), "Audio files should have same lenght"

                dataset_item.update(
                    {
                        "sp1_audio_path": str(speaker_1_path),
                        "sp2_audio_path": str(speaker_2_path),
                    }
                )

            if load_video:
                sp1_id, sp2_id = str(mix_path.name).rstrip(".wav").split("_")
                dataset_item.update(
                    {
                        "sp1_video_path": str(video_data_path / sp1_id) + ".npz",
                        "sp2_video_path": str(video_data_path / sp2_id) + ".npz",
                    }
                )

            if load_video:
                for speaker_id in [1, 2]:
                    video_dataset_item = {
                        "mix_path": dataset_item["mix_path"],
                        "audio_lenght": mix_lenght,
                        "target_video_path": dataset_item[f"sp{speaker_id}_video_path"],
                    }
                    if part != "test":
                        video_dataset_item.update(
                            {
                                "target_audio_path": dataset_item[
                                    f"sp{speaker_id}_audio_path"
                                ]
                            }
                        )
                    index.append(video_dataset_item)
            else:
                index.append(dataset_item)

        return index
