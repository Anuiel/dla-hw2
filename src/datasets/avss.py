import json
from pathlib import Path
from typing import Literal, Any

import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
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
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            part (str): partition part
            load_video (bool): load video part or not
        """
        self.load_video = load_video
        self.data_dir = ROOT_PATH / "data" / "dla_dataset" if data_dir is None else data_dir

        index = self._get_or_load_index(part, load_video)
        super().__init__(index, *args, **kwargs)
    
    @staticmethod
    def load_files(path: Path) -> tuple[Path, int]:
        info = torchaudio.info(path)
        lenght = info.num_frames / info.sample_rate
        return path, lenght

    def _get_or_load_index(self, part: str, load_video: bool) -> list[dict[str, Any]]:
        if self.load_video:
            # TODO: separate index with video? Or not, idk
            raise NotImplementedError
        index_path = self.data_dir / f"{part}_index.json"
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
        if load_video == True:
            raise NotImplementedError("Video is not yet supported!")
        index = []
        audio_data_path = ROOT_PATH / "data" / "dla_dataset" / "audio" / part
        assert audio_data_path.exists() and audio_data_path.is_dir(), f"No {audio_data_path} found!"

        print(f"Loading {part} AVSS dataset")
        
        for mix_path in tqdm((audio_data_path / "mix").iterdir()):
            item_id = mix_path.name
            speaker_1_path = audio_data_path / "s1" / item_id
            mix_path, mix_lenght = self.load_files(mix_path)
            speaker_1_path, speaker_1_lenght = self.load_files(audio_data_path / "s1" / item_id)
            speaker_2_path, speaker_2_lenght = self.load_files(audio_data_path / "s2" / item_id)
            assert (
                mix_lenght == speaker_1_lenght and mix_lenght == speaker_2_lenght,
                "Audio files should have same lenght"
            )
            index.append({
                "mix_path": str(mix_path),
                "speaker_1_path": str(speaker_1_path),
                "speaker_2_path": str(speaker_2_path),
                "audio_lenght": mix_lenght
            })

        return index
