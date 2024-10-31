from pathlib import Path
from typing import Literal

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
        name: Literal["train", "val", "test"] = "train",
        load_video: bool = False,
        *args,
        **kwargs
    ):
        """
        Args:
            name (str): partition name
            load_video (bool): load video part or not
        """
        index = self._create_index(name, load_video)

        super().__init__(index, *args, **kwargs)
    
    @staticmethod
    def load_files(path: Path) -> tuple[Path, int]:
        # info = torchaudio.info(path, backend="ffmpeg")
        # lenght = info.num_frames / info.sample_rate
        lenght = 1_000_000 # TODO: debug this
        return path, lenght

    def _create_index(self, name: str, load_video: bool):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
            load_video (bool): load video part or not
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        if load_video == True:
            raise NotImplementedError("Video is not yet supported!")
        index = []
        audio_data_path = ROOT_PATH / "data" / "dla_dataset" / "audio" / name
        assert audio_data_path.exists() and audio_data_path.is_dir(), f"No {audio_data_path} found!"

        print(f"Loading {name} AVSS dataset")
        
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
                "mix_path": mix_path,
                "speaker_1_path": speaker_1_path,
                "speaker_2_path": speaker_2_path,
                "audio_lenght": mix_lenght
            })

        return index
