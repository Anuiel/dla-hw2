import logging
import random
from pathlib import Path
from typing import Any, Callable, List, NewType

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


DatasetItem = NewType("DatasetItem", dict[str, Any])


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index: list[DatasetItem],
        load_video: bool,
        target_sr: int = 16000,
        max_audio_length: int | None = None,
        limit: int | None = None,
        shuffle_index: bool = False,
        dynamic_mixing: bool = False, 
        random_seed: int = 42, 
        instance_transforms: dict[str, Callable[[Any], Any]] | None = None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            load_video (bool): load video of lips
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            dynamic_mixing (bool): generate pairs online instead of premade pairs.
            random_seed (int): for dynamic mixing random id generation.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index,
            max_audio_length,
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index = index
        self.is_load_video = load_video

        self.dynamic_mixing = dynamic_mixing
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms
        self.random_generator = random.Random(random_seed)


    def __getitem__(self, ind: int) -> DatasetItem:
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        
        if self.dynamic_mixing:
            ind_pair = self.random_generator.randint(0, len(self._index) - 1)
            sp1_audio, sp2_audio = (
                self.load_audio(Path(path))
                for path in (
                    self._index[ind]["speaker_1_path"],
                    self._index[ind_pair]["speaker_2_path"],
                )
            )
            mix_audio = sp1_audio + sp2_audio

            instance_data = {
                "mix_audio": mix_audio,
                "sp1_audio": sp1_audio,
                "sp2_audio": sp2_audio,
            }
        else: 
            instance_data = {
                "mix_audio": self.load_audio(Path(data_dict["mix_path"])),
                "id": data_dict["id"],
            }
            if not self.is_load_video:
                sp1_audio, sp2_audio = (
                    self.load_audio(Path(path) if path is not None else None)
                    for path in (
                        data_dict.get("sp1_audio_path", None),
                        data_dict.get("sp2_audio_path", None),
                    )
                )

                if sp1_audio is not None:
                    # There is ground truth found
                    instance_data.update(
                        {
                            "sp1_audio": sp1_audio,
                            "sp2_audio": sp2_audio,
                        }
                    )
            else:
                if "target_audio_path" in data_dict:
                    target_audio = self.load_audio(Path(data_dict["target_audio_path"]))
                    instance_data.update({"target_audio": target_audio})

                instance_data.update(
                    {
                        "target_video": self.load_video(data_dict["target_video_path"]),
                    }
                )

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_video(self, path: Path) -> torch.Tensor:
        video = np.load(path)["embedding"]
        return torch.tensor(video)

    def load_audio(self, path: Path | None) -> torch.Tensor | None:
        if path is None:
            return None
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def __len__(self) -> int:
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_object(self, path: str) -> torch.Tensor:
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        data_object = torch.load(path)
        return data_object

    def preprocess_data(self, instance_data: DatasetItem) -> DatasetItem:
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list[DatasetItem], max_audio_length: int | None
    ) -> list[DatasetItem]:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_lenght"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = None

        if exceeds_audio_length is not None and exceeds_audio_length.any():
            _total = exceeds_audio_length.sum()
            index = [
                el for el, exclude in zip(index, exceeds_audio_length) if not exclude
            ]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index: list[DatasetItem]) -> list[DatasetItem]:
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "mix_path" in entry, (
                "Each dataset item should include field 'mix_path'"
                " - path to mix of audio."
            )

    @staticmethod
    def _sort_index(index: list[DatasetItem]) -> list[DatasetItem]:
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["audio_lenght"])

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[DatasetItem], limit: int | None, shuffle_index: bool
    ) -> list[DatasetItem]:
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
