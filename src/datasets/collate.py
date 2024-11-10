import torch

from src.datasets.base_dataset import DatasetItem


def pad_sequence(
    data: list[torch.Tensor], padding_item: float | int = 0
) -> torch.Tensor:
    """
    Pad sequence of tensors [1, ..., variable_lenght] -> [batch_size, ..., max(variable_lenght)]

    Args:
        data: list if torch.Tensor with identical shapes except last one
    Returns:
        batch: torch.Tensor with shape [len(data), ..., max(data.shape[-1])]
    """
    max_lenght = max(item.shape[-1] for item in data)

    padded_data = []
    for item in data:
        time_padding = max_lenght - item.shape[-1]
        padded_item = torch.nn.functional.pad(
            item, (0, time_padding), mode="constant", value=padding_item
        )
        padded_data.append(padded_item)
    return torch.stack(padded_data)


def collate_fn(dataset_items: list[DatasetItem]) -> DatasetItem:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    for speaker in ["sp1", "sp2", "mix"]:
        # # For inference with no ground truth
        # spectrogram_key = f"{speaker}_spectrogram"
        # spectrogram_length = torch.tensor(
        #     [item[spectrogram_key].shape[-1] for item in dataset_items]
        # )
        # result_batch[f"{speaker}_spectrogram_lenght"] = spectrogram_length

        # spectrogram = pad_sequence(
        #     [item[spectrogram_key].squeeze(0) for item in dataset_items],
        #     padding_item=math.log(1e-6),
        # )
        # result_batch[spectrogram_key] = spectrogram
        # For inference with no ground truth
        audio_key = f"{speaker}_audio"
        audio_length = torch.tensor(
            [item[audio_key].shape[-1] for item in dataset_items]
        )
        result_batch[f"{speaker}_audio_lenght"] = audio_length

        audio = pad_sequence(
            [item[audio_key].squeeze(0) for item in dataset_items],
            padding_item=0.0,
        )
        result_batch[audio_key] = audio

    return result_batch
