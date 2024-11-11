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
    audio_length = torch.tensor(
        [item["mix_audio"].shape[-1] for item in dataset_items]
    )
    result_batch["audio_lenght"] = audio_length

    mix_audio = pad_sequence(
        [item["mix_audio"].squeeze(0) for item in dataset_items],
        padding_item=0.0,
    )
    result_batch["mix_audio"] = mix_audio

    sp1_audio = pad_sequence(
        [item["sp1_audio"].squeeze(0) for item in dataset_items],
        padding_item=0.0,
    )
    sp2_audio = pad_sequence(
        [item["sp2_audio"].squeeze(0) for item in dataset_items],
        padding_item=0.0,
    )

    # [batch_size, n_speakers, seq_len]
    target_audio = torch.cat((sp1_audio.unsqueeze(1), sp2_audio.unsqueeze(1)), dim=1)
    result_batch["targets"] = target_audio
    return result_batch
