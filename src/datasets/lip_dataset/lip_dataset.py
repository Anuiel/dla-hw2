import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from src.datasets.lip_dataset.lipreading.lipreading.utils import load_json, load_model
from src.datasets.lip_dataset.lipreading.lipreading.model import Lipreading
from src.datasets.lip_dataset.lipreading.lipreading.dataloaders import get_preprocessing_pipelines


class LipReadingConfig:
    modality: str = "video"
    num_classes: int = 500
    extract_feats: bool = True
    
    def __init__(
        self,
        data_path: str,
        embeddings_path: str,
        model_config_path: str,
        model_path: str,
        device: str = 'cuda'
    ) -> None:
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_config_path = Path(model_config_path)
        self.model_path = Path(model_path)
        self.device = device
        assert device == "cuda", "Lipreading library suppoting only CUDA devices"


def create_lipreading_index(config: LipReadingConfig) -> bool:
    assert config.data_path.exists() and config.data_path.is_dir(), f"No such directory {config.data_path}!"
    assert config.model_config_path.exists() and config.model_config_path.is_file(), f"No such file: {config.config_path}!"
    assert config.model_path.exists() and config.model_path.is_file(), f"No such file: {config.model_path}!"

    if config.embeddings_path.exists():
        return False
    else:
        config.embeddings_path.mkdir(parents=True)

    model = get_model_from_json(config)
    model = load_model(config.model_path, model)
    model.eval()
    model.to(config.device)

    preprocessing = get_preprocessing_pipelines("video")['test']
    with torch.no_grad():
        for file in tqdm(config.data_path.iterdir()):
            lips = np.load(file)['data']
            model_input = torch.FloatTensor(preprocessing(lips))
            model_input = model_input[None, None, :, :, :].to(config.device)
            output = model(model_input, lengths=[1]).cpu().numpy()[0, :, :]
            np.savez_compressed(config.embeddings_path / file.name, embedding=output)
    return True


# copy from
# https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/main.py#L191-L231
def get_model_from_json(args: LipReadingConfig):
    assert str(args.model_config_path).endswith('.json') and os.path.isfile(args.model_config_path), \
        f"'.json' config path does not exist. Path input: {args.model_config_path}"
    args_loaded = load_json(str(args.model_config_path))
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
    else:
        densetcn_options = {}

    model = Lipreading( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        use_boundary=args.use_boundary,
                        extract_feats=args.extract_feats).to(args.device)
    return model
