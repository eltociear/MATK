from .mami import MamiDataModule
from .fhm import FHMDataModule
from .harmeme import HarMemeDataModule

def load_datamodule(dataset_name, model_class_or_path, **kwargs):
    if "fhm" in dataset_name:
        return FHMDataModule(dataset_name, model_class_or_path, **kwargs)

    elif "mami" in dataset_name:
        return MamiDataModule(dataset_name, model_class_or_path, **kwargs)

    elif "harmeme" in dataset_name:
        return HarMemeDataModule(dataset_name, model_class_or_path, **kwargs)

    else:
        raise NotImplementedError(f"'{dataset_name}' datamodule not implemented")