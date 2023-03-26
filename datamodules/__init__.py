from .mami import MamiDataModule

def load_datamodule(dataset_name, model_class_or_path, **kwargs):
    if "fhm" in dataset_name:
        return FHMDataModule(dataset_name, model_class_or_path, **kwargs)

    if "mami" in dataset_name:
        return MamiDataModule(dataset_name, model_class_or_path, **kwargs)
    else:
        raise NotImplementedError(f"'{dataset_name}' datamodule not implemented")