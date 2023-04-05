# Standard library
from typing import Callable, Dict, Sequence, Tuple

# Third party
import numpy
import torch

# Local application
from climate_learn.data.climate_dataset.args import StackedClimateDatasetArgs
from climate_learn.data.climate_dataset import ClimateDataset


class StackedClimateDataset(ClimateDataset):
    _args_class: Callable[..., StackedClimateDatasetArgs] = StackedClimateDatasetArgs

    def __init__(self, data_args: StackedClimateDatasetArgs) -> None:
        self.climate_datasets: Sequence[ClimateDataset] = []
        for data_arg in data_args.child_data_args:
            if isinstance(data_arg._data_class, str):
                climate_dataset_class: Callable[..., ClimateDataset] = eval(
                    data_arg._data_class
                )
            else:
                climate_dataset_class: Callable[
                    ..., ClimateDataset
                ] = data_arg._data_class
            self.climate_datasets.append(climate_dataset_class(data_arg))

    def setup(
        self, style: str = "map", setup_args: Dict = {}
    ) -> Tuple[int, Sequence[Dict[str, Sequence[str]]]]:
        dataset_length: Sequence[int] = []
        variables_to_update: Sequence[Dict[str, Sequence[str]]] = []
        for climate_dataset in self.climate_datasets:
            length, var_to_update = climate_dataset.setup(style, setup_args)
            dataset_length.append(length)
            variables_to_update.append(var_to_update)
        assert len(set(dataset_length)) == 1
        return dataset_length[0], variables_to_update

    def load_chunk(self, chunk_id: int) -> int:
        dataset_length: Sequence[int] = []
        for climate_dataset in self.climate_datasets:
            length: int = climate_dataset.load_chunk(chunk_id)
            dataset_length.append(length)
        assert len(set(dataset_length)) == 1
        return dataset_length[0]

    def get_item(self, index: int) -> Sequence[Dict[str, torch.tensor]]:
        return [dataset.get_item(index) for dataset in self.climate_datasets]

    def get_constants_data(self) -> Sequence[Dict[str, torch.tensor]]:
        return [dataset.get_constants_data() for dataset in self.climate_datasets]

    def get_metadata(self) -> Sequence[Dict[str, numpy.ndarray]]:
        return [dataset.get_metadata() for dataset in self.climate_datasets]


StackedClimateDatasetArgs._data_class = StackedClimateDataset
