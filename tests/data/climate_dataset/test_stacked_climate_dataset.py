from climate_learn.data.climate_dataset.args import (
    ClimateDatasetArgs,
    ERA5Args,
)
from climate_learn.data.climate_dataset import (
    StackedClimateDatasetArgs,
    StackedClimateDataset,
)


class TestStackedClimateDatasetInstantiation:
    def test_initialization(self):
        data_args = []
        data_arg1 = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"],
            constants=["random_constant"],
            split="train",
        )
        data_arg2 = ERA5Args(
            root_dir="my_data_path",
            variables=["geopotential", "2m_temperature"],
            years=range(2010, 2015),
            constants=["land_sea_mask", "orography"],
            split="train",
        )
        data_args.append(data_arg1)
        data_args.append(data_arg2)
        StackedClimateDataset(StackedClimateDatasetArgs(data_args=data_args))
