from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.task.args import TaskArgs
from climate_learn.data.dataset.args import ShardDatasetArgs


class TestShardDatasetArgsInstantiation:
    def test_initialization(self):
        climate_dataset_args = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"],
            constants=["random_constant"],
            split="train",
        )
        task_args = TaskArgs(
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
            constants=["random_constant"],
            subsample=3,
        )
        ShardDatasetArgs(climate_dataset_args, task_args, 4)
