# Standard library
from typing import Callable, Dict, Tuple

# Local application
from climate_learn.data.task.task import Task
from climate_learn.data.task.args import RetrievalArgs

# Third party
import torch

Data = Dict[str, torch.tensor]


class Retrieval(Task):
    _args_class: Callable[..., RetrievalArgs] = RetrievalArgs

    def __init__(self, task_args: RetrievalArgs):
        super().__init__(task_args)
        self.vars = task_args.in_vars
        
    def get_raw_index(self, index: int) -> int:
        return index * self.subsample
    
    def get_time_index(self, index: int) -> int:
        return index * self.subsample
    
    def create_constants_data(self, *args, **kwargs) -> Data:
        return {}
    
    def create_inp_out(
        self,
        raw_data: Data,
        constants_data: Data,
        apply_transform: bool = True
    ) -> Tuple[Data, Data]:
        inp_data: Data = {
            k: raw_data[k] for k in self.vars
        }
        if apply_transform:
            # Need to unsqueeze for inp_data as history is not the same as channel
            inp_data = {
                k: (self.inp_transform[k](inp_data[k].unsqueeze(1))).squeeze(1)
                for k in self.in_vars
            }
        return inp_data, {}


RetrievalArgs._task_class = Retrieval
