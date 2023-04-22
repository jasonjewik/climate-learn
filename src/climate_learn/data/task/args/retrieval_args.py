# Standard library
from __future__ import annotations
from typing import Callable, Sequence, TYPE_CHECKING, Union

# Local application
from climate_learn.data.task.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data.task import Retrieval

class RetrievalArgs(TaskArgs):
    _task_class: Union[Callable[..., Retrieval], str] = "Retrieval"

    def __init__(
        self,
        vars: Sequence[str],
        subsample: int = 1
    ):
        super().__init__(vars, [], [], subsample)
        self.check_validity()