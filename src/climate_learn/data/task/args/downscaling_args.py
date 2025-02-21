# Standard library
from __future__ import annotations
from typing import Callable, Sequence, TYPE_CHECKING, Union

# Local application
from climate_learn.data.task.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data.task import Downscaling


class DownscalingArgs(TaskArgs):
    _task_class: Union[Callable[..., Downscaling], str] = "Downscaling"

    def __init__(
        self,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constants: Sequence[str] = [],
        subsample: int = 1,
    ) -> None:
        super().__init__(in_vars, out_vars, constants, subsample)
        self.check_validity()
