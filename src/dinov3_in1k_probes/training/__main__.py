"""DINOv3 IN1K linear probe: extraction and training.

Usage:
    uv run python -m dinov3_in1k_probes.training extract ...
    uv run python -m dinov3_in1k_probes.training train ...
"""

from typing import Annotated, Union

import tyro

from dinov3_in1k_probes.training.config import ExtractionConfig, TrainConfig
from dinov3_in1k_probes.training.extract import run_extraction
from dinov3_in1k_probes.training.train import run_training

_Command = Union[
    Annotated[ExtractionConfig, tyro.conf.subcommand("extract")],
    Annotated[TrainConfig, tyro.conf.subcommand("train")],
]


def main() -> None:
    cmd: ExtractionConfig | TrainConfig = tyro.cli(_Command)  # pyright: ignore[reportCallIssue,reportArgumentType]
    match cmd:
        case ExtractionConfig():
            run_extraction(cmd)
        case TrainConfig():
            run_training(cmd)


if __name__ == "__main__":
    main()
