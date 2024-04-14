"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
import typer
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator


def batches(
    device: str,
    size: int,
    height: int,
    width: int,
    total: int | None,
) -> Iterator[torch.Tensor]:
    # Spawn the data on the GPU directly and randomize.
    # Purpose is to pressurize the GPU with compute.
    # Therefore, we don't real need to set the seed.
    # As the ResNet model expects data to be normalized (thus FP32),
    # we directly generate the data.
    # However, this normalization could (and should) be done on GPU directly.
    batch = torch.rand(
        size=(size, 3, height, width),  # BCHW, Torch is channels first
        dtype=torch.float32,
        device=device,
    )
    i = 0

    while total is None or i < total:
        torch.rand(size=(size, 3, height, width), out=batch)

        yield batch

        i += 1


def main(
    device: str = "cuda:0",
    size: int = 16,
    height: int = 224,
    width: int = 224,
    # New Python 3.10 unions not supported by Typer
    total: Optional[int] = None,  # noqa: UP007
) -> None:
    # Using a small model to CPU really has to saturate the GPU with instructions
    model = torchvision.models.resnet18().eval().to(device)

    with torch.inference_mode(), tqdm(unit="image") as bar:
        for batch in batches(
            device,
            size=size,
            height=height,
            width=width,
            total=total,
        ):
            logits: torch.Tensor = model(batch)
            scores = F.softmax(logits, dim=0)

            bar.update(len(scores))


# BS 1 : ~500 FPS on RTX 3080 Ti
# BS 2 : ~1000 FPS on RTX 3080 Ti
# BS 4 : ~2000 FPS on RTX 3080 Ti
# BS 8 : ~3000 FPS on RTX 3080 Ti
# BS 16 : ~4000 FPS on RTX 3080 Ti
# BS 32 : ~4400 FPS on RTX 3080 Ti
# BS 64 : ~4700 FPS on RTX 3080 Ti
# BS 128 : ~5000 FPS on RTX 3080 Ti
# BS 256 : ~5000 FPS on RTX 3080 Ti
# BS 512 : ~5000 FPS on RTX 3080 Ti
# BS 1024 : ~5100 FPS on RTX 3080 Ti

# BS 1 : ~100 FPS on i7-12700KF
# BS 2 : ~100 FPS on i7-12700KF
# BS 4 : ~110 FPS on i7-12700KF
# BS 8 : ~120 FPS on i7-12700KF
# BS 16 : ~120 FPS on i7-12700KF
# BS 32 : ~120 FPS on i7-12700KF
# BS 64 : ~110 FPS on i7-12700KF
# BS 128 : ~120 FPS on i7-12700KF
# BS 256 : ~125 FPS on i7-12700KF
# BS 512 : ~130 FPS on i7-12700KF
# BS 1024 : ~135 FPS on i7-12700KF
if __name__ == "__main__":
    typer.run(main)
