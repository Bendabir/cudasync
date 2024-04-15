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
    size: int,
    height: int,
    width: int,
    total: int | None,
) -> Iterator[torch.Tensor]:
    # Spawn the data on the GPU directly and randomize.
    # Purpose is to pressurize the GPU with compute.
    # Therefore, we don't real need to set the seed.
    # Data normalization will occur on GPU.
    # The overall preprocesing should be as follow, but we'll mock this
    # as we only need the good shape and data type for testing.
    #    The images are resized to resize_size=[256]
    #    using interpolation=BILINEAR,
    #    followed by a central crop of crop_size=[224].
    #    Finally the values are first rescaled to [0.0, 1.0]
    #    and then normalized using mean=[0.485, 0.456, 0.406]
    #    and std=[0.229, 0.224, 0.225].
    batch = torch.randint(
        low=0,
        high=256,
        size=(size, 3, height, width),  # BCHW, Torch is channels first
        dtype=torch.uint8,
        pin_memory=True,  # Force the Tensor to stay in memory
    )
    i = 0

    while total is None or i < total:
        torch.randint(low=0, high=256, size=(size, 3, height, width), out=batch)

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

    # Define an output buffer for async GPU to CPU transfer
    # (otherwise, it is sent to paged memory)
    output = torch.zeros((size, 1_000), dtype=torch.float32, pin_memory=True)

    with torch.inference_mode(), tqdm(unit="image") as bar:
        for _batch in batches(
            size=size,
            height=height,
            width=width,
            total=total,
        ):
            batch = _batch.to(device, non_blocking=True)
            batch = batch.to(torch.float32).mul_(1.0 / 255)
            logits: torch.Tensor = model(batch)
            scores = F.softmax(logits, dim=0)

            scores.to(output, non_blocking=True)

            bar.update(len(scores))


# BS 1 : ~550 FPS on RTX 3080 Ti
# BS 2 : ~850 FPS on RTX 3080 Ti
# BS 4 : ~1300 FPS on RTX 3080 Ti
# BS 8 : ~1700 FPS on RTX 3080 Ti
# BS 16 : ~2000 FPS on RTX 3080 Ti
# BS 32 : ~2200 FPS on RTX 3080 Ti
# BS 64 : ~2350 FPS on RTX 3080 Ti
# BS 128 : ~2450 FPS on RTX 3080 Ti
# BS 256 : ~2500 FPS on RTX 3080 Ti
# BS 512 : ~2500 FPS on RTX 3080 Ti
# BS 1024 : ~2500 FPS on RTX 3080 Ti
if __name__ == "__main__":
    typer.run(main)
