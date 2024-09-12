"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import cyclopts
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
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
        torch.cuda.nvtx.range_push("random_generation")  # type: ignore[no-untyped-call]
        torch.randint(low=0, high=256, size=(size, 3, height, width), out=batch)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

        yield batch

        i += 1


Model = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

app = cyclopts.App()


@app.default
def main(  # noqa: PLR0913
    *,
    model: Model = "resnet18",
    device: str = "cuda:0",
    size: int = 16,
    height: int = 224,
    width: int = 224,
    total: int | None = None,
) -> None:
    # Typing seems to be broken for TorchVision
    resnet: torchvision.models.ResNet  # type: ignore[no-any-unimported]

    match model:
        case "resnet18":
            resnet = torchvision.models.resnet18()
        case "resnet34":
            resnet = torchvision.models.resnet34()
        case "resnet50":
            resnet = torchvision.models.resnet50()
        case "resnet101":
            resnet = torchvision.models.resnet101()
        case "resnet152":
            resnet = torchvision.models.resnet152()
        case never:
            assert_never(never)

    resnet = resnet.eval().to(device)

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
            torch.cuda.nvtx.range_push("cpu_to_gpu")  # type: ignore[no-untyped-call]
            batch = _batch.to(device, non_blocking=True)
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

            torch.cuda.nvtx.range_push("normalization")  # type: ignore[no-untyped-call]
            batch = batch.to(torch.float32).mul_(1.0 / 255)
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

            torch.cuda.nvtx.range_push("model")  # type: ignore[no-untyped-call]
            logits: torch.Tensor = resnet(batch)
            scores = F.softmax(logits, dim=0)
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

            torch.cuda.nvtx.range_push("gpu_to_cpu")  # type: ignore[no-untyped-call]
            scores.to(output, non_blocking=True)
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

            bar.update(len(scores))


# BS 1 : ~550 FPS on RTX 3080 Ti for ResNet18
# BS 2 : ~850 FPS on RTX 3080 Ti for ResNet18
# BS 4 : ~1300 FPS on RTX 3080 Ti for ResNet18
# BS 8 : ~1700 FPS on RTX 3080 Ti for ResNet18
# BS 16 : ~2000 FPS on RTX 3080 Ti for ResNet18
# BS 32 : ~2200 FPS on RTX 3080 Ti for ResNet18
# BS 64 : ~2350 FPS on RTX 3080 Ti for ResNet18
# BS 128 : ~2450 FPS on RTX 3080 Ti for ResNet18
# BS 256 : ~2500 FPS on RTX 3080 Ti for ResNet18
# BS 512 : ~2500 FPS on RTX 3080 Ti for ResNet18
# BS 1024 : ~2500 FPS on RTX 3080 Ti for ResNet18

# BS 1 : ~275 FPS on RTX 3080 Ti for ResNet50
# BS 2 : ~500 FPS on RTX 3080 Ti for ResNet50
# BS 4 : ~800 FPS on RTX 3080 Ti for ResNet50
# BS 8 : ~1250 FPS on RTX 3080 Ti for ResNet50
# BS 16 : ~1350 FPS on RTX 3080 Ti for ResNet50
# BS 32 : ~1425 FPS on RTX 3080 Ti for ResNet50
# BS 64 : ~1500 FPS on RTX 3080 Ti for ResNet50
# BS 128 : ~1600 FPS on RTX 3080 Ti for ResNet50
# BS 256 : ~1625 FPS on RTX 3080 Ti for ResNet50
# BS 512 : ~1625 FPS on RTX 3080 Ti for ResNet50
if __name__ == "__main__":
    app()
