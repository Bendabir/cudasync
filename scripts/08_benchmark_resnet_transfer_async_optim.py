"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional, assert_never, final

import torch
import torch.jit
import torch.nn.functional as F  # noqa: N812
import torchvision as tv
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
        torch.cuda.nvtx.range_push("random_generation")  # type: ignore[no-untyped-call]
        torch.randint(low=0, high=256, size=(size, 3, height, width), out=batch)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

        yield batch

        i += 1


@final
class Models(enum.StrEnum):
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"

    def get(self) -> tv.models.ResNet:  # type: ignore[no-any-unimported]
        match self:
            case Models.RESNET_18:
                return tv.models.resnet18()
            case Models.RESNET_34:
                return tv.models.resnet34()
            case Models.RESNET_50:
                return tv.models.resnet50()
            case Models.RESNET_101:
                return tv.models.resnet101()
            case Models.RESNET_152:
                return tv.models.resnet152()
            case never:
                assert_never(never)


@final
class Optimizations(enum.StrEnum):
    TORCH_SCRIPT = "script"
    TORCH_COMPILE = "compile"

    def optimize(  # type: ignore[no-any-unimported]
        self,
        model: tv.models.ResNet,
    ) -> tv.models.ResNet:
        match self:
            case Optimizations.TORCH_COMPILE:
                return torch.compile(model.eval(), mode="reduce-overhead")
            case Optimizations.TORCH_SCRIPT:
                return torch.jit.optimize_for_inference(torch.jit.script(model))
            case never:
                assert_never(never)


def main(  # noqa: PLR0913, PLR0917
    model: Models = Models.RESNET_18,
    device: str = "cuda:0",
    size: int = 16,
    height: int = 224,
    width: int = 224,
    # New Python 3.10 unions not supported by Typer
    total: Optional[int] = None,  # noqa: UP007
    optimization: Optional[Optimizations] = None,  # noqa: UP007
) -> None:
    resnet = model.get().eval().to(device)

    if optimization is not None:
        torch.set_float32_matmul_precision("high")

        resnet = optimization.optimize(resnet)

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


# BS 1 : ~1250 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 2 : ~1900 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 4 : ~2100 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 8 : ~2300 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 16 : ~2400 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 32 : ~2400 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 64 : ~2450 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 128 : ~2500 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 256 : ~2500 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 512 : ~2550 FPS on RTX 3080 Ti for ResNet18 with torch.compile
# BS 1024 : ~2550 FPS on RTX 3080 Ti for ResNet18 with torch.compile

# BS 1 : ~900 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 2 : ~1350 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 4 : ~1650 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 8 : ~2000 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 16 : ~2200 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 32 : ~2350 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 64 : ~2450 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 128 : ~2450 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 256 : ~2500 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 512 : ~2550 FPS on RTX 3080 Ti for ResNet18 with TorchScript
# BS 1024 : ~2550 FPS on RTX 3080 Ti for ResNet18 with TorchScript

# BS 1 : ~750 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 2 : ~1100 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 4 : ~1425 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 8 : ~1650 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 16 : ~1825 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 32 : ~2000 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 64 : ~2150 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 128 : ~2500 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 256 : ~2500 FPS on RTX 3080 Ti for ResNet50 with torch.compile
# BS 512 : ~2500 FPS on RTX 3080 Ti for ResNet50 with torch.compile

# BS 1 : ~450 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 2 : ~775 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 4 : ~1100 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 8 : ~1475 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 16 : ~1750 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 32 : ~2000 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 64 : ~2100 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 128 : ~2400 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 256 : ~2500 FPS on RTX 3080 Ti for ResNet50 with TorchScript
# BS 512 : ~2525 FPS on RTX 3080 Ti for ResNet50 with TorchScript
if __name__ == "__main__":
    typer.run(main)
