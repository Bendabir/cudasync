"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import cyclopts
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision as tv
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
    resnet: tv.models.ResNet  # type: ignore[no-any-unimported]

    match model:
        case "resnet18":
            resnet = tv.models.resnet18()
        case "resnet34":
            resnet = tv.models.resnet34()
        case "resnet50":
            resnet = tv.models.resnet50()
        case "resnet101":
            resnet = tv.models.resnet101()
        case "resnet152":
            resnet = tv.models.resnet152()
        case never:
            assert_never(never)

    resnet = resnet.eval().to(device)

    with torch.inference_mode(), tqdm(unit="image") as bar:
        for batch in batches(
            device,
            size=size,
            height=height,
            width=width,
            total=total,
        ):
            logits: torch.Tensor = resnet(batch)
            scores = F.softmax(logits, dim=0)

            bar.update(len(scores))


# BS 1 : ~500 FPS on RTX 3080 Ti for ResNet18
# BS 2 : ~1000 FPS on RTX 3080 Ti for ResNet18
# BS 4 : ~2000 FPS on RTX 3080 Ti for ResNet18
# BS 8 : ~3000 FPS on RTX 3080 Ti for ResNet18
# BS 16 : ~4000 FPS on RTX 3080 Ti for ResNet18
# BS 32 : ~4400 FPS on RTX 3080 Ti for ResNet18
# BS 64 : ~4700 FPS on RTX 3080 Ti for ResNet18
# BS 128 : ~5000 FPS on RTX 3080 Ti for ResNet18
# BS 256 : ~5000 FPS on RTX 3080 Ti for ResNet18
# BS 512 : ~5000 FPS on RTX 3080 Ti for ResNet18
# BS 1024 : ~5100 FPS on RTX 3080 Ti for ResNet18

# BS 1 : ~100 FPS on i7-12700KF for ResNet18
# BS 2 : ~100 FPS on i7-12700KF for ResNet18
# BS 4 : ~110 FPS on i7-12700KF for ResNet18
# BS 8 : ~120 FPS on i7-12700KF for ResNet18
# BS 16 : ~120 FPS on i7-12700KF for ResNet18
# BS 32 : ~120 FPS on i7-12700KF for ResNet18
# BS 64 : ~110 FPS on i7-12700KF for ResNet18
# BS 128 : ~120 FPS on i7-12700KF for ResNet18
# BS 256 : ~125 FPS on i7-12700KF for ResNet18
# BS 512 : ~130 FPS on i7-12700KF for ResNet18
# BS 1024 : ~135 FPS on i7-12700KF for ResNet18

# BS 1 : ~300 FPS on RTX 3080 Ti for ResNet50
# BS 2 : ~600 FPS on RTX 3080 Ti for ResNet50
# BS 4 : ~1150 FPS on RTX 3080 Ti for ResNet50
# BS 8 : ~1300 FPS on RTX 3080 Ti for ResNet50
# BS 16 : ~1400 FPS on RTX 3080 Ti for ResNet50
# BS 32 : ~1450 FPS on RTX 3080 Ti for ResNet50
# BS 64 : ~1525 FPS on RTX 3080 Ti for ResNet50
# BS 128 : ~1600 FPS on RTX 3080 Ti for ResNet50
# BS 256 : ~1625 FPS on RTX 3080 Ti for ResNet50
# BS 512 : ~1625 FPS on RTX 3080 Ti for ResNet50
if __name__ == "__main__":
    app()
