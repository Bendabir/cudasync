"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import itertools as it
from typing import Literal, assert_never

import cyclopts
import joblib
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision as tv
from tqdm import tqdm


# This simulate the processing a batch
def run(  # type: ignore[no-any-unimported]  # noqa: PLR0913, PLR0917
    device: str,
    stream: torch.cuda.Stream,
    model: tv.models.ResNet,
    batch: torch.Tensor,  # The input (pinned)
    result: torch.Tensor,  # The output (pinned)
    size: int,
    height: int,
    width: int,
    id_: int,
) -> int:
    assert batch.is_pinned()  # noqa: S101
    assert result.is_pinned()  # noqa: S101

    with torch.inference_mode(), torch.cuda.stream(stream):
        torch.cuda.nvtx.range_push(  # type: ignore[no-untyped-call]
            f"random_generation_{id_}"
        )
        torch.randint(low=0, high=256, size=(size, 3, height, width), out=batch)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

        torch.cuda.nvtx.range_push(f"cpu_to_gpu_{id_}")  # type: ignore[no-untyped-call]
        batch = batch.to(device, non_blocking=True)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

        torch.cuda.nvtx.range_push(  # type: ignore[no-untyped-call]
            f"normalization_{id_}"
        )
        batch = batch.to(torch.float32).mul_(1.0 / 255)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

        torch.cuda.nvtx.range_push(f"model_{id_}")  # type: ignore[no-untyped-call]
        logits: torch.Tensor = model(batch)
        scores = F.softmax(logits, dim=0)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

        torch.cuda.nvtx.range_push(f"gpu_to_cpu_{id_}")  # type: ignore[no-untyped-call]
        scores.to(result, non_blocking=True)
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]

    return len(batch)


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
    threads: int = 2,
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

    # Prepare input buffers for batches
    # Also define an output buffers for async GPU to CPU transfer
    # (otherwise, it is sent to paged memory)
    threads = max(1, threads)
    batches = [
        torch.zeros(
            (size, 3, height, width),  # BCHW, Torch is channels first
            dtype=torch.uint8,
            pin_memory=True,  # Force the Tensor to stay in memory
        )
        for _ in range(threads)
    ]
    results = [
        torch.zeros((size, 1_000), dtype=torch.float32, pin_memory=True)
        for _ in range(threads)
    ]
    streams = [
        torch.cuda.Stream() for _ in range(threads)  # type: ignore[no-untyped-call]
    ]

    buffers = it.cycle(enumerate(zip(batches, results, streams, strict=True), start=1))

    with (
        tqdm(unit="image") as bar,
        joblib.Parallel(
            n_jobs=threads,
            backend="threading",
            return_as="generator",
        ) as parallel,
    ):
        for processed in parallel(
            joblib.delayed(run)(
                device,
                s,
                resnet,
                b,
                r,
                size,
                height,
                width,
                i,
            )
            for i, (b, r, s) in (
                # Define how to inject data to workers
                buffers
                if total is None
                else it.islice(buffers, total)
            )
        ):
            bar.update(processed)


# BS 1 : ~500 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 2 : ~950 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 4 : ~1800 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 8 : ~2400 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 16 : ~2500 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 32 : ~2500 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 64 : ~2500 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 128 : ~2500 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 256 : ~2500 FPS on RTX 3080 Ti for ResNet18 with 2 threads
# BS 512 : ~2550 FPS on RTX 3080 Ti for ResNet18 with 2 threads

# BS 1 : ~225 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 2 : ~450 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 4 : ~900 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 8 : ~1400 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 16 : ~1550 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 32 : ~1600 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 64 : ~1625 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 128 : ~1625 FPS on RTX 3080 Ti for ResNet50 with 2 threads
# BS 256 : ~1650 FPS on RTX 3080 Ti for ResNet50 with 2 threads
if __name__ == "__main__":
    app()
