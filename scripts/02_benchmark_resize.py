"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import itertools as it
from pathlib import Path  # noqa: TCH003
from typing import Annotated, Literal, NoReturn, assert_never, cast

import anyio
import cv2
import cyclopts
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


async def read_image(path: Path) -> bytes:
    async with await anyio.open_file(path, mode="rb") as file:
        return await file.read()


def decode_image(data: bytes) -> npt.NDArray[np.uint8]:
    return cast(
        npt.NDArray[np.uint8],
        cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR),
    )


Interpolation = Literal["area", "linear", "cubic"]


def flag(interpolation: Interpolation) -> int:
    match interpolation:
        case "area":
            return cv2.INTER_AREA
        case "linear":
            return cv2.INTER_LINEAR
        case "cubic":
            return cv2.INTER_CUBIC
        case never:
            assert_never(never)


def resize_image(
    image: npt.NDArray[np.uint8],
    height: int,
    width: int,
    interpolation: Interpolation,
) -> npt.NDArray[np.uint8]:
    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(image, (width, height), interpolation=flag(interpolation)),
    )


async def process_image(  # noqa: PLR0913
    path: Path,
    height: int,
    width: int,
    interpolation: Interpolation,
    bar: tqdm[NoReturn],
    *,
    task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
) -> npt.NDArray[np.uint8]:
    task_status.started()

    # Only read operation is async
    data = await read_image(path)
    image = decode_image(data)
    image = resize_image(image, height, width, interpolation)

    bar.update(1)

    return image


async def run(
    directory: Path,
    height: int,
    width: int,
    interpolation: Interpolation,
) -> None:
    bar = tqdm(unit="image")

    try:
        async with anyio.create_task_group() as task_group:
            for path in it.cycle(directory.glob("*.jpg")):
                _ = await task_group.start(
                    process_image,
                    path,
                    height,
                    width,
                    interpolation,
                    bar,
                )
    finally:
        bar.close()


AsyncBackend = Literal["asyncio", "trio"]

app = cyclopts.App()


@app.default
def main(
    *,
    directory: Path,
    height: int = 1024,
    width: int = 1024,
    interpolation: Annotated[
        Interpolation,
        cyclopts.Parameter(
            help="Use 'area' when shrinking and 'cubic' or 'linear' when upscaling."
        ),
    ] = "cubic",
    backend: AsyncBackend = "trio",
) -> None:
    anyio.run(run, directory, height, width, interpolation, backend=backend)


# ~90-100 FPS on my machine (i7-12700KF)
if __name__ == "__main__":
    app()
