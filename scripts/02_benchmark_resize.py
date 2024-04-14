"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import enum
import itertools as it
from pathlib import Path  # noqa: TCH003
from typing import NoReturn, assert_never, cast, final

import anyio
import cv2
import numpy as np
import numpy.typing as npt
import typer
from tqdm import tqdm


async def read_image(path: Path) -> bytes:
    async with await anyio.open_file(path, mode="rb") as file:
        return await file.read()


def decode_image(data: bytes) -> npt.NDArray[np.uint8]:
    return cast(
        npt.NDArray[np.uint8],
        cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR),
    )


# For compat with Typer, easier
@final
class Interpolation(enum.StrEnum):
    # When shrinking : cv2.INTER_AREA
    # When upscaling : cv2.INTER_LINEAR or cv2.INTER_CUBIC
    AREA = "area"
    LINEAR = "linar"
    CUBIC = "cubic"

    @property
    def flag(self) -> int:
        match self:
            case Interpolation.AREA:
                return cv2.INTER_AREA
            case Interpolation.LINEAR:
                return cv2.INTER_LINEAR
            case Interpolation.CUBIC:
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
        cv2.resize(image, (width, height), interpolation=interpolation.flag),
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


@final
class AsyncBackend(enum.StrEnum):  # Typer doesn't support Literal yet...
    ASYNCIO = "asyncio"
    TRIO = "trio"


def main(
    directory: Path,
    height: int = 1024,
    width: int = 1024,
    interpolation: Interpolation = Interpolation.CUBIC,
    backend: AsyncBackend = AsyncBackend.TRIO,
) -> None:
    anyio.run(run, directory, height, width, interpolation, backend=backend)


# ~90-100 FPS on my machine (i7-12700KF)
if __name__ == "__main__":
    typer.run(main)
