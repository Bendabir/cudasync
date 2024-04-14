"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import enum
import itertools as it
from pathlib import Path  # noqa: TCH003
from typing import NoReturn, assert_never, cast, final

import cv2
import joblib
import numpy as np
import numpy.typing as npt
import typer
from tqdm import tqdm


def read_image(path: Path) -> bytes:
    # Not really useful but then it's somehow iso with the async counterpart
    with path.open("rb") as file:
        return file.read()


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


def process_image(
    path: Path,
    height: int,
    width: int,
    interpolation: Interpolation,
    bar: tqdm[NoReturn],
) -> None:
    data = read_image(path)
    image = decode_image(data)
    image = resize_image(image, height, width, interpolation)

    bar.update(1)


def run(
    directory: Path,
    height: int,
    width: int,
    interpolation: Interpolation,
    workers: int,
) -> None:
    bar = tqdm(unit="image")

    try:
        # NOTE : Using threading should be okay because OpenCV should release the GIL
        with joblib.Parallel(
            n_jobs=max(1, workers),
            backend="threading",
            return_as="generator",
        ) as parallel:
            for _ in parallel(
                joblib.delayed(process_image)(
                    path,
                    height,
                    width,
                    interpolation,
                    bar,
                )
                for path in it.cycle(directory.glob("*.jpg"))
            ):
                pass
    finally:
        bar.close()


def main(
    directory: Path,
    height: int = 1024,
    width: int = 1024,
    interpolation: Interpolation = Interpolation.CUBIC,
    workers: int = 8,
) -> None:
    run(directory, height, width, interpolation, workers)


# ~100 FPS on my machine (i7-12700KF) with 1 worker
# ~190 FPS on my machine (i7-12700KF) with 2 workers
# ~350 FPS on my machine (i7-12700KF) with 4 workers
# ~600 FPS on my machine (i7-12700KF) with 8 workers
# ~800 FPS on my machine (i7-12700KF) with 16 workers
# ~750 FPS on my machine (i7-12700KF) with 20 workers
if __name__ == "__main__":
    typer.run(main)
