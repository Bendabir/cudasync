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


# NOTE : We can convince ourselves that the GIL is actually released
#        by replacing the following instructions with a dummy instructions
#        that hold the GIL.
#        On my machine, this simple for-loop produces the same throughput with
#        single worker. When spawning more workers, performances start to degrades
#        because of the GIL.
def process_image_with_gil(
    _path: Path,
    _height: int,
    _width: int,
    _interpolation: Interpolation,
    bar: tqdm[NoReturn],
    load: int = 1_500_000,
) -> None:
    for _ in range(load):
        pass

    bar.update(1)


def process_image_no_gil(
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


def run(  # noqa: PLR0913
    directory: Path,
    height: int,
    width: int,
    interpolation: Interpolation,
    workers: int,
    *,
    simulate: bool,
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
                joblib.delayed(
                    process_image_with_gil if simulate else process_image_no_gil
                )(
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


def main(  # noqa: PLR0913, PLR0917
    directory: Path,
    height: int = 1024,
    width: int = 1024,
    interpolation: Interpolation = Interpolation.CUBIC,
    workers: int = 8,
    simulate: bool = False,  # noqa: FBT001, FBT002
) -> None:
    run(directory, height, width, interpolation, workers, simulate=simulate)


# ~100 FPS on my machine (i7-12700KF) with 1 worker
# ~190 FPS on my machine (i7-12700KF) with 2 workers
# ~350 FPS on my machine (i7-12700KF) with 4 workers
# ~600 FPS on my machine (i7-12700KF) with 8 workers
# ~800 FPS on my machine (i7-12700KF) with 16 workers
# ~750 FPS on my machine (i7-12700KF) with 20 workers

# ~100 FPS on my machine (i7-12700KF) with 1 worker and GIL locking instructions
# ~100 FPS on my machine (i7-12700KF) with 2 workers and GIL locking instructions
# ~90 FPS on my machine (i7-12700KF) with 4 workers and GIL locking instructions
# ~70 FPS on my machine (i7-12700KF) with 8 workers and GIL locking instructions
# ~10 FPS on my machine (i7-12700KF) with 16 workers and GIL locking instructions
if __name__ == "__main__":
    typer.run(main)
