"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import enum
import random
from pathlib import Path  # noqa: TCH003 (for Typer)
from typing import NoReturn, final

import anyio
import anyio.abc
import httpx
import typer
from tqdm import tqdm


async def download_image(client: httpx.AsyncClient, endpoint: str) -> bytes:
    response = await client.get(endpoint)

    response.raise_for_status()

    return response.content


async def save_image(content: bytes, path: Path) -> None:
    async with await anyio.open_file(path, mode="wb") as file:
        await file.write(content)


async def process_image(  # noqa: PLR0913, PLR0917
    seed: str,
    index: int,
    directory: Path,
    client: httpx.AsyncClient,
    limiter: anyio.CapacityLimiter,
    bar: tqdm[NoReturn],
    *,
    task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
) -> None:
    async with limiter:
        task_status.started()

        # Force the seed for every image as I'm not sure if calls will come in order
        seed = f"{seed}_{index}"

        random.seed(seed)

        height = random.randint(256, 2048)  # noqa: S311
        width = random.randint(256, 2048)  # noqa: S311
        content = await download_image(client, f"/seed/{seed}/{width}/{height}.jpg")

        await save_image(content, directory / f"{index}.jpg")

        bar.update(1)


async def run(
    directory: Path,
    base_url: str,
    seed: str,
    count: int,
    concurrency: int,
) -> None:
    client = httpx.AsyncClient(
        base_url=base_url,
        follow_redirects=True,
    )
    limiter = anyio.CapacityLimiter(concurrency)
    bar = tqdm(total=count, unit="image")

    directory.mkdir(parents=True, exist_ok=True)

    try:
        async with anyio.create_task_group() as task_group:
            for index in range(count):
                _ = await task_group.start(
                    process_image,
                    seed,
                    index,
                    directory,
                    client,
                    limiter,
                    bar,
                )
    finally:
        bar.close()


@final
class AsyncBackend(enum.StrEnum):  # Typer doesn't support Literal yet...
    ASYNCIO = "asyncio"
    TRIO = "trio"


def main(  # noqa: PLR0913, PLR0917
    directory: Path,
    base_url: str = "https://picsum.photos",
    seed: str = "cuda",
    count: int = 100,
    concurrency: int = 10,
    backend: AsyncBackend = AsyncBackend.TRIO,
) -> None:
    anyio.run(run, directory, base_url, seed, count, concurrency, backend=backend)


if __name__ == "__main__":
    typer.run(main)
