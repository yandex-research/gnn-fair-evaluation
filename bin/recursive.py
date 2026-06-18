import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Literal

import lib

_cancelled = threading.Event()


def _run_experiment(
    config: Path,
    queue: Queue[int | None],
    script: str,
    args: list[str],
    continue_: bool,
    force: bool,
    quiet: bool,
) -> None:
    if _cancelled.is_set():
        return

    resource = queue.get()
    try:
        label = f'GPU {resource}' if resource is not None else 'CPU'
        print(f'[{label}] Starting {config}')
        cmd = [sys.executable, script, str(config), *args]

        if continue_:
            cmd.append('--continue')

        if force:
            cmd.append('--force')

        env = dict(os.environ)
        if resource is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(resource)

        stdio = (
            {'stdout': subprocess.DEVNULL, 'stderr': subprocess.PIPE} if quiet else {}
        )
        result = subprocess.run(cmd, env=env, **(stdio if quiet else {}))  # type: ignore
        if result.returncode != 0:
            stderr = (
                result.stderr.decode(errors='replace').rstrip()
                if quiet and result.stderr
                else ''
            )
            raise RuntimeError(
                f'Process exited with code {result.returncode}'
                + (f'\n{stderr}' if stderr else '')
            )
        print(f'[{label}] Finished {config}')
    finally:
        queue.put(resource)


def main(
    path: str | Path,
    patterns: None | list[str] = None,
    script: str = 'bin/go.py',
    procedure: Literal['tuning', 'evaluation'] = 'tuning',
    args: list[str] | None = None,
    *,
    gpus: list[int] | None = None,
    n_jobs: int | None = None,
    continue_: bool = False,
    force: bool = False,
    dry: bool = False,
):
    path = Path(path)
    assert not lib.is_deprecated_exp(path), 'This experiment branch is deprecated!'

    if gpus is not None and n_jobs is not None:
        raise ValueError('--gpus and --n_jobs are mutually exclusive')

    if gpus is not None:
        n_workers = len(gpus)
        resources: list[int | None] = list(gpus)
    elif n_jobs is not None:
        n_workers = n_jobs
        resources = [None] * n_jobs
    else:
        n_workers = 1
        resources = [0]

    configs = []
    for config in sorted(path.rglob(f'{procedure}.toml')):
        if not patterns or any(p in str(config) for p in patterns):
            output = config.with_suffix('')
            if not lib.is_done(output) or force:
                configs.append(config)

    terminal_size = os.get_terminal_size().columns
    print('─' * terminal_size)
    print(f'Found {len(configs)} {procedure}.toml configs at {path}')
    print('─' * terminal_size)
    print(json.dumps([str(config) for config in configs], indent=4))
    print('─' * terminal_size)
    print()
    print(f'Number of ops: {len(configs)}')
    print(f'Number of workers: {n_workers}')
    print(
        f'Indices of GPUs: {resources}'
        if resources[0] is not None
        else 'No GPUs to be used'
    )
    print()
    print('─' * terminal_size)

    if dry:
        return

    queue: Queue[int | None] = Queue()
    for r in resources:
        queue.put(r)

    quiet = n_workers > 1
    failures: list[Path] = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: dict[Future[None], Path] = {
            executor.submit(
                _run_experiment,
                config,
                queue,
                script,
                args if args is not None else [],
                continue_,
                force,
                quiet,
            ): config
            for config in configs
        }
        try:
            for future in as_completed(futures):
                config = futures[future]
                try:
                    future.result()
                except Exception as error:
                    message = str(error)
                    print(f'\n{config} failed with the following message:\n{message}')
                    failures.append(config)
        except KeyboardInterrupt:
            print('\nInterrupted. Cancelling pending experiments...')
            _cancelled.set()
            for future in futures:
                future.cancel()
            raise

    if failures:
        print(f'\n{len(failures)} experiment(s) failed:')
        for config in failures:
            print(config)


if __name__ == '__main__':
    lib.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--patterns', nargs='+')
    parser.add_argument('--script', type=str, default='bin/go.py')
    parser.add_argument('--procedure', choices=['tuning', 'evaluation'], required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpus', nargs='+', type=int)
    group.add_argument('--n_jobs', type=int)

    parser.add_argument('--continue', action='store_true', dest='continue_')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry', action='store_true')

    # NOTE: `parse_known_args` enables us to pass script args
    known, unknown = parser.parse_known_args()
    main(**vars(known), args=unknown)
