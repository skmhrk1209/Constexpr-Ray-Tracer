import os
import random
import argparse
import textwrap
import itertools
import functools
import threading
import subprocess
import concurrent
import contextlib
import skimage.io
import numpy as np


@contextlib.contextmanager
def lock_guard(lock, *args, **kwargs):
    locked = lock.acquire(*args, **kwargs)
    try:
        yield locked
    finally:
        locked and lock.release()


def main(args):

    lock = threading.Lock()

    def render(patch_coord_x, patch_coord_y):

        with lock_guard(lock):
            print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
            print(f"Rendering process has just been launched!")

        dirname = f"build/patch_{patch_coord_x}_{patch_coord_y}"
        os.makedirs(dirname, exist_ok=True)

        with open(os.path.join(dirname, "build.sh"), "w") as file:

            file.write(textwrap.dedent(f"""\
                #!/bin/bash

                echo -------------------------------- CMake --------------------------------

                cmake \\
                    -D CMAKE_BUILD_TYPE=Release \\
                    -D CONSTEXPR={"ON" if args.constexpr else "OFF"} \\
                    -D IMAGE_WIDTH={args.image_width} \\
                    -D IMAGE_HEIGHT={args.image_height} \\
                    -D PATCH_WIDTH={args.patch_width} \\
                    -D PATCH_HEIGHT={args.patch_height} \\
                    -D PATCH_COORD_X={patch_coord_x} \\
                    -D PATCH_COORD_Y={patch_coord_y} \\
                    -D MAX_DEPTH={args.max_depth} \\
                    -D NUM_SAMPLES={args.num_samples} \\
                    -D RANDOM_SEED={args.random_seed} \\
                    -S {os.path.dirname(os.path.abspath(__file__))} \\
                    -B {os.path.join(dirname, "build")}

                echo -------------------------------- Make --------------------------------

                cmake --build {os.path.join(dirname, "build")}

                echo -------------------------------- Rendering --------------------------------

                {os.path.join(dirname, "build", "ray_tracing")}
            """))

        process = subprocess.Popen(
            f"srun bash {os.path.join(dirname, 'build.sh')}",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )

        while process.poll() is None:

            with lock_guard(lock):
                print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
                print(process.stdout.readline())

        process.wait()

        return (patch_coord_x, patch_coord_y), process


    os.makedirs("outputs", exist_ok=True)

    patch_coords_x, patch_coords_y = zip(*itertools.product(range(args.image_height // args.patch_height), range(args.image_width // args.patch_width)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers or os.cpu_count()) as executor:

        futures = list(map(functools.partial(executor.submit, render), patch_coords_x, patch_coords_y))

        returncodes = []

        for future in concurrent.futures.as_completed(futures):

            (patch_coord_x, patch_coord_y), process = future.result()

            with lock_guard(lock):
                print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
                print("Rendering process failed..." if process.returncode else "Rendering process succeeded!")

            returncodes.append(process.returncode)

        if not any(returncodes):

            print("\n================================================================")
            print("All the patches were successfully rendered!")

            image = np.concatenate([
                np.concatenate([
                    skimage.io.imread(f"outputs/patch_{patch_coord_x}_{patch_coord_y}.ppm")
                    for patch_coord_x in range(args.image_width // args.patch_width)
                ], axis=1)
                for patch_coord_y in range(args.image_height // args.patch_height)
            ], axis=0)

            skimage.io.imsave(f"outputs/image.png", image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Separate Rendering Script")
    parser.add_argument("--constexpr", action="store_true", help="whether to enable compile-time ray tracing")
    parser.add_argument("--image_width", type=int, default=1200, help="width of the image")
    parser.add_argument("--image_height", type=int, default=800, help="height of the image")
    parser.add_argument("--patch_width", type=int, default=300, help="width of each patch")
    parser.add_argument("--patch_height", type=int, default=200, help="height of each patch")
    parser.add_argument("--max_depth", type=int, default=50, help="maximum depth for recursive ray tracing")
    parser.add_argument("--num_samples", type=int, default=500, help="number of samples for MSAA (Multi-Sample Anti-Aliasing)")
    parser.add_argument("--random_seed", type=int, default=random.randrange(1 << 32), help="random seed for Monte Carlo approximation")
    parser.add_argument("--max_workers", type=int, default=16, help="maximum number of workers for multiprocessing")

    main(parser.parse_args())
