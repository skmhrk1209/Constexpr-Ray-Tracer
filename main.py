import os
import atexit
import random
import asyncio
import argparse
import textwrap
import itertools
import subprocess
import skimage.io
import numpy as np


def main(args):

    processes = set()

    @atexit.register
    def killall():
        for process in processes:
            process.kill()

    async def compile_image():

        semaphore = asyncio.Semaphore(args.max_workers)

        async def compile_patch(patch_coord_x, patch_coord_y):

            async with semaphore:

                print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
                print("Launching...")

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

                process = await asyncio.create_subprocess_shell(
                    f"srun bash {os.path.join(dirname, 'build.sh')}",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    limit=1024 * 128,
                )
                processes.add(process)

                print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
                print("Launched!")

                while not process.stdout.at_eof():

                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), args.stdout_timeout)

                    except asyncio.TimeoutError:
                        pass

                    else:
                        print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
                        print(line.decode())

                await process.wait()

            return (patch_coord_x, patch_coord_y), process

        patch_coords_x, patch_coords_y = zip(*itertools.product(range(args.image_height // args.patch_height), range(args.image_width // args.patch_width)))
        patch_compilations = list(map(asyncio.create_task, map(compile_patch, patch_coords_x, patch_coords_y)))

        for patch_compilation in asyncio.as_completed(patch_compilations):

            (patch_coord_x, patch_coord_y), process = await patch_compilation

            print(f"\n================================ Patch ({patch_coord_x}, {patch_coord_y}) ================================")
            print("Process failed..." if process.returncode else "Process succeeded!")

            if process.returncode: break

            processes.remove(process)

        else:

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

        for patch_compilation in patch_compilations:
            if not patch_compilation.done():
                patch_compilation.cancel()

    asyncio.run(compile_image())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Separate Compilation Script")
    parser.add_argument("--constexpr", action="store_true", help="whether to enable compile-time ray tracing")
    parser.add_argument("--image_width", type=int, default=1200, help="width of the image")
    parser.add_argument("--image_height", type=int, default=800, help="height of the image")
    parser.add_argument("--patch_width", type=int, default=300, help="width of each patch")
    parser.add_argument("--patch_height", type=int, default=200, help="height of each patch")
    parser.add_argument("--max_depth", type=int, default=50, help="maximum depth for recursive ray tracing")
    parser.add_argument("--num_samples", type=int, default=500, help="number of samples for MSAA (Multi-Sample Anti-Aliasing)")
    parser.add_argument("--random_seed", type=int, default=random.randrange(1 << 32), help="random seed for Monte Carlo approximation")
    parser.add_argument("--max_workers", type=int, default=16, help="maximum number of workers for multiprocessing")
    parser.add_argument("--stdout_timeout", type=float, default=1.0, help="Timeout for reading one line from the stream of each child process")

    main(parser.parse_args())
