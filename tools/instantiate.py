import os
import math
import argparse
import textwrap


def main(args):

    os.makedirs("src/instantiated", exist_ok=True)
    instantiated_sources = [f"src/instantiated/main_{i:0{math.ceil(math.log10(args.width * args.height / args.chunk_size))}}.cpp" for i in range(
        args.width * args.height // args.chunk_size)]

    for i, instantiated_source in enumerate(instantiated_sources):
        with open(instantiated_source, "w") as file:
            file.write(textwrap.dedent(f"""\
                #include "main.hpp"

                template constexpr auto render<{args.type}, {args.width}, {args.height}, {args.samples}, {args.chunk_size * i}, {args.chunk_size}>();
            """))

    with open("src/main.cpp", "w") as file:
        file.write(textwrap.dedent(f'''\
            #include "main.hpp"

            #include <algorithm>
            #include <array>

            #include "image.hpp"

            int main() {{
                constexpr auto W = {args.width};
                constexpr auto H = {args.height};
                constexpr auto S = {args.samples};
                constexpr auto N = {args.chunk_size};

                constexpr auto image = [&]() constexpr {{
                    std::array<std::array<double, 3>, W * H> colors;
                    auto iterator = std::begin(colors);
                    [function = [&]<auto I, auto... Is>(auto self, std::index_sequence<I, Is...>) {{
                        auto subcolors = render<double, W, H, S, N * I, N>();
                        iterator = std::copy(std::begin(subcolors), std::end(subcolors), iterator);
                        if constexpr (sizeof...(Is)) self(self, std::index_sequence<Is...>{{}});
                    }}](auto &&...args) {{
                        return function(function, std::forward<decltype(args)>(args)...);
                    }}(std::make_index_sequence<W * H / N>{{}});
                    return colors;
                }}();

                rendex::image::write_ppm("image.ppm", image, W, H);
            }}
        '''))

    os.environ["INSTANTIATED_SOURCES"] = " ".join(instantiated_sources)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Automatic Instantiation for Separate Rendering")
    parser.add_argument("-T", "--type", type=str, default="double", help="Floating-point type used for rendering")
    parser.add_argument("-W", "--width", type=int, default=1200, help="Width of the image to be rendered")
    parser.add_argument("-H", "--height", type=int, default=800, help="Height of the image to be rendered")
    parser.add_argument("-S", "--samples", type=int, default=500,
                        help="Number of samples for MSAA (Multi-Sample Anti-Aliasing)")
    parser.add_argument("-N", "--chunk_size", type=int, default=10000, help="Number of pixels in each chunk")

    main(parser.parse_args())
