import os
import math
import argparse
import textwrap


def main(args):

    os.makedirs("src/instantiated", exist_ok=True)
    instantiated_sources = [
        f"src/instantiated/main_{index:0{math.ceil(math.log10(args.chunks))}}.cpp" for index in range(args.chunks)]

    for index, instantiated_source in enumerate(instantiated_sources):
        with open(instantiated_source, "w") as file:
            file.write(textwrap.dedent(f'''\
                #include "main.hpp"

                namespace {{
                #ifdef CONSTEXPR
                constexpr
                #else
                #endif
                auto chunk = render<{args.type}, {args.width}, {args.height}, {args.chunks}, {index}>();
                }}

                template <>
                const std::invoke_result_t<decltype(render<{args.type}, {args.width}, {args.height}, {args.chunks}, {index}>)>& fetch<{args.type}, {args.width}, {args.height}, {args.chunks}, {index}>() {{
                    return chunk;
                }}
            '''))

    with open("src/main.cpp", "w") as file:
        file.write(textwrap.dedent(f'''\
            #include "main.hpp"

            #include <algorithm>
            #include <array>
            #include <execution>

            #include "image.hpp"

            {"🥶".join([
                f"extern template const std::invoke_result_t<decltype(render<{args.type}, {args.width}, {args.height}, {args.chunks}, {index}>)>& fetch<{args.type}, {args.width}, {args.height}, {args.chunks}, {index}>();"
                for index in range(args.chunks)
            ])}

            int main() {{
                using T = {args.type};
                constexpr auto W = {args.width};
                constexpr auto H = {args.height};
                constexpr auto N = {args.chunks};

                std::vector<rendex::tensor::Vector<T, 3>> colors(W * H);
                auto iterator = std::begin(colors);
                [function = [&]<auto I, auto... Is>(auto self, std::integer_sequence<decltype(N), I, Is...>) {{
                    const auto& chunk = fetch<T, W, H, N, I>();
                    iterator = std::copy(std::execution::par_unseq, std::begin(chunk), std::end(chunk), iterator);
                    if constexpr (sizeof...(Is)) self(self, std::integer_sequence<decltype(N), Is...>{{}});
                }}](auto&&... args) {{
                    return function(function, std::forward<decltype(args)>(args)...);
                }}(std::make_integer_sequence<decltype(N), N>{{}});

                rendex::image::write_ppm("image.ppm", colors, W, H);
            }}
        ''').replace("🥶", "\n"))

    with open(".CMakeLists.txt") as src_file:
        with open("CMakeLists.txt", "w") as dst_file:
            dst_file.write(src_file.read().replace("${INSTANTIATED_SOURCES}", "\n\t".join(instantiated_sources)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Automatic Instantiation for Separate Rendering")
    parser.add_argument("-T", "--type", type=str, default="double", help="Floating-point type used for rendering")
    parser.add_argument("-W", "--width", type=int, default=100, help="Width of the image to be rendered")
    parser.add_argument("-H", "--height", type=int, default=100, help="Height of the image to be rendered")
    parser.add_argument("-N", "--chunks", type=int, default=100, help="Number of chunks all the pixels are split into")

    main(parser.parse_args())
