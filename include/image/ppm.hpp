#pragma once

#include <concepts>
#include <type_traits>

#include "tensor.hpp"

namespace rendex::image {

template <typename Image>
auto write_ppm(const Image &image, const auto &filename)
    requires(rendex::tensor::rank_v<Image> == 3)
{
    std::ofstream ostream(filename);

    ostream << "P3"
            << "\n";
    ostream << rendex::tensor::dimension_v<Image, 1> << "\n";
    ostream << rendex::tensor::dimension_v<Image, 0> << "\n";
    ostream << (1 << 8) - 1 << "\n";

    for (const auto &colors : image) {
        for (const auto &color : colors) {
            for (const auto &value : color) {
                ostream << value * ((1 << 8) - 1) << " ";
            }
            ostream << "\n";
        }
    }
}

}  // namespace rendex::image
