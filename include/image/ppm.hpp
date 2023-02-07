#pragma once

#include <fstream>

namespace rendex::image {

auto write_ppm(const auto &filename, const auto &colors, auto width, auto height) {
    std::ofstream ostream(filename);

    ostream << "P3\n";
    ostream << width << " " << height << "\n";
    ostream << (1 << 8) - 1 << "\n";

    for (const auto &color : colors) {
        for (const auto &component : color) {
            ostream << component * ((1 << 8) - 1) << " ";
        }
        ostream << "\n";
    }
}

}  // namespace rendex::image
