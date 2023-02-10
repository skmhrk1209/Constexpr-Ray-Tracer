#include <string>

#include "image.hpp"
#include "math.hpp"
#include "rendering.hpp"
#include "scene.hpp"
#include "tensor.hpp"

int main() {
    using namespace std::literals::string_literals;

    constexpr auto ImageWidth = IMAGE_WIDTH;
    constexpr auto ImageHeight = IMAGE_HEIGHT;
    constexpr auto PatchWidth = PATCH_WIDTH;
    constexpr auto PatchHeight = PATCH_HEIGHT;
    constexpr auto PatchCoordX = PATCH_COORD_X;
    constexpr auto PatchCoordY = PATCH_COORD_Y;
    constexpr auto MaxDepth = MAX_DEPTH;
    constexpr auto NumSamples = NUM_SAMPLES;
    constexpr auto RandomSeed = RANDOM_SEED;

    CONSTEXPR auto image = [&]() constexpr {
        // rendering
        auto colors =
            coex::rendering::ray_tracing<Scalar, ImageWidth, ImageHeight, PatchWidth, PatchHeight, PatchCoordX,
                                           PatchCoordY>(object, camera, background, MaxDepth, NumSamples, RandomSeed);

        // gamma correction
        std::transform(std::begin(colors), std::end(colors), std::begin(colors),
                       [](const auto &color) { return coex::tensor::elemwise(coex::math::sqrt<Scalar>, color); });
        return colors;
    }();

    auto filename = "outputs/patch_"s + std::to_string(PatchCoordX) + "_"s + std::to_string(PatchCoordY) + ".ppm"s;
    coex::image::write_ppm(filename, image, PatchWidth, PatchHeight);
}
