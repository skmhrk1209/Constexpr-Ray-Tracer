#pragma once

#include "generators.hpp"
#include "math.hpp"

namespace rendex::random {

template <typename G>
constexpr auto uniform(G& generator, auto min, auto max) {
    return rendex::math::lerp(generator(), G::min, G::max, min, max);
}

}  // namespace rendex::random
