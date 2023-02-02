#pragma once

#include <optional>
#include <variant>

#include "sphere.hpp"

namespace rendex::geometry {

template <typename Scalar, typename Vector, typename Material>
class Sphere;

template <typename Scalar, typename Vector, typename Material>
using Geometry = std::variant<Sphere<Scalar, Vector, Material>>;

}  // namespace rendex::geometry
