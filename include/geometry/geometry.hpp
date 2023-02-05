#pragma once

#include <optional>
#include <variant>

#include "reflection.hpp"
#include "sphere.hpp"

namespace rendex::geometry {

template <typename Scalar, template <typename, auto> typename Vector,
          template <typename, template <typename, auto> typename> typename Material>
class Sphere;

template <typename Scalar, template <typename, auto> typename Vector>
using Geometry = std::variant<Sphere<Scalar, Vector, rendex::reflection::Dielectric>,
                              Sphere<Scalar, Vector, rendex::reflection::Metal>>;

}  // namespace rendex::geometry
