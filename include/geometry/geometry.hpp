#pragma once

#include <optional>
#include <variant>

#include "reflection.hpp"
#include "sphere.hpp"

namespace coex::geometry {

template <typename Scalar, template <typename, auto> typename Vector,
          template <typename, template <typename, auto> typename> typename Material>
class Sphere;

template <typename Scalar, template <typename, auto> typename Vector>
using Geometry = std::variant<Sphere<Scalar, Vector, coex::reflection::Lambertian>,
                              Sphere<Scalar, Vector, coex::reflection::Dielectric>,
                              Sphere<Scalar, Vector, coex::reflection::Metal>>;

}  // namespace coex::geometry
