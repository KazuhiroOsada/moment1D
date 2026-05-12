#ifndef PUSH_HYPERBOLIC_HPP
#define PUSH_HYPERBOLIC_HPP

#include <Kokkos_Core.hpp>

#include "core.hpp"
#include "field_flux_solver.hpp"
#include "moment_flux_solver.hpp"


namespace pushH {

inline Kokkos::View<double**> rk_temp_moment;
inline Kokkos::View<double**> rk_temp_field;

void push_moment_rk2(core::Species& species, const double dt);
void push_field_rk2(const double dt);

} // namespace pushH

#endif // PUSH_HYPERBOLIC_HPP
