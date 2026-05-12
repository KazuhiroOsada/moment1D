#ifndef FIELD_FLUX_SOLVER_HPP
#define FIELD_FLUX_SOLVER_HPP

#include <Kokkos_Core.hpp>

#include "core.hpp"


namespace field_flux {

KOKKOS_INLINE_FUNCTION
double minmod(const double a, const double b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (Kokkos::fabs(a) < Kokkos::fabs(b)) ? a : b;
}

KOKKOS_INLINE_FUNCTION
Kokkos::pair<core::FieldArray, core::FieldArray> reconstruct_field_muscl(const Kokkos::View<double**>& U, int hx) {
    core::FieldArray UL, UR;
    for (int var = 0; var < core::N_FIELD; ++var) {
        const double uim1 = U(var, hx - 1);
        const double ui = U(var, hx);
        const double uip1 = U(var, hx + 1);
        const double uip2 = U(var, hx + 2);
        const double slope_L = minmod(ui - uim1, uip1 - ui);
        const double slope_R = minmod(uip1 - ui, uip2 - uip1);
        UL[var] = ui + 0.5 * slope_L;
        UR[var] = uip1 - 0.5 * slope_R;
    }
    return Kokkos::make_pair(UL, UR);
}

KOKKOS_INLINE_FUNCTION
core::FieldArray compute_field_flux_hll(const core::FieldArray& UL, const core::FieldArray& UR) {
    core::FieldArray FL = core::get_em_flux(UL);
    core::FieldArray FR = core::get_em_flux(UR);

    const double SL = -core::c;
    const double SR = core::c;

    core::FieldArray F;
    for (int var = 0; var < core::N_FIELD; ++var) {
        F[var] = (SR * FL[var] - SL * FR[var] + SL * SR * (UR[var] - UL[var])) / (SR - SL);
    }
    return F;
}

} // namespace field_flux

#endif // FIELD_FLUX_SOLVER_HPP
