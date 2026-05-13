#ifndef MOMENT_FLUX_SOLVER_HPP
#define MOMENT_FLUX_SOLVER_HPP

#include <Kokkos_Core.hpp>

#include "core.hpp"


namespace moment_flux {

KOKKOS_INLINE_FUNCTION
double minmod(const double a, const double b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (Kokkos::fabs(a) < Kokkos::fabs(b)) ? a : b;
}

KOKKOS_INLINE_FUNCTION
Kokkos::pair<core::MomentArray, core::MomentArray> reconstruct_moment_muscl(const Kokkos::View<double**>& U, int hx) {
    core::MomentArray U_im1 = {U(core::RHO, hx - 1), U(core::MX, hx - 1), U(core::MY, hx - 1), U(core::MZ, hx - 1), U(core::ENE, hx - 1)};
    core::MomentArray prim_im1 = core::get_moment_primitives(U_im1);

    core::MomentArray U_i = {U(core::RHO, hx), U(core::MX, hx), U(core::MY, hx), U(core::MZ, hx), U(core::ENE, hx)};
    core::MomentArray prim_i = core::get_moment_primitives(U_i);

    core::MomentArray U_ip1 = {U(core::RHO, hx + 1), U(core::MX, hx + 1), U(core::MY, hx + 1), U(core::MZ, hx + 1), U(core::ENE, hx + 1)};
    core::MomentArray prim_ip1 = core::get_moment_primitives(U_ip1);

    core::MomentArray U_ip2 = {U(core::RHO, hx + 2), U(core::MX, hx + 2), U(core::MY, hx + 2), U(core::MZ, hx + 2), U(core::ENE, hx + 2)};
    core::MomentArray prim_ip2 = core::get_moment_primitives(U_ip2);

    core::MomentArray prim_L, prim_R;
    for (int var = 0; var < core::N_MOMENT; ++var) {
        double slope_L = minmod(prim_i[var] - prim_im1[var], prim_ip1[var] - prim_i[var]);
        double slope_R = minmod(prim_ip1[var] - prim_i[var], prim_ip2[var] - prim_ip1[var]);
        prim_L[var] = prim_i[var] + 0.5 * slope_L;
        prim_R[var] = prim_ip1[var] - 0.5 * slope_R;
    }

    core::MomentArray UL = core::get_moment_conservatives(prim_L);
    core::MomentArray UR = core::get_moment_conservatives(prim_R);
    return Kokkos::make_pair(UL, UR);
}

KOKKOS_INLINE_FUNCTION
core::MomentArray compute_moment_flux_hll(const core::MomentArray& UL, const core::MomentArray& UR) {
    // compute primitives
    core::MomentArray UprimL = core::get_moment_primitives(UL);
    core::MomentArray UprimR = core::get_moment_primitives(UR);
    const double rhoL = UprimL[core::RHO];
    const double uxL = UprimL[core::UX];
    const double pL = UprimL[core::PRS];
    const double rhoR = UprimR[core::RHO];
    const double uxR = UprimR[core::UX];
    const double pR = UprimR[core::PRS];

    // compute the propagation speeds
    double cL = Kokkos::sqrt(core::gamma * pL / rhoL);
    double cR = Kokkos::sqrt(core::gamma * pR / rhoR);
    double SL = Kokkos::fmin(Kokkos::fmin(uxL - cL, uxR - cR), 0.0);
    double SR = Kokkos::fmax(Kokkos::fmax(uxL + cL, uxR + cR), 0.0);

    // compute the fluxes for left and right states
    core::MomentArray FL = core::get_moment_flux(UL);
    core::MomentArray FR = core::get_moment_flux(UR);

    core::MomentArray F;
    // compute the HLL flux
    for (int var = 0; var < core::N_MOMENT; ++var) {
        F[var] = (SR * FL[var] - SL * FR[var] + SL * SR * (UR[var] - UL[var])) / (SR - SL);
    }
    return F;
}

} // namespace moment_flux

#endif // MOMENT_FLUX_SOLVER_HPP
