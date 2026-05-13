#include <cmath>
#include <iostream>

#include "core.hpp"
#include "moment_flux_solver.hpp"

namespace {

core::MomentArray conservative_from_primitive(const double rho, const double ux, const double uy, const double uz,
                                              const double p) {
    core::MomentArray Uprim = {rho, ux, uy, uz, p};
    core::MomentArray U = core::get_moment_conservatives(Uprim);
    return U;
}

bool close_enough(const double a, const double b, const double tol = 1e-12) {
    return std::fabs(a - b) <= tol;
}

bool expect_array_close(const core::MomentArray& actual, const core::MomentArray& expected, const char* label,
                        const double tol = 1e-12) {
    bool ok = true;
    for (int i = 0; i < core::N_MOMENT; ++i) {
        if (!close_enough(actual[i], expected[i], tol)) {
            ok = false;
            std::cerr << "FAIL: " << label << " at component " << i << " (actual=" << actual[i]
                      << ", expected=" << expected[i] << ")\n";
        }
    }
    return ok;
}

bool test_reconstruct_muscl_limited_linear() {
    Kokkos::View<double**> U("U", core::N_MOMENT, 5);
    auto U_h = Kokkos::create_mirror_view(U);

    for (int ix = 0; ix < 5; ++ix) {
        const double rho = 1.0 + 0.1 * ix;
        const double ux = 0.2 + 0.01 * ix;
        const double uy = -0.1 + 0.005 * ix;
        const double uz = 0.03 - 0.002 * ix;
        const double p = 0.9 + 0.02 * ix;
        const core::MomentArray U_cell = conservative_from_primitive(rho, ux, uy, uz, p);
        for (int i = 0; i < core::N_MOMENT; ++i) {
            U_h(i, ix) = U_cell[i];
        }
    }
    Kokkos::deep_copy(U, U_h);

    const auto [UL, UR] = moment_flux::reconstruct_moment_muscl(U, 2);

    const double rho_mid = 1.0 + 0.1 * 2.5;
    const double ux_mid = 0.2 + 0.01 * 2.5;
    const double uy_mid = -0.1 + 0.005 * 2.5;
    const double uz_mid = 0.03 - 0.002 * 2.5;
    const double p_mid = 0.9 + 0.02 * 2.5;
    const core::MomentArray expected_mid = conservative_from_primitive(rho_mid, ux_mid, uy_mid, uz_mid, p_mid);

    bool ok = true;
    for (int i = 0; i < core::N_MOMENT; ++i) {
        if (!close_enough(UL[i], expected_mid[i])) {
            ok = false;
            std::cerr << "FAIL: reconstruct_muscl UL at component " << i << "\n";
        }
        if (!close_enough(UR[i], expected_mid[i])) {
            ok = false;
            std::cerr << "FAIL: reconstruct_muscl UR at component " << i << "\n";
        }
    }
    return ok;
}

bool test_compute_flux_hll_equal_states() {
    const core::MomentArray U = conservative_from_primitive(1.3, 0.2, -0.1, 0.05, 0.9);

    const core::MomentArray F_hll = moment_flux::compute_moment_flux_hll(U, U);
    const core::MomentArray F_local = core::get_moment_flux(U);
    return expect_array_close(F_hll, F_local, "compute_flux_hll equal-state consistency");
}

bool test_compute_flux_hll_supersonic_right_returns_left_flux() {
    const core::MomentArray UL = conservative_from_primitive(1.0, 4.0, 0.0, 0.0, 0.4);
    const core::MomentArray UR = conservative_from_primitive(0.8, 3.5, 0.0, 0.0, 0.3);

    const core::MomentArray F_hll = moment_flux::compute_moment_flux_hll(UL, UR);
    const core::MomentArray FL = core::get_moment_flux(UL);
    return expect_array_close(F_hll, FL, "compute_flux_hll right-going supersonic");
}

} // namespace

int main() {
    Kokkos::initialize();
    bool ok = true;
    {
        ok = test_reconstruct_muscl_limited_linear() && ok;
        ok = test_compute_flux_hll_equal_states() && ok;
        ok = test_compute_flux_hll_supersonic_right_returns_left_flux() && ok;
    }
    Kokkos::finalize();

    if (!ok) {
        std::cerr << "moment_flux_solver tests failed.\n";
        return 1;
    }
    std::cout << "moment_flux_solver tests passed.\n";
    return 0;
}
