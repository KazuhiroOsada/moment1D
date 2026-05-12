#include <cmath>

#include "core.hpp"
#include "diag.hpp"
#include "push_hyperbolic.hpp"


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        core::U_em = Kokkos::View<double**>("U_em", core::N_FIELD, core::Nx_total);
        core::F_em = Kokkos::View<double**>("F_em", core::N_FIELD, core::Nx_total - 1);
        pushH::rk_temp_field = Kokkos::View<double**>("RK_temp_field", core::N_FIELD, core::Nx_total);
        auto U_em = core::U_em;

        const int ix_center = core::lb + core::Nx / 4;
        const double width = 0.05 * static_cast<double>(core::Nx);
        const double amp = 1.0e-3;

        Kokkos::parallel_for("initialize_em_wave", core::Nx_total, KOKKOS_LAMBDA(const int ix) {
            for (int var = 0; var < core::N_FIELD; ++var) {
                U_em(var, ix) = 0.0;
            }

            const double x = static_cast<double>(ix - ix_center);
            const double profile = amp * Kokkos::exp(-(x * x) / (2.0 * width * width));
            U_em(core::EY, ix) = profile;
            U_em(core::BZ, ix) = profile / core::c;
        });

        Kokkos::fence();

        const double dt = 0.01;
        const int max_iters = 5001;
        const int diag_interval = 50;
        diag::write_parameters("em_wave", dt, max_iters, diag_interval);

        for (int it = 0; it < max_iters; ++it) {
            if (it % diag_interval == 0) {
                diag::diag_field(core::U_em, it);
            }
            pushH::push_field_rk2(dt);
        }

        pushH::rk_temp_field = Kokkos::View<double**>();
        core::F_em = Kokkos::View<double**>();
        core::U_em = Kokkos::View<double**>();
    }
    Kokkos::finalize();
    return 0;
}
