#include "core.hpp"
#include "diag.hpp"
#include "push_hyperbolic.hpp"
#include "push_source.hpp"


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        core::Species electrons(1.0, -1.0);
        pushH::rk_temp_moment = Kokkos::View<double**>("RK_temp", core::N_MOMENT, core::Nx_total);

        core::U_em = Kokkos::View<double**>("U_em", core::N_FIELD, core::Nx_total);
        core::F_em = Kokkos::View<double**>("F_em", core::N_FIELD, core::Nx_total - 1);
        pushH::rk_temp_field = Kokkos::View<double**>("RK_temp_field", core::N_FIELD, core::Nx_total);
        auto U_em = core::U_em;

        int const ix_foot = core::lb + core::Nx / 4;
        double const rho_max = 20;
        Kokkos::parallel_for("initialize", core::Nx_total, KOKKOS_LAMBDA(int ix) {
            const double l = static_cast<double>(Kokkos::max(0, ix - ix_foot)) / (core::Nx - ix_foot);
            electrons.U(core::RHO, ix) = 1e-2 + rho_max * Kokkos::pow(l, 2);
            electrons.U(core::MX, ix) = 0.0;
            electrons.U(core::MY, ix) = 0.0;
            electrons.U(core::MZ, ix) = 0.0;
            electrons.U(core::ENE, ix) = 1e-6; 

            U_em(core::EX, ix) = 0.0;
            U_em(core::EY, ix) = 0.0;
            U_em(core::EZ, ix) = 0.0;
            U_em(core::BX, ix) = 0.0;
            U_em(core::BY, ix) = 0.0;
            U_em(core::BZ, ix) = 0.0;
        });

        Kokkos::fence();

        const double E0 = 1.0;
        const double B0 = E0 / core::c;
        const double omega_force = 2.0;

        const double dt = 0.01;
        int const max_iters = 5001;
        int const diag_interval = 50;
        diag::write_parameters("plasma_wave_beach", dt, max_iters, diag_interval);

        for (int it = 0; it < max_iters; ++it) {
            if (it % diag_interval == 0) {
                diag::diag_moment(electrons, "electrons", it);
                diag::diag_field(core::U_em, it);
            }
            Kokkos::Array<core::Species, 1> species = {electrons};

            pushS::push_source<1>(species, core::U_em, dt/2);
            pushH::push_moment_rk2(electrons, dt);
            pushH::push_field_rk2(dt);
            pushS::push_source<1>(species, core::U_em, dt/2);

            for (int ix = 0; ix < core::lb + 1; ++ix) {
                U_em(core::EY, ix) = E0 * Kokkos::sin(omega_force * (it * dt - ix / core::c));
                U_em(core::BZ, ix) = B0 * Kokkos::sin(omega_force * (it * dt - ix / core::c));
            }
        }
        pushH::rk_temp_moment = Kokkos::View<double**>();
        pushH::rk_temp_field = Kokkos::View<double**>();
        core::F_em = Kokkos::View<double**>();
        core::U_em = Kokkos::View<double**>();
    }
    Kokkos::finalize();
    return 0;
}