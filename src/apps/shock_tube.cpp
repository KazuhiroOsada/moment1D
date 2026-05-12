#include <iostream>
#include <cmath>

#include "core.hpp"
#include "push_hyperbolic.hpp"
#include "diag.hpp"


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        core::Species ions(1.0, 1.0);
        pushH::rk_temp_moment = Kokkos::View<double**>("RK_temp", core::N_MOMENT, core::Nx_total);

        // SOD's shock tube initial condition
        const int ix_split = core::lb + core::Nx / 2;
        Kokkos::parallel_for("initialize_moments", core::Nx_total, KOKKOS_LAMBDA(int ix) {
            if (ix < ix_split) {
                ions.U(core::RHO, ix) = 1.0; // density
                ions.U(core::MX, ix) = 0.0; // momentum x
                ions.U(core::MY, ix) = 0.0; // momentum y
                ions.U(core::MZ, ix) = 0.0; // momentum z
                ions.U(core::ENE, ix) = 1.0; // energy
            } else {
                ions.U(core::RHO, ix) = 0.125;
                ions.U(core::MX, ix) = 0.0;
                ions.U(core::MY, ix) = 0.0;
                ions.U(core::MZ, ix) = 0.0;
                ions.U(core::ENE, ix) = 0.1;
            }
        });

        Kokkos::fence();

        double const dt = 0.01;
        int const max_iters = 20001;
        int const diag_interval = 100;
        diag::write_parameters("shock_tube", dt, max_iters, diag_interval);

        for (int it = 0; it < max_iters; ++it) {
            if (it % diag_interval == 0) {
                diag::diag_moment(ions, "ions", it);
            }
            pushH::push_moment_rk2(ions, dt);
        }

        pushH::rk_temp_moment = Kokkos::View<double**>(); // free the temporary view
    }
    Kokkos::finalize();
    return 0;
}
