#include "core.hpp"
#include "diag.hpp"
#include "push_source.hpp"


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        core::Species electrons(1.0, -1.0);
        core::U_em = Kokkos::View<double**>("U_em", core::N_FIELD, core::Nx_total);

        // uniform Ex and stationary electrons
        Kokkos::parallel_for("initialize", core::Nx_total, KOKKOS_LAMBDA(int ix) {
            electrons.U(core::RHO, ix) = 1.0; // density
            electrons.U(core::MX, ix) = 0.0; // momentum x
            electrons.U(core::MY, ix) = 0.0; // momentum y
            electrons.U(core::MZ, ix) = 0.0; // momentum z
            electrons.U(core::ENE, ix) = 0.0; // energy

            core::U_em(core::EX, ix) = 1.0; // uniform electric field
            core::U_em(core::EY, ix) = 0.0;
            core::U_em(core::EZ, ix) = 0.0;
            core::U_em(core::BX, ix) = 0.0;
            core::U_em(core::BY, ix) = 0.0;
            core::U_em(core::BZ, ix) = 0.0;
        });

        Kokkos::fence();

        const double dt = 10;
        int max_iters = 5001;
        int diag_interval = 10;
        diag::write_parameters("plasma_oscillation", dt, max_iters, diag_interval);

        for (int it = 0; it < max_iters; ++it) {
            if (it % diag_interval == 0) {
                diag::diag_moment(electrons, "electrons", it);
                diag::diag_field(core::U_em, it);
            }
            Kokkos::Array<core::Species, 1> species = {electrons};
            pushS::push_source<1>(species, core::U_em, dt);
        }
        core::U_em = Kokkos::View<double**>();
    }
    Kokkos::finalize();
    return 0;
}
