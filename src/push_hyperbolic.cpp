#include "push_hyperbolic.hpp"


namespace pushH {

void push_moment_rk2(core::Species& species, const double dt) {
    auto U = species.U;
    auto F = species.F;
    auto rk_temp = pushH::rk_temp_moment;

    // Preserve all cells in the temporary state, then update only physical cells.
    Kokkos::deep_copy(rk_temp, U);

    Kokkos::parallel_for("compute_flux_at_current_step", Kokkos::RangePolicy<>(core::lb - 1, core::ub), KOKKOS_LAMBDA(int hx) {
        const auto [UL, UR] = moment_flux::reconstruct_moment_muscl(U, hx);
        const core::MomentArray F_local = moment_flux::compute_moment_flux_hll(UL, UR);
        for (int var = 0; var < core::N_MOMENT; ++var) {
            F(var, hx) = F_local[var];
        }
    });

    Kokkos::parallel_for("update_moment_half_step", Kokkos::RangePolicy<>(core::lb, core::ub), KOKKOS_LAMBDA(int ix) {
        for (int var = 0; var < core::N_MOMENT; ++var) {
            rk_temp(var, ix) = U(var, ix) - (0.5 * dt) * (F(var, ix) - F(var, ix - 1));
        }
    });

    Kokkos::parallel_for("compute_flux_at_half_step", Kokkos::RangePolicy<>(core::lb - 1, core::ub), KOKKOS_LAMBDA(int hx) {
        const auto [UL, UR] = moment_flux::reconstruct_moment_muscl(rk_temp, hx);
        const core::MomentArray F_local = moment_flux::compute_moment_flux_hll(UL, UR);
        for (int var = 0; var < core::N_MOMENT; ++var) {
            F(var, hx) = F_local[var];
        }
    });

    Kokkos::parallel_for("update_moment_full_step", Kokkos::RangePolicy<>(core::lb, core::ub), KOKKOS_LAMBDA(int ix) {
        for (int var = 0; var < core::N_MOMENT; ++var) {
            U(var, ix) = U(var, ix) - dt * (F(var, ix) - F(var, ix - 1));
        }
    });
}

void push_field_rk2(const double dt) {
    auto U = core::U_em;
    auto F = core::F_em;
    auto rk_temp = pushH::rk_temp_field;

    Kokkos::deep_copy(rk_temp, U);

    Kokkos::parallel_for("compute_field_flux_at_current_step", Kokkos::RangePolicy<>(core::lb - 1, core::ub), KOKKOS_LAMBDA(int hx) {
        const auto [UL, UR] = field_flux::reconstruct_field_muscl(U, hx);
        const core::FieldArray F_local = field_flux::compute_field_flux_hll(UL, UR);
        for (int var = 0; var < core::N_FIELD; ++var) {
            F(var, hx) = F_local[var];
        }
    });

    Kokkos::parallel_for("update_field_half_step", Kokkos::RangePolicy<>(core::lb, core::ub), KOKKOS_LAMBDA(int ix) {
        for (int var = 0; var < core::N_FIELD; ++var) {
            rk_temp(var, ix) = U(var, ix) - (0.5 * dt) * (F(var, ix) - F(var, ix - 1));
        }
    });

    Kokkos::parallel_for("compute_field_flux_at_half_step", Kokkos::RangePolicy<>(core::lb - 1, core::ub), KOKKOS_LAMBDA(int hx) {
        const auto [UL, UR] = field_flux::reconstruct_field_muscl(rk_temp, hx);
        const core::FieldArray F_local = field_flux::compute_field_flux_hll(UL, UR);
        for (int var = 0; var < core::N_FIELD; ++var) {
            F(var, hx) = F_local[var];
        }
    });

    Kokkos::parallel_for("update_field_full_step", Kokkos::RangePolicy<>(core::lb, core::ub), KOKKOS_LAMBDA(int ix) {
        for (int var = 0; var < core::N_FIELD; ++var) {
            U(var, ix) = U(var, ix) - dt * (F(var, ix) - F(var, ix - 1));
        }
    });
}

} // namespace pushH
