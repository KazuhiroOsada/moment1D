#include <cmath>
#include <iostream>

#include "core.hpp"
#include "push_source.hpp"

namespace {

bool close_enough(const double a, const double b, const double tol = 1e-10) {
    return std::fabs(a - b) <= tol;
}

bool expect_close(const double actual, const double expected, const char* label, const double tol = 1e-10) {
    if (!close_enough(actual, expected, tol)) {
        std::cerr << "FAIL: " << label << " (actual=" << actual << ", expected=" << expected << ")\n";
        return false;
    }
    return true;
}

bool expect_vector_close(const core::Vector3D& actual, const core::Vector3D& expected, const char* label,
                         const double tol = 1e-10) {
    bool ok = true;
    if (!close_enough(actual.vx, expected.vx, tol)) {
        ok = false;
        std::cerr << "FAIL: " << label << " at vx (actual=" << actual.vx << ", expected=" << expected.vx << ")\n";
    }
    if (!close_enough(actual.vy, expected.vy, tol)) {
        ok = false;
        std::cerr << "FAIL: " << label << " at vy (actual=" << actual.vy << ", expected=" << expected.vy << ")\n";
    }
    if (!close_enough(actual.vz, expected.vz, tol)) {
        ok = false;
        std::cerr << "FAIL: " << label << " at vz (actual=" << actual.vz << ", expected=" << expected.vz << ")\n";
    }
    return ok;
}

template<int N_SPECIES>
void initialize_state(Kokkos::Array<core::MomentArray, N_SPECIES>& U_sp_local,
                      Kokkos::Array<double, N_SPECIES>& m_sp,
                      Kokkos::Array<double, N_SPECIES>& q_sp,
                      core::FieldArray& U_em_local) {
    m_sp[0] = 1.0;
    q_sp[0] = 1.0;
    U_sp_local[0] = {2.0, 0.3, -0.1, 0.2, 1.0};

    m_sp[1] = 4.0;
    q_sp[1] = -1.0;
    U_sp_local[1] = {1.5, -0.2, 0.4, 0.1, 0.8};

    U_em_local = {0.1, -0.05, 0.2, 0.4, -0.2, 0.3};
}

template<int N_SPECIES>
double compute_total_energy(const Kokkos::Array<core::MomentArray, N_SPECIES>& U_sp_local,
                            const Kokkos::Array<double, N_SPECIES>& m_sp,
                            const Kokkos::Array<double, N_SPECIES>& q_sp,
                            const core::FieldArray& U_em_local) {
    const core::Vector3D E{U_em_local[core::EX], U_em_local[core::EY], U_em_local[core::EZ]};
    double total = 0.5 * core::epsilon0 * core::dot_product(E, E);
    for (int is = 0; is < N_SPECIES; ++is) {
        const double omega_p_sq =
            pushS::calc_plasma_frequency_squared(m_sp[is], q_sp[is], U_sp_local[is][core::RHO]);
        const core::Vector3D momentum{
            U_sp_local[is][core::MX], U_sp_local[is][core::MY], U_sp_local[is][core::MZ]};
        const core::Vector3D J = (q_sp[is] / m_sp[is]) * momentum;
        total += 0.5 * core::dot_product(J, J) / (core::epsilon0 * omega_p_sq);
    }
    return total;
}

bool test_locally_implicit_residuals() {
    constexpr int N_SPECIES = 2;
    const double dt = 0.05;

    Kokkos::Array<core::MomentArray, N_SPECIES> U_sp_local;
    Kokkos::Array<double, N_SPECIES> m_sp;
    Kokkos::Array<double, N_SPECIES> q_sp;
    core::FieldArray U_em_local;
    initialize_state<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local);
    const core::Vector3D E_n{U_em_local[core::EX], U_em_local[core::EY], U_em_local[core::EZ]};
    const core::Vector3D B{U_em_local[core::BX], U_em_local[core::BY], U_em_local[core::BZ]};

    core::Vector3D J_n_sp[N_SPECIES];
    for (int is = 0; is < N_SPECIES; ++is) {
        const core::Vector3D momentum{
            U_sp_local[is][core::MX], U_sp_local[is][core::MY], U_sp_local[is][core::MZ]};
        J_n_sp[is] = (q_sp[is] / m_sp[is]) * momentum;
    }

    pushS::push_source_locally<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local, dt);

    const core::Vector3D E_new{U_em_local[core::EX], U_em_local[core::EY], U_em_local[core::EZ]};
    const core::Vector3D E_bar = 0.5 * (E_new + E_n);

    core::Vector3D J_bar_sum{0.0, 0.0, 0.0};
    bool ok = true;
    for (int is = 0; is < N_SPECIES; ++is) {
        const core::Vector3D momentum_new{
            U_sp_local[is][core::MX], U_sp_local[is][core::MY], U_sp_local[is][core::MZ]};
        const core::Vector3D J_new = (q_sp[is] / m_sp[is]) * momentum_new;
        const core::Vector3D J_bar = 0.5 * (J_new + J_n_sp[is]);
        J_bar_sum = J_bar_sum + J_bar;

        const double omega_p_sq =
            pushS::calc_plasma_frequency_squared(m_sp[is], q_sp[is], U_sp_local[is][core::RHO]);
        const core::Vector3D Omega_c = pushS::calc_cyclotron_frequency(m_sp[is], q_sp[is], B);
        const core::Vector3D rhs =
            J_n_sp[is]
            + 0.5 * dt * (omega_p_sq * core::epsilon0 * E_bar + core::cross_product(J_bar, Omega_c));
        const core::Vector3D residual = J_bar - rhs;

        ok = expect_vector_close(residual, core::Vector3D{0.0, 0.0, 0.0}, "implicit J_bar residual") && ok;
    }

    const core::Vector3D E_rhs = E_n - (0.5 * dt / core::epsilon0) * J_bar_sum;
    const core::Vector3D E_residual = E_bar - E_rhs;
    ok = expect_vector_close(E_residual, core::Vector3D{0.0, 0.0, 0.0}, "implicit E_bar residual") && ok;

    return ok;
}

bool test_locally_implicit_energy_conservation() {
    constexpr int N_SPECIES = 2;
    const double dt = 0.05;

    Kokkos::Array<core::MomentArray, N_SPECIES> U_sp_local;
    Kokkos::Array<double, N_SPECIES> m_sp;
    Kokkos::Array<double, N_SPECIES> q_sp;
    core::FieldArray U_em_local;
    initialize_state<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local);

    const double energy_before = compute_total_energy<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local);
    pushS::push_source_locally<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local, dt);
    const double energy_after = compute_total_energy<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local);

    return expect_close(energy_after, energy_before, "energy conservation", 1e-9);
}

} // namespace

int main() {
    Kokkos::initialize();
    bool ok = true;
    {
        ok = test_locally_implicit_residuals() && ok;
        ok = test_locally_implicit_energy_conservation() && ok;
    }
    Kokkos::finalize();

    if (!ok) {
        std::cerr << "push_source tests failed.\n";
        return 1;
    }
    std::cout << "push_source tests passed.\n";
    return 0;
}
