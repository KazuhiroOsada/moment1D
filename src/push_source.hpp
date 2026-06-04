#ifndef PUSH_SOURCE_HPP
#define PUSH_SOURCE_HPP

#include <Kokkos_Core.hpp>

#include "core.hpp"

namespace pushS {

using core::Vector3D;

KOKKOS_INLINE_FUNCTION
double calc_plasma_frequency_squared(const double mass, const double charge, const double rho) {
    const double n = rho / mass;
    return (charge * charge * n) / (core::epsilon0 * mass);
}

KOKKOS_INLINE_FUNCTION
Vector3D calc_cyclotron_frequency(const double mass, const double charge, const Vector3D& B) {
    return B * (charge / mass);
}

KOKKOS_INLINE_FUNCTION
Vector3D solve_implicit_rotation(const Vector3D& v, const Vector3D& b, const Vector3D& Omega_c, const double dt) {
    const double dt2 = dt * dt;
    const double factor = 0.25 * core::dot_product(Omega_c, Omega_c) * dt2;
    const double sqrt_factor = 0.5 * core::dot_product(Omega_c, b) * dt;

    return 1.0 / (1 + factor) * (v + factor * core::projected_vector(v, b) - sqrt_factor * core::cross_product(b, v));
}

KOKKOS_INLINE_FUNCTION
Vector3D compute_K_s(const Vector3D& J_s, const Vector3D& B, const Vector3D& Omega_c, const double dt) {
    const Vector3D b = core::unit_vector(B);
    const Vector3D K = -dt * solve_implicit_rotation(J_s, b, Omega_c, dt);
    return K;
}

template<int N_SPECIES>
KOKKOS_INLINE_FUNCTION
Vector3D compute_E_bar(const Vector3D& E, const Vector3D& B, const Vector3D& K,
                       const Kokkos::Array<double, N_SPECIES>& omega_p_sq_sp,
                       const Kokkos::Array<Vector3D, N_SPECIES>& Omega_c_sp,
                       const double dt) {
    const Vector3D b = core::unit_vector(B);
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    // compute local coefficients
    double omega_0_sq = 0.0;
    double gamma_sq   = 0.0;
    double delta      = 0.0;
    for (int is = 0; is < N_SPECIES; ++is) {
        const double Omega_c_sq = core::dot_product(Omega_c_sp[is], Omega_c_sp[is]);
        const double denom      = 1.0 + 0.25 * Omega_c_sq * dt2;
        const double omega_p_sq = omega_p_sq_sp[is];
        omega_0_sq += omega_p_sq * dt2 / denom;
        gamma_sq   += omega_p_sq * Omega_c_sq * dt4 / denom;
        delta      += omega_p_sq * core::dot_product(Omega_c_sp[is], b) * dt3 / denom;
    }
    const double factor = 1.0 + 0.25 * omega_0_sq;
    const double Delta_sq = delta * delta / factor;

    const Vector3D F_plus_half_K = core::epsilon0 * E + 0.5 * K;
    const double delta_term = Delta_sq / 64.0;
    const double gamma_term = gamma_sq / 16.0;
    const double denom1 = factor + delta_term;
    const double denom2 = factor + gamma_term;
    const Vector3D projected = core::projected_vector(F_plus_half_K, b);
    const Vector3D crossed = core::cross_product(b, F_plus_half_K);

    const Vector3D E_bar = (1.0 / core::epsilon0) * (
                                (1.0 / denom1) * F_plus_half_K
                              + (delta_term - gamma_term) / (denom1 * denom2) * projected
                              + (delta / 8.0) / (factor * denom1) * crossed
                           );
    return E_bar;
}

KOKKOS_INLINE_FUNCTION
Vector3D compute_J_s_bar(const Vector3D& J_s, const Vector3D& E_bar, const Vector3D& B,
                         const double omega_p_sq, const Vector3D& Omega_c, const double dt) {   
    const Vector3D J_s_star = J_s + 0.5 * core::epsilon0 * omega_p_sq * dt * E_bar;
    
    const Vector3D b = core::unit_vector(B); 
    const Vector3D J_s_bar = solve_implicit_rotation(J_s_star, b, Omega_c, dt);
    return J_s_bar;
}

template<int N_SPECIES>
KOKKOS_INLINE_FUNCTION
void push_source_locally(Kokkos::Array<core::MomentArray, N_SPECIES>& U_sp_local,
                         const Kokkos::Array<double, N_SPECIES>& m_sp,
                         const Kokkos::Array<double, N_SPECIES>& q_sp,
                         core::FieldArray& U_em_local,
                         const double dt) {
    Vector3D E = {U_em_local[core::EX], U_em_local[core::EY], U_em_local[core::EZ]};
    Vector3D B = {U_em_local[core::BX], U_em_local[core::BY], U_em_local[core::BZ]};

    Kokkos::Array<double, N_SPECIES> omega_p_sq_sp;
    Kokkos::Array<Vector3D, N_SPECIES> Omega_c_sp;
    Kokkos::Array<Vector3D, N_SPECIES> J_n_sp;
    for (int is = 0; is < N_SPECIES; ++is) {
        omega_p_sq_sp[is] = calc_plasma_frequency_squared(m_sp[is], q_sp[is], U_sp_local[is][core::RHO]);
        Omega_c_sp[is] = calc_cyclotron_frequency(m_sp[is], q_sp[is], B);
        const Vector3D momentum = {U_sp_local[is][core::MX], U_sp_local[is][core::MY], U_sp_local[is][core::MZ]};
        J_n_sp[is] = (q_sp[is] / m_sp[is]) * momentum;
    }

    Vector3D K = {0.0, 0.0, 0.0};
    for (int is = 0; is < N_SPECIES; ++is) {
        Vector3D Ks = compute_K_s(J_n_sp[is], B, Omega_c_sp[is], dt);
        K = K + Ks;
    }

    Vector3D E_bar = compute_E_bar<N_SPECIES>(E, B, K, omega_p_sq_sp, Omega_c_sp, dt);
    Vector3D E_new = 2.0 * E_bar - E;
    U_em_local[core::EX] = E_new.vx;
    U_em_local[core::EY] = E_new.vy;
    U_em_local[core::EZ] = E_new.vz;

    for (int is = 0; is < N_SPECIES; ++is) {
        Vector3D J_bar = compute_J_s_bar(J_n_sp[is], E_bar, B, omega_p_sq_sp[is], Omega_c_sp[is], dt);
        Vector3D J_new = 2.0 * J_bar - J_n_sp[is];

        const double m_over_q = m_sp[is] / q_sp[is];
        U_sp_local[is][core::MX] = m_over_q * J_new.vx;
        U_sp_local[is][core::MY] = m_over_q * J_new.vy;
        U_sp_local[is][core::MZ] = m_over_q * J_new.vz;
    }
}

template<int N_SPECIES>
void push_source(const Kokkos::Array<core::Species, N_SPECIES>& species_array, Kokkos::View<double**> U_em, const double dt) {
    const auto species = species_array;
    Kokkos::parallel_for("push_source", Kokkos::RangePolicy<>(core::lb, core::ub), KOKKOS_LAMBDA(int ix) {
        Kokkos::Array<core::MomentArray, N_SPECIES> U_sp_local;
        Kokkos::Array<double, N_SPECIES> m_sp;
        Kokkos::Array<double, N_SPECIES> q_sp;
        for (int is = 0; is < N_SPECIES; ++is) {
            U_sp_local[is] = {species[is].U(core::RHO, ix), species[is].U(core::MX, ix), species[is].U(core::MY, ix), species[is].U(core::MZ, ix), species[is].U(core::ENE, ix)};
            m_sp[is] = species[is].m;
            q_sp[is] = species[is].q;
        }
        core::FieldArray U_em_local = {U_em(core::EX, ix), U_em(core::EY, ix), U_em(core::EZ, ix), U_em(core::BX, ix), U_em(core::BY, ix), U_em(core::BZ, ix)};
        push_source_locally<N_SPECIES>(U_sp_local, m_sp, q_sp, U_em_local, dt);
        // write back the updated values to global memory
        for (int is = 0; is < N_SPECIES; ++is) {
            species[is].U(core::MX, ix) = U_sp_local[is][core::MX];
            species[is].U(core::MY, ix) = U_sp_local[is][core::MY];
            species[is].U(core::MZ, ix) = U_sp_local[is][core::MZ];
        }
        U_em(core::EX, ix) = U_em_local[core::EX];
        U_em(core::EY, ix) = U_em_local[core::EY];
        U_em(core::EZ, ix) = U_em_local[core::EZ];
    });
}

} // namespace pushS

#endif // PUSH_SOURCE_HPP
