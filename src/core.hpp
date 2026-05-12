#ifndef CORE_HPP
#define CORE_HPP

#include <Kokkos_Core.hpp>


namespace core {

// grid parameters
const int Nx = 1000;
const int N_ghost = 2;
const int Nx_total = Nx + 2 * N_ghost;
// [lb, ub) are the indices of the physical cells
const int lb = N_ghost;
const int ub = N_ghost + Nx;

// moment
constexpr int N_MOMENT = 5;

constexpr int RHO = 0;
constexpr int MX = 1;
constexpr int MY = 2;
constexpr int MZ = 3;
constexpr int ENE = 4;

constexpr int UX = 1;
constexpr int UY = 2;
constexpr int UZ = 3;
constexpr int PRS = 4;

const double gamma = 1.4;

using MomentArray = Kokkos::Array<double, N_MOMENT>;

struct Species {
    double m, q;
    Kokkos::View<double**> U;
    Kokkos::View<double**> F;

    Species(double m, double q) : m(m), q(q) {
        U = Kokkos::View<double**>("Moment", N_MOMENT, Nx_total);
        F = Kokkos::View<double**>("Flux", N_MOMENT, Nx_total - 1); // F(i) = F_{i+1/2}
    }
};

KOKKOS_INLINE_FUNCTION
MomentArray get_moment_primitives(const MomentArray& U) {
    const double rho = U[RHO];
    const double ux = U[MX] / rho;
    const double uy = U[MY] / rho;
    const double uz = U[MZ] / rho;
    const double kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz);
    const double p = (core::gamma - 1.0) * (U[ENE] - kinetic);

    return MomentArray{rho, ux, uy, uz, p};
}

KOKKOS_INLINE_FUNCTION
MomentArray get_moment_conservatives(const MomentArray& Uprim) {
    const double rho = Uprim[RHO];
    const double ux = Uprim[UX];
    const double uy = Uprim[UY];
    const double uz = Uprim[UZ];
    const double p = Uprim[PRS];
    double kinetic = 0.5 * rho * (ux*ux + uy*uy + uz*uz);

    return MomentArray{rho, rho * ux, rho * uy, rho * uz, p / (core::gamma - 1.0) + kinetic};
}

KOKKOS_INLINE_FUNCTION
MomentArray get_moment_flux(const MomentArray& U) {
    MomentArray Uprim = get_moment_primitives(U);

    const double rho = U[RHO];
    const double rho_ux = U[MX];
    const double rho_uy = U[MY];
    const double rho_uz = U[MZ];
    const double energy = U[ENE];
    const double ux = Uprim[UX];
    const double p = Uprim[PRS];

    return MomentArray{rho * ux, rho_ux * ux + p, rho_uy * ux, rho_uz * ux, (energy + p) * ux};
}

// field
constexpr int N_FIELD = 6; // Ex, Ey, Ez, Bx, By, Bz

constexpr int EX = 0;
constexpr int EY = 1;
constexpr int EZ = 2;
constexpr int BX = 3;
constexpr int BY = 4;
constexpr int BZ = 5;

const double c = 20; // speed of light
const double epsilon0 = 1.0; // vacuum permittivity
const double mu0 = 1.0 / (epsilon0 * c * c); // vacuum permeability

inline Kokkos::View<double**> U_em; // (N_FIELD, Nx_total)
inline Kokkos::View<double**> F_em; // (N_FIELD, Nx_total - 1)

using FieldArray = Kokkos::Array<double, N_FIELD>;

KOKKOS_INLINE_FUNCTION
FieldArray get_em_flux(const FieldArray& U) {
    const double Ex = U[EX];
    const double Ey = U[EY];
    const double Ez = U[EZ];
    const double Bx = U[BX];
    const double By = U[BY];
    const double Bz = U[BZ];

    double c2 = c * c;

    return FieldArray{
        0.0, c2 * Bz, - c2 * By,
        0.0, - Ez, Ey
    };
}

// local vector operations
struct Vector3D {
    double vx, vy, vz;

    KOKKOS_INLINE_FUNCTION
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D{vx + other.vx, vy + other.vy, vz + other.vz};
    }

    KOKKOS_INLINE_FUNCTION
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D{vx - other.vx, vy - other.vy, vz - other.vz};
    }

    KOKKOS_INLINE_FUNCTION
    Vector3D operator*(double scalar) const {
        return Vector3D{vx * scalar, vy * scalar, vz * scalar};
    }
};

KOKKOS_INLINE_FUNCTION
double dot_product(const Vector3D& a, const Vector3D& b) {
    return a.vx * b.vx + a.vy * b.vy + a.vz * b.vz;
}

KOKKOS_INLINE_FUNCTION
Vector3D cross_product(const Vector3D& a, const Vector3D& b) {
    return Vector3D{
        a.vy * b.vz - a.vz * b.vy,
        a.vz * b.vx - a.vx * b.vz,
        a.vx * b.vy - a.vy * b.vx
    };
}

KOKKOS_INLINE_FUNCTION
Vector3D projected_vector(const Vector3D& a, const Vector3D& v_onto) {
    const double v_onto2 = dot_product(v_onto, v_onto);
    if (v_onto2 < 1e-12) {
        return Vector3D{0.0, 0.0, 0.0};
    }
    return v_onto * (dot_product(a, v_onto) / v_onto2);
}

} // namespace core

#endif // CORE_HPP
