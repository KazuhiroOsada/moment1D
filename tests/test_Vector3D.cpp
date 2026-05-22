#include <cmath>
#include <iostream>

#include "core.hpp"

namespace {

bool close_enough(const double a, const double b, const double tol = 1e-12) {
    return std::fabs(a - b) <= tol;
}

bool expect_vector_close(const core::Vector3D& actual, const core::Vector3D& expected, const char* label,
                         const double tol = 1e-12) {
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

bool test_vector_add_sub_mul() {
    const core::Vector3D a{1.0, -2.0, 3.5};
    const core::Vector3D b{-4.0, 5.5, 0.5};

    bool ok = true;
    ok = expect_vector_close(a + b, core::Vector3D{-3.0, 3.5, 4.0}, "operator+") && ok;
    ok = expect_vector_close(a - b, core::Vector3D{5.0, -7.5, 3.0}, "operator-") && ok;
    ok = expect_vector_close(a * 2.5, core::Vector3D{2.5, -5.0, 8.75}, "operator*") && ok;
    return ok;
}

bool test_dot_product() {
    const core::Vector3D a{1.0, 2.0, 3.0};
    const core::Vector3D b{-4.0, 5.0, -6.0};
    const double expected = -12.0; // 1*(-4) + 2*5 + 3*(-6)
    const double actual = core::dot_product(a, b);
    if (!close_enough(actual, expected)) {
        std::cerr << "FAIL: dot_product (actual=" << actual << ", expected=" << expected << ")\n";
        return false;
    }
    return true;
}

bool test_cross_product() {
    const core::Vector3D ex{1.0, 0.0, 0.0};
    const core::Vector3D ey{0.0, 1.0, 0.0};
    const core::Vector3D ez{0.0, 0.0, 1.0};

    bool ok = true;
    ok = expect_vector_close(core::cross_product(ex, ey), ez, "cross_product(ex, ey)") && ok;
    ok = expect_vector_close(core::cross_product(ey, ex), core::Vector3D{0.0, 0.0, -1.0}, "cross_product(ey, ex)") && ok;
    return ok;
}

bool test_projected_vector_regular_case() {
    const core::Vector3D a{3.0, 4.0, 0.0};
    const core::Vector3D onto{2.0, 0.0, 0.0};
    const core::Vector3D expected{3.0, 0.0, 0.0};
    const core::Vector3D actual = core::projected_vector(a, onto);
    return expect_vector_close(actual, expected, "projected_vector regular case");
}

bool test_projected_vector_degenerate_direction() {
    const core::Vector3D a{1.0, -2.0, 3.0};
    const core::Vector3D tiny{1e-8, 1e-8, 1e-8}; // squared norm < 1e-12 threshold
    const core::Vector3D expected{0.0, 0.0, 0.0};
    const core::Vector3D actual = core::projected_vector(a, tiny);
    return expect_vector_close(actual, expected, "projected_vector degenerate direction");
}

bool test_unit_vector_regular_case() {
    const core::Vector3D v{3.0, 4.0, 0.0};
    const core::Vector3D expected{0.6, 0.8, 0.0};
    const core::Vector3D actual = core::unit_vector(v);
    return expect_vector_close(actual, expected, "unit_vector regular case");
}

bool test_unit_vector_degenerate_case() {
    const core::Vector3D tiny{1e-13, -1e-13, 1e-13}; // norm < 1e-12 threshold
    const core::Vector3D expected{0.0, 0.0, 0.0};
    const core::Vector3D actual = core::unit_vector(tiny);
    return expect_vector_close(actual, expected, "unit_vector degenerate case");
}

} // namespace

int main() {
    Kokkos::initialize();
    bool ok = true;
    {
        ok = test_vector_add_sub_mul() && ok;
        ok = test_dot_product() && ok;
        ok = test_cross_product() && ok;
        ok = test_projected_vector_regular_case() && ok;
        ok = test_projected_vector_degenerate_direction() && ok;
        ok = test_unit_vector_regular_case() && ok;
        ok = test_unit_vector_degenerate_case() && ok;
    }
    Kokkos::finalize();

    if (!ok) {
        std::cerr << "Vector3D tests failed.\n";
        return 1;
    }
    std::cout << "Vector3D tests passed.\n";
    return 0;
}