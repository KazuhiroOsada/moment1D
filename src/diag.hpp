#ifndef DIAG_HPP
#define DIAG_HPP

#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

#include <Kokkos_Core.hpp>

#include "core.hpp"


namespace diag {

void write_parameters(const std::string& problem_name, const double dt, const int max_iters, const int diag_interval);

void diag_moment(const core::Species& species, const std::string& name, const int it);

void diag_field(const Kokkos::View<double**>& U_em, const int it);

} // namespace diag

#endif // DIAG_HPP
