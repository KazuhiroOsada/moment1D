#include "diag.hpp"

#include <filesystem>


namespace diag {

namespace {
const std::string base_output_dir = "../results/";
std::string output_dir = base_output_dir;
} // namespace

void write_parameters(const std::string& problem_name, const double dt, const int max_iters, const int diag_interval) {
    output_dir = base_output_dir + problem_name + "/";
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "parameter.txt";
    std::ofstream ofs(path);

    ofs << "problem_name " << problem_name << "\n";
    ofs << "Nx " << core::Nx << "\n";
    ofs << "N_ghost " << core::N_ghost << "\n";
    ofs << "Nx_total " << core::Nx_total << "\n";
    ofs << "lb " << core::lb << "\n";
    ofs << "ub " << core::ub << "\n";
    ofs << "N_MOMENT " << core::N_MOMENT << "\n";
    ofs << "N_FIELD " << core::N_FIELD << "\n";
    ofs << "gamma " << core::gamma << "\n";
    ofs << "c " << core::c << "\n";
    ofs << "dt " << dt << "\n";
    ofs << "max_iters " << max_iters << "\n";
    ofs << "diag_interval " << diag_interval << "\n";
}

void diag_moment(const core::Species& species, const std::string& name, const int it) {
    std::ostringstream oss;
    oss << output_dir << name << "_" << std::setw(5) << std::setfill('0') << it << ".dat";
    std::ofstream ofs(oss.str());

    auto U_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), species.U);
    const int Nx = core::Nx;

    std::vector<double> rho(Nx), ux(Nx), uy(Nx), uz(Nx), p(Nx);
    
    for (int ix = core::lb; ix < core::ub; ++ix) {
        const int i = ix - core::lb;
        core::MomentArray U_cell = {
            U_host(core::RHO, ix),
            U_host(core::MX, ix),
            U_host(core::MY, ix),
            U_host(core::MZ, ix),
            U_host(core::ENE, ix)
        };
        core::MomentArray Uprim_cell = core::get_moment_primitives(U_cell);
        rho[i] = Uprim_cell[core::RHO];
        ux[i] = Uprim_cell[core::UX];
        uy[i] = Uprim_cell[core::UY];
        uz[i] = Uprim_cell[core::UZ];
        p[i] = Uprim_cell[core::PRS];
    }

    auto write_vec = [&](const std::vector<double>& v) {
        ofs.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
    };

    write_vec(rho);
    write_vec(ux);
    write_vec(uy);
    write_vec(uz);
    write_vec(p);

    ofs.close();
}

void diag_field(const Kokkos::View<double**>& U_em, const int it) {
    std::ostringstream oss;
    oss << output_dir << "fields_" << std::setw(5) << std::setfill('0') << it << ".dat";
    std::ofstream ofs(oss.str());

    auto U_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U_em);
    const int Nx = core::Nx;

    std::vector<double> Ex(Nx), Ey(Nx), Ez(Nx), Bx(Nx), By(Nx), Bz(Nx);

    for (int ix = core::lb; ix < core::ub; ++ix) {
        const int i = ix - core::lb;
        Ex[i] = U_host(core::EX, ix);
        Ey[i] = U_host(core::EY, ix);
        Ez[i] = U_host(core::EZ, ix);
        Bx[i] = U_host(core::BX, ix);
        By[i] = U_host(core::BY, ix);
        Bz[i] = U_host(core::BZ, ix);
    }

    auto write_vec = [&](const std::vector<double>& v) {
        ofs.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
    };

    write_vec(Ex);
    write_vec(Ey);
    write_vec(Ez);
    write_vec(Bx);
    write_vec(By);
    write_vec(Bz);

    ofs.close();
}

} // namespace diag
