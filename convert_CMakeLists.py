"""This script converts the CMakeLists.txt file written for the local development environment to one in imadalab-gpuserver environment.
"""
import shutil

FILE_PATH = "CMakeLists.txt"
BACKUP_PATH = "CMakeLists.txt.bak"

shutil.copy(FILE_PATH, BACKUP_PATH)

def convert_settings(lines):
    for i, line in enumerate(lines):
        if line.startswith("cmake_minimum_required"):
            lines[i] = line.replace("VERSION 3.18", "VERSION 3.16")
        if line.startswith("set(CMAKE_CXX_STANDARD 17)"):
            lines[i] = line.replace("17", "20")
        if line.startswith("set(Kokkos_ROOT"):
            lines[i] = line.replace("$ENV{HOME}/kokkos/install", "$ENV{HOME}/kokkos-install")

def delete_OPENMP_settings(lines):
    delete_line = False
    for i, line in enumerate(lines):
        if line.startswith("# ---- OpenMP ----"):
            delete_line = True
        if line.startswith("# ---- Kokkos ----"):
            delete_line = False
        if delete_line:
            lines[i] = ""
        if "OpenMP::OpenMP_CXX" in line:
            lines[i] = line.replace("OpenMP::OpenMP_CXX", "")


if __name__ == "__main__":
    with open("CMakeLists.txt", "r") as f:
        lines = f.readlines()
        convert_settings(lines)
        delete_OPENMP_settings(lines)
    with open("CMakeLists.txt", "w") as f:
        f.writelines(lines)