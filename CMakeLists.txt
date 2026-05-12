cmake_minimum_required(VERSION 3.18)
project(1Dmoment CXX)

set(CMAKE_CXX_STANDARD 17)

# ---- OpenMP ----
execute_process(
  COMMAND brew --prefix libomp
  OUTPUT_VARIABLE OMP_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "${OMP_PREFIX}/lib/libomp.dylib")
find_package(OpenMP REQUIRED)

# ---- Kokkos ----
set(Kokkos_ROOT "$ENV{HOME}/kokkos/install")
find_package(Kokkos REQUIRED)

# ---- Output Directories ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

# ---- Sources ----
set(PROJECT_HEADERS
    src/core.hpp
    src/diag.hpp
    src/field_flux_solver.hpp
    src/moment_flux_solver.hpp
    src/push_hyperbolic.hpp
)

set(PROJECT_SOURCES
    src/diag.cpp
    src/push_hyperbolic.cpp
)

# ---- Executables ----
set(APPS
    em_wave
    shock_tube
)

# ---- Tests ----
set (TESTS
    test_moment_flux_solver
)

foreach(app ${APPS})
  add_executable(${app}
    "src/apps/${app}.cpp"
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
  )
  target_include_directories(${app} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${app} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
endforeach()

foreach(test ${TESTS})
  add_executable(${test}
    "tests/${test}.cpp"
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
  )
  target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${test} PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX)
endforeach()
