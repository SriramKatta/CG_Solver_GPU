#pragma once

#include <cstring>
#include <string_view>
#include <tuple>
#include <fmt/format.h>

#include "util.h"


template <typename tpe>
void initConjugateGradient(tpe *__restrict__ u, tpe *__restrict__ rhs, const size_t nx, const size_t ny) {
    constexpr auto app = 1;

    switch (app) {
        case 0:
        // version 0 - zero boundary, unit inner values
        for (size_t i1 = 0; i1 < ny; ++i1) {
            for (size_t i0 = 0; i0 < nx; ++i0) {
                if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
                    u[i1 * nx + i0] = 0;
                else
                    u[i1 * nx + i0] = 1;

                rhs[i1 * nx + i0] = 0;
            }
        }

        break;

        case 1:
        // version 1 - simple trigonometric test problem
        for (size_t i1 = 0; i1 < ny; ++i1) {
            for (size_t i0 = 0; i0 < nx; ++i0) {
                auto x = i0 - 1 + 0.5;
                auto y = i1 - 1 + 0.5;

                if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
                    u[i1 * nx + i0] = sin(1e-3 * x) * cos(2e-3 * y);
                else
                    u[i1 * nx + i0] = 1;
                rhs[i1 * nx + i0] = 5e-6 * sin(1e-3 * x) * cos(2e-3 * y);
            }
        }
        break;
    }
}

template <typename tpe>
void checkSolutionConjugateGradient(const tpe *const __restrict__ u, const tpe *const __restrict__ rhs, const size_t nx, const size_t ny) {
    double res = 0;
    for (size_t i1 = 1; i1 < ny - 1; ++i1) {
        for (size_t i0 = 1; i0 < nx - 1; ++i0) {
            const double localRes = rhs[i1 * nx + i0] - (
                    4 * u[i1 * nx + i0] - (u[i1 * nx + i0 - 1] + u[i1 * nx + i0 + 1] + u[(i1 - 1) * nx + i0] + u[(i1 + 1) * nx + i0]));

            res += localRes * localRes;
        }
    }

    res = sqrt(res);

    fmt::print("  Final residual is {}", res);
}

inline std::tuple<std::string_view, size_t, size_t, size_t, size_t> parseCLA_2d(int argc, char **argv) {
    // default values
    std::string_view tpeName;
    size_t nx = 4096;
    size_t ny = 4096;

    size_t nItWarmUp = 0;
    size_t nIt = 20000;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        tpeName = argv[i];
    ++i;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        ny = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;

    
    return {tpeName, nx, ny, nItWarmUp, nIt};
}