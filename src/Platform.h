/*
    This file is part of Leela Zero.
    Copyright (C) 2026 Leela Zero contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PLATFORM_H_INCLUDED
#define PLATFORM_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <string>
#include <thread>

#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>
#ifdef USE_BLAS
#include <Accelerate/Accelerate.h>
#endif
#endif

// Small host queries with portable fallbacks. Only Apple Silicon has
// performance/efficiency cores to tell apart; elsewhere every core counts as a
// performance core.
namespace Platform {

#ifdef __APPLE__
// Returns 0 if the sysctl does not exist (Intel Macs, older macOS).
inline std::size_t sysctl_count(const char* const name) {
    int value = 0;
    auto size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 || value < 0) {
        return 0;
    }
    return static_cast<std::size_t>(value);
}
#endif

// Logical CPUs, never 0.
inline std::size_t num_cpus() {
    return std::max(std::size_t{1},
                    static_cast<std::size_t>(
                        std::thread::hardware_concurrency()));
}

// Performance cores (the fastest cluster): physical cores on Apple Silicon,
// logical CPUs elsewhere.
inline std::size_t num_perf_cores() {
#ifdef __APPLE__
    const auto n = sysctl_count("hw.perflevel0.physicalcpu");
    if (n > 0) {
        return n;
    }
#endif
    return num_cpus();
}

// Efficiency cores; 0 when the CPU is homogeneous.
inline std::size_t num_eff_cores() {
#ifdef __APPLE__
    return sysctl_count("hw.perflevel1.physicalcpu");
#else
    return 0;
#endif
}

// Ask the scheduler to run the calling thread on performance cores.
// Returns true if the request was made (Apple only).
inline bool set_thread_qos_interactive() {
#ifdef __APPLE__
    return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0;
#else
    return false;
#endif
}

// Keep each Accelerate sgemm on the calling thread: the search threads provide
// the parallelism, like openblas_set_num_threads(1). The setting is per thread
// (vecLib thread_api.h), so every thread that runs sgemm must call this.
// Returns true if Accelerate accepted it (macOS 15+, BLAS builds only).
inline bool set_blas_single_threaded() {
#if defined(__APPLE__) && defined(USE_BLAS)
    if (__builtin_available(macOS 15.0, *)) {
        return BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0;
    }
#endif
    return false;
}

// Per-thread setup for every thread that searches (the main thread and the
// pool threads).
inline void init_search_thread() {
    set_thread_qos_interactive();
    set_blas_single_threaded();
}

// "SME", "AMX" or "" when unknown, for the startup log.
inline std::string cpu_feature_string() {
#if defined(__APPLE__) && defined(__aarch64__)
    return sysctl_count("hw.optional.arm.FEAT_SME") > 0 ? "SME" : "AMX";
#else
    return "";
#endif
}

} // namespace Platform

#endif
