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

#ifndef METALCONTEXT_H_INCLUDED
#define METALCONTEXT_H_INCLUDED

#include <cstddef>
#include <memory>
#include <string>

// The Metal device, command queue and runtime shader compiler. This header is
// plain C++: all Objective-C lives in MetalContext.mm behind the pimpl, so the
// rest of the engine never sees Metal types.
class MetalContext {
public:
    // Returns null and fills `error` when there is no usable Metal device.
    static std::unique_ptr<MetalContext> create(std::string& error);

    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    std::string device_name() const;
    // True when CPU and GPU share memory (every Apple Silicon Mac).
    bool has_unified_memory() const;
    // True for Apple7 (M1) and later GPUs.
    bool supports_apple_gpu_family() const;
    std::size_t max_threads_per_threadgroup() const;
    std::size_t recommended_working_set_bytes() const;

    // One line for the startup log, e.g. "Metal: Apple M4, unified memory".
    std::string describe() const;

    // Compile MSL source at runtime. Returns false and fills `error` on
    // failure. Used to check that the toolchain works before building the
    // network kernels on top of it.
    bool compile_check(const std::string& msl_source, std::string& error) const;

    // End-to-end sanity check: compiles a small kernel, runs it on shared
    // buffers (no copies) and verifies the result on the CPU.
    bool self_test(std::string& error) const;

    // Objective-C state (device, queue); defined in MetalContextImpl.h, which
    // only .mm files may include.
    struct Impl;
    const Impl& impl() const {
        return *m_impl;
    }

private:
    MetalContext();
    std::unique_ptr<Impl> m_impl;
};

#endif
