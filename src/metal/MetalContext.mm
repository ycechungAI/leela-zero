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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalContext.h"

#include <cmath>
#include <vector>

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
};

namespace {

std::string to_std(NSString* const s) {
    return s == nil ? std::string() : std::string([s UTF8String]);
}

std::string error_text(NSError* const e) {
    return e == nil ? "unknown error" : to_std([e localizedDescription]);
}

// y[i] = a * x[i] + y[i]
const char* const SELF_TEST_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;
kernel void axpy(device const float* x [[buffer(0)]],
                 device float* y       [[buffer(1)]],
                 constant float& a     [[buffer(2)]],
                 uint i                [[thread_position_in_grid]]) {
    y[i] = a * x[i] + y[i];
}
)";

} // namespace

MetalContext::MetalContext() : m_impl(std::make_unique<Impl>()) {}
MetalContext::~MetalContext() = default;

std::unique_ptr<MetalContext> MetalContext::create(std::string& error) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            error = "no Metal device available";
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            error = "could not create a Metal command queue";
            return nullptr;
        }
        std::unique_ptr<MetalContext> ctx(new MetalContext());
        ctx->m_impl->device = device;
        ctx->m_impl->queue = queue;
        return ctx;
    }
}

std::string MetalContext::device_name() const {
    return to_std([m_impl->device name]);
}

bool MetalContext::has_unified_memory() const {
    return [m_impl->device hasUnifiedMemory];
}

bool MetalContext::supports_apple_gpu_family() const {
    return [m_impl->device supportsFamily:MTLGPUFamilyApple7];
}

std::size_t MetalContext::max_threads_per_threadgroup() const {
    const auto size = [m_impl->device maxThreadsPerThreadgroup];
    return static_cast<std::size_t>(size.width);
}

std::size_t MetalContext::recommended_working_set_bytes() const {
    return static_cast<std::size_t>(
        [m_impl->device recommendedMaxWorkingSetSize]);
}

std::string MetalContext::describe() const {
    return "Metal: " + device_name()
           + (has_unified_memory() ? ", unified memory" : ", discrete memory");
}

bool MetalContext::compile_check(const std::string& msl_source,
                                 std::string& error) const {
    @autoreleasepool {
        NSError* err = nil;
        NSString* source = [NSString stringWithUTF8String:msl_source.c_str()];
        id<MTLLibrary> lib = [m_impl->device newLibraryWithSource:source
                                                          options:nil
                                                            error:&err];
        if (lib == nil) {
            error = error_text(err);
            return false;
        }
        return true;
    }
}

bool MetalContext::self_test(std::string& error) const {
    @autoreleasepool {
        auto* const dev = m_impl->device;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev
            newLibraryWithSource:[NSString stringWithUTF8String:SELF_TEST_SOURCE]
                         options:nil
                           error:&err];
        if (lib == nil) {
            error = "shader compile failed: " + error_text(err);
            return false;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"axpy"];
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) {
            error = "pipeline creation failed: " + error_text(err);
            return false;
        }

        constexpr auto N = std::size_t{1024};
        constexpr auto a = 3.0f;
        const auto bytes = N * sizeof(float);
        // Shared storage: the CPU writes the inputs and reads the results
        // through the same memory the GPU uses.
        id<MTLBuffer> x = [dev newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> y = [dev newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
        if (x == nil || y == nil) {
            error = "could not allocate shared buffers";
            return false;
        }
        auto* const xp = static_cast<float*>([x contents]);
        auto* const yp = static_cast<float*>([y contents]);
        for (auto i = std::size_t{0}; i < N; i++) {
            xp[i] = static_cast<float>(i);
            yp[i] = 1.0f;
        }

        id<MTLCommandBuffer> cb = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:y offset:0 atIndex:1];
        [enc setBytes:&a length:sizeof(a) atIndex:2];
        const auto tg = std::min<NSUInteger>(
            [pso maxTotalThreadsPerThreadgroup], 256);
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            error = "command buffer failed: " + error_text([cb error]);
            return false;
        }

        for (auto i = std::size_t{0}; i < N; i++) {
            const auto expected = a * static_cast<float>(i) + 1.0f;
            if (std::fabs(yp[i] - expected) > 1e-5f) {
                error = "wrong result at index " + std::to_string(i);
                return false;
            }
        }
        return true;
    }
}
