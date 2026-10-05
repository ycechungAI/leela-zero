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

#include "MetalWinograd.h"

#include <stdexcept>
#include <string>

#include "Network.h"
#include "WinogradKernels.h"

namespace {

constexpr int PLANES = Network::INPUT_CHANNELS;
constexpr int PLANE = BOARD_SIZE * BOARD_SIZE;
constexpr int TILES = WINOGRAD_P;       // tiles per board
constexpr int ELEMENTS = WINOGRAD_TILE; // 36 Winograd elements

struct Params {
    int C;
    int K;
    int N;
};

id<MTLBuffer> shared_buffer(id<MTLDevice> device, const NSUInteger bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error("Metal: could not allocate a buffer");
    }
    return buffer;
}

id<MTLBuffer> buffer_with(id<MTLDevice> device, const std::vector<float>& data) {
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:data.data()
                            length:data.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error("Metal: could not allocate a buffer");
    }
    return buffer;
}

// The same values rounded to fp16.
id<MTLBuffer> half_buffer_with(id<MTLDevice> device,
                               const std::vector<float>& data) {
    std::vector<_Float16> half(data.size());
    for (auto i = std::size_t{0}; i < data.size(); i++) {
        half[i] = static_cast<_Float16>(data[i]);
    }
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:half.data()
                            length:half.size() * sizeof(_Float16)
                           options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error("Metal: could not allocate a buffer");
    }
    return buffer;
}

// `matrices` row-major matrices of rows x columns, back to back.
MPSMatrix* matrix_view(id<MTLBuffer> buffer, const int rows, const int columns,
                       const int matrices, const NSUInteger element_bytes,
                       const MPSDataType type) {
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows
                         columns:columns
                        matrices:matrices
                        rowBytes:columns * element_bytes
                     matrixBytes:static_cast<NSUInteger>(rows) * columns
                                 * element_bytes
                        dataType:type];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
}

MPSMatrixMultiplication* make_gemm(id<MTLDevice> device, const int n,
                                   const int c, const int k) {
    MPSMatrixMultiplication* gemm =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:NO
                                         transposeRight:NO
                                             resultRows:n
                                          resultColumns:k
                                        interiorColumns:c
                                                  alpha:1.0
                                                   beta:0.0];
    gemm.batchStart = 0;
    gemm.batchSize = ELEMENTS;
    return gemm;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
                                     NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) {
        throw std::runtime_error("Metal: missing kernel "
                                 + std::string([name UTF8String]));
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) {
        throw std::runtime_error(
            "Metal: kernel pipeline failed: "
            + std::string([[err localizedDescription] UTF8String]));
    }
    return pso;
}

// A grid of (x, y) threads, in groups that fill a SIMD width.
void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
              const NSUInteger x, const NSUInteger y) {
    const auto width = pso.threadExecutionWidth;
    const auto rows = std::max<NSUInteger>(
        1, std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup / width, 8));
    [enc dispatchThreads:MTLSizeMake(x, y, 1)
        threadsPerThreadgroup:MTLSizeMake(width, rows, 1)];
}

} // namespace

WinogradNet::WinogradNet(id<MTLDevice> device, id<MTLCommandQueue> queue,
                         const int channels, const int blocks,
                         const ForwardPipe::ForwardPipeWeights& weights,
                         const MetalPrecision precision)
    : m_device(device),
      m_queue(queue),
      m_channels(channels),
      m_blocks(blocks),
      m_store_bytes(precision == MetalPrecision::Half ? 2 : 4),
      m_store_type(precision == MetalPrecision::Half ? MPSDataTypeFloat16
                                                     : MPSDataTypeFloat32),
      m_m_bytes(4),
      m_m_type(MPSDataTypeFloat32) {
    const auto layers = 1 + 2 * blocks;
    if (weights.m_conv_weights.size() != static_cast<std::size_t>(layers)
        || weights.m_batchnorm_means.size() != static_cast<std::size_t>(layers)
        || weights.m_batchnorm_stddevs.size()
               != static_cast<std::size_t>(layers)) {
        throw std::runtime_error("Metal: layer count does not match blocks");
    }

    @autoreleasepool {
        NSString* source = [NSString
            stringWithFormat:@"#define store_t %s\n#define mstore_t float\n%s",
                             precision == MetalPrecision::Single ? "float"
                                                                 : "half",
                             WINOGRAD_MSL];
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&err];
        if (lib == nil) {
            throw std::runtime_error(
                "Metal: Winograd kernels failed to compile: "
                + std::string([[err localizedDescription] UTF8String]));
        }
        m_in_float = pipeline(device, lib, @"in_transform_float");
        m_in_act = pipeline(device, lib, @"in_transform_act");
        m_out = pipeline(device, lib, @"out_transform");
        m_heads = pipeline(device, lib, @"heads");

        for (auto i = 0; i < layers; i++) {
            const auto in_ch = i == 0 ? PLANES : channels;
            const auto u = Network::winograd_transform_f(
                weights.m_conv_weights[i], channels, in_ch);
            if (u.size() != static_cast<std::size_t>(ELEMENTS) * in_ch * channels
                || weights.m_batchnorm_means[i].size()
                       != static_cast<std::size_t>(channels)
                || weights.m_batchnorm_stddevs[i].size()
                       != static_cast<std::size_t>(channels)) {
                throw std::runtime_error(
                    "Metal: unexpected convolution weight shape");
            }
            // Weights, V and the activations are stored in the chosen
            // precision; M (the GEMM result) is always float, which costs
            // about 7% speed and halves the fp16 value error on a real net.
            // The transforms compute in float, and the heads and the batch
            // norm constants stay fp32.
            m_u.push_back(precision == MetalPrecision::Half
                              ? half_buffer_with(device, u)
                              : buffer_with(device, u));
            m_u_mat.push_back(matrix_view(m_u.back(), in_ch, channels,
                                          ELEMENTS, m_store_bytes,
                                          m_store_type));
            m_means.push_back(buffer_with(device, weights.m_batchnorm_means[i]));
            m_scales.push_back(
                buffer_with(device, weights.m_batchnorm_stddevs[i]));
        }

        if (weights.m_conv_pol_w.size()
                != static_cast<std::size_t>(Network::OUTPUTS_POLICY) * channels
            || weights.m_conv_val_w.size()
                   != static_cast<std::size_t>(Network::OUTPUTS_VALUE)
                          * channels) {
            throw std::runtime_error("Metal: unexpected head weight shape");
        }
        std::vector<float> head(weights.m_conv_pol_w);
        head.insert(head.end(), weights.m_conv_val_w.begin(),
                    weights.m_conv_val_w.end());
        m_head_w = buffer_with(device, head);
    }
}

WinogradScratch WinogradNet::make_scratch(const int batch) const {
    @autoreleasepool {
        WinogradScratch s;
        const auto n = batch * TILES;
        const auto k = m_channels;
        s.batch = batch;
        s.v = shared_buffer(m_device, static_cast<NSUInteger>(ELEMENTS) * n
                                          * std::max(PLANES, k) * m_store_bytes);
        s.m = shared_buffer(m_device, static_cast<NSUInteger>(ELEMENTS) * n * k
                                          * m_m_bytes);
        const auto act_bytes =
            static_cast<NSUInteger>(batch) * k * PLANE * m_store_bytes;
        s.act_a = shared_buffer(m_device, act_bytes);
        s.act_b = shared_buffer(m_device, act_bytes);
        s.v_in = matrix_view(s.v, n, PLANES, ELEMENTS, m_store_bytes,
                             m_store_type);
        s.v_res = matrix_view(s.v, n, k, ELEMENTS, m_store_bytes, m_store_type);
        s.m_mat = matrix_view(s.m, n, k, ELEMENTS, m_m_bytes, m_m_type);
        s.gemm_in = make_gemm(m_device, n, PLANES, k);
        s.gemm_res = make_gemm(m_device, n, k, k);
        return s;
    }
}

void WinogradNet::run(const WinogradScratch& s, id<MTLBuffer> in,
                      id<MTLBuffer> pol, id<MTLBuffer> val) const {
    @autoreleasepool {
        const auto n = s.batch * TILES;
        const auto k = m_channels;
        id<MTLCommandBuffer> cb = [m_queue commandBuffer];

        const auto in_transform = [&](id<MTLComputePipelineState> pso,
                                      id<MTLBuffer> src, const int channels) {
            const Params p{channels, k, n};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:s.v offset:0 atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            dispatch(enc, pso, channels, n);
            [enc endEncoding];
        };
        const auto out_transform = [&](const int layer, id<MTLBuffer> dst,
                                       const int add_residual) {
            const Params p{0, k, n};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:m_out];
            [enc setBuffer:s.m offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBuffer:m_means[layer] offset:0 atIndex:2];
            [enc setBuffer:m_scales[layer] offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            [enc setBytes:&add_residual length:sizeof(add_residual) atIndex:5];
            dispatch(enc, m_out, k, n);
            [enc endEncoding];
        };

        // Input layer: network input (float) -> act_a.
        in_transform(m_in_float, in, PLANES);
        [s.gemm_in encodeToCommandBuffer:cb
                              leftMatrix:s.v_in
                             rightMatrix:m_u_mat[0]
                            resultMatrix:s.m_mat];
        out_transform(0, s.act_a, 0);

        // Residual blocks: act_a -> act_b -> act_a (+ act_a, in place).
        auto layer = 1;
        for (auto b = 0; b < m_blocks; b++) {
            in_transform(m_in_act, s.act_a, k);
            [s.gemm_res encodeToCommandBuffer:cb
                                   leftMatrix:s.v_res
                                  rightMatrix:m_u_mat[layer]
                                 resultMatrix:s.m_mat];
            out_transform(layer, s.act_b, 0);
            layer++;

            in_transform(m_in_act, s.act_b, k);
            [s.gemm_res encodeToCommandBuffer:cb
                                   leftMatrix:s.v_res
                                  rightMatrix:m_u_mat[layer]
                                 resultMatrix:s.m_mat];
            out_transform(layer, s.act_a, 1);
            layer++;
        }

        {
            const Params p{k, k, s.batch};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:m_heads];
            [enc setBuffer:s.act_a offset:0 atIndex:0];
            [enc setBuffer:m_head_w offset:0 atIndex:1];
            [enc setBuffer:pol offset:0 atIndex:2];
            [enc setBuffer:val offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            dispatch(enc, m_heads, PLANE, s.batch);
            [enc endEncoding];
        }

        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            dispatch_semaphore_signal(done);
        }];
        [cb commit];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (cb.status != MTLCommandBufferStatusCompleted) {
            NSString* why = cb.error ? [cb.error localizedDescription]
                                     : @"unknown error";
            throw std::runtime_error("Metal: GPU error: "
                                     + std::string([why UTF8String]));
        }
    }
}
