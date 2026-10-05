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

#ifndef METALWINOGRAD_H_INCLUDED
#define METALWINOGRAD_H_INCLUDED

// Objective-C++ only; never include this from a .cpp or a plain header.
//
// The residual tower as Winograd F(4x4, 3x3) in Metal: an input-transform
// kernel, one batched matrix multiply per layer (Apple's
// MPSMatrixMultiplication over the 36 Winograd elements), and an
// output-transform kernel that also applies batch norm, the residual add and
// ReLU. Then the 1x1 heads. Kernel source: WinogradKernels.h; step 2.11 of
// docs/apple-silicon/12-plan-2.11-metal-winograd.md.
//
// Everything immutable (pipelines, weights) lives in WinogradNet and is shared
// by all workers. Everything a run writes lives in a WinogradScratch, one per
// slot, so run() needs no lock: each call builds its own command buffer.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <vector>

#include "ForwardPipe.h"
#include "MetalNetwork.h"

struct WinogradScratch {
    int batch = 0;
    id<MTLBuffer> v = nil;     // 36 x N x C (C = 18 for the first layer)
    id<MTLBuffer> m = nil;     // 36 x N x K
    id<MTLBuffer> act_a = nil; // batch x K x 361, the running activation
    id<MTLBuffer> act_b = nil; // batch x K x 361, the block's middle
    MPSMatrix* v_in = nil;     // v as 36 matrices of N x 18
    MPSMatrix* v_res = nil;    // v as 36 matrices of N x K
    MPSMatrix* m_mat = nil;    // m as 36 matrices of N x K
    MPSMatrixMultiplication* gemm_in = nil;  // N x 18 times 18 x K
    MPSMatrixMultiplication* gemm_res = nil; // N x K times K x K
};

class WinogradNet {
public:
    // Throws std::runtime_error with a readable message on failure.
    WinogradNet(id<MTLDevice> device, id<MTLCommandQueue> queue, int channels,
                int blocks, const ForwardPipe::ForwardPipeWeights& weights,
                MetalPrecision precision);

    WinogradScratch make_scratch(int batch) const;

    // Evaluates in (batch x 18 x 361 floats) into pol and val, and returns
    // when the GPU is done. Thread-safe for different scratches.
    // Throws std::runtime_error if the GPU reports an error.
    void run(const WinogradScratch& scratch, id<MTLBuffer> in,
             id<MTLBuffer> pol, id<MTLBuffer> val) const;

private:
    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    int m_channels;
    int m_blocks;
    NSUInteger m_store_bytes; // 4 (float) or 2 (half)
    MPSDataType m_store_type;
    NSUInteger m_m_bytes;     // M is float even for half storage
    MPSDataType m_m_type;
    id<MTLComputePipelineState> m_in_float = nil;
    id<MTLComputePipelineState> m_in_act = nil;
    id<MTLComputePipelineState> m_out = nil;
    id<MTLComputePipelineState> m_heads = nil;
    // Per tower layer: weights as 36 matrices of C x K, and batch norm.
    std::vector<id<MTLBuffer>> m_u;
    std::vector<MPSMatrix*> m_u_mat;
    std::vector<id<MTLBuffer>> m_means;
    std::vector<id<MTLBuffer>> m_scales;
    id<MTLBuffer> m_head_w = nil; // [3][K]: policy rows, then value
};

#endif
