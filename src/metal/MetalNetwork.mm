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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "MetalNetwork.h"

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "MetalContextImpl.h"
#include "Network.h"

namespace {

constexpr auto PLANES = Network::INPUT_CHANNELS;
constexpr auto POL_PLANES = Network::OUTPUTS_POLICY;
constexpr auto VAL_PLANES = Network::OUTPUTS_VALUE;

NSArray<NSNumber*>* shape4(const int a, const int b, const int c, const int d) {
    return @[ @(a), @(b), @(c), @(d) ];
}

NSData* float_data(const std::vector<float>& v) {
    return [NSData dataWithBytes:v.data() length:v.size() * sizeof(float)];
}

} // namespace

// One compiled graph and its shared-memory buffers for a fixed batch size.
struct BatchGraph {
    MPSGraphExecutable* exe = nil;
    id<MTLBuffer> in_buf = nil;
    id<MTLBuffer> pol_buf = nil;
    id<MTLBuffer> val_buf = nil;
    MPSGraphTensorData* in_data = nil;
    MPSGraphTensorData* pol_data = nil;
    MPSGraphTensorData* val_data = nil;
    // Index of the policy / value tensor in the executable's result order.
    int pol_index = 0;
    int val_index = 1;
};

struct MetalNetwork::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    int channels = 0;
    int blocks = 0;
    // Raw weights are kept so that graphs for other batch sizes can be built
    // lazily; BN folding is redone per graph (cheap next to compilation).
    ForwardPipe::ForwardPipeWeights weights;
    std::map<int, BatchGraph> graphs;

    // Constant tensors and convolution descriptors are graph-bound, so they
    // are created while building each graph.
    BatchGraph build(int batch);
};

namespace {

// y = scale * (conv(x) - mean) folded into conv_{w * scale}(x) + b'.
struct FoldedLayer {
    std::vector<float> w; // OIHW
    std::vector<float> b; // K
};

FoldedLayer fold_bn(const std::vector<float>& w, const std::vector<float>& mean,
                    const std::vector<float>& scale, const int K, const int C) {
    FoldedLayer f;
    f.w.resize(w.size());
    f.b.resize(K);
    const auto per_out = static_cast<std::size_t>(C) * 9;
    if (w.size() != static_cast<std::size_t>(K) * per_out
        || mean.size() != static_cast<std::size_t>(K)
        || scale.size() != static_cast<std::size_t>(K)) {
        throw std::runtime_error("Metal: unexpected convolution weight shape");
    }
    for (auto k = 0; k < K; k++) {
        for (auto i = std::size_t{0}; i < per_out; i++) {
            f.w[k * per_out + i] = w[k * per_out + i] * scale[k];
        }
        f.b[k] = -mean[k] * scale[k];
    }
    return f;
}

MPSGraphConvolution2DOpDescriptor* conv_descriptor(const int pad) {
    return [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:1
                      strideInY:1
                dilationRateInX:1
                dilationRateInY:1
                         groups:1
                    paddingLeft:pad
                   paddingRight:pad
                     paddingTop:pad
                  paddingBottom:pad
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
}

} // namespace

BatchGraph MetalNetwork::Impl::build(const int batch) {
    constexpr auto B = BOARD_SIZE;
    const auto C = channels;
    MPSGraph* g = [[MPSGraph alloc] init];

    MPSGraphTensor* input = [g placeholderWithShape:shape4(batch, PLANES, B, B)
                                           dataType:MPSDataTypeFloat32
                                               name:@"input"];
    const auto conv3 = conv_descriptor(1);

    // conv + folded BN (+ residual) + ReLU
    auto layer = [&](MPSGraphTensor* x, const int index, const int in_ch,
                     MPSGraphTensor* skip) {
        const auto f = fold_bn(weights.m_conv_weights[index],
                               weights.m_batchnorm_means[index],
                               weights.m_batchnorm_stddevs[index], C, in_ch);
        MPSGraphTensor* w =
            [g constantWithData:float_data(f.w)
                          shape:shape4(C, in_ch, 3, 3)
                       dataType:MPSDataTypeFloat32];
        MPSGraphTensor* bias =
            [g constantWithData:float_data(f.b)
                          shape:shape4(1, C, 1, 1)
                       dataType:MPSDataTypeFloat32];
        MPSGraphTensor* t = [g convolution2DWithSourceTensor:x
                                               weightsTensor:w
                                                  descriptor:conv3
                                                        name:nil];
        t = [g additionWithPrimaryTensor:t secondaryTensor:bias name:nil];
        if (skip != nil) {
            t = [g additionWithPrimaryTensor:t secondaryTensor:skip name:nil];
        }
        return [g reLUWithTensor:t name:nil];
    };

    MPSGraphTensor* x = layer(input, 0, PLANES, nil);
    for (auto b = 0; b < blocks; b++) {
        MPSGraphTensor* skip = x;
        MPSGraphTensor* y = layer(x, 1 + 2 * b, C, nil);
        x = layer(y, 2 + 2 * b, C, skip);
    }

    // Heads: raw 1x1 convolutions. BN, FC, softmax and tanh stay on the CPU.
    const auto conv1 = conv_descriptor(0);
    auto head = [&](const std::vector<float>& w, const int outputs) {
        if (w.size() != static_cast<std::size_t>(outputs) * C) {
            throw std::runtime_error("Metal: unexpected head weight shape");
        }
        MPSGraphTensor* wt = [g constantWithData:float_data(w)
                                           shape:shape4(outputs, C, 1, 1)
                                        dataType:MPSDataTypeFloat32];
        return [g convolution2DWithSourceTensor:x
                                  weightsTensor:wt
                                     descriptor:conv1
                                           name:nil];
    };
    MPSGraphTensor* pol = head(weights.m_conv_pol_w, POL_PLANES);
    MPSGraphTensor* val = head(weights.m_conv_val_w, VAL_PLANES);

    MPSGraphShapedType* in_type =
        [[MPSGraphShapedType alloc] initWithShape:shape4(batch, PLANES, B, B)
                                         dataType:MPSDataTypeFloat32];
    MPSGraphDevice* gdev = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraphExecutable* exe = [g compileWithDevice:gdev
                                             feeds:@{input : in_type}
                                     targetTensors:@[ pol, val ]
                                  targetOperations:nil
                             compilationDescriptor:nil];
    if (exe == nil) {
        throw std::runtime_error("Metal: MPSGraph compilation failed");
    }

    BatchGraph bg;
    bg.exe = exe;
    const auto bytes = [](const int n) { return n * sizeof(float); };
    const auto plane = B * B;
    // Shared storage: the CPU and the GPU use the same memory, no copies.
    bg.in_buf = [device newBufferWithLength:bytes(batch * PLANES * plane)
                                    options:MTLResourceStorageModeShared];
    bg.pol_buf = [device newBufferWithLength:bytes(batch * POL_PLANES * plane)
                                     options:MTLResourceStorageModeShared];
    bg.val_buf = [device newBufferWithLength:bytes(batch * VAL_PLANES * plane)
                                     options:MTLResourceStorageModeShared];
    if (bg.in_buf == nil || bg.pol_buf == nil || bg.val_buf == nil) {
        throw std::runtime_error("Metal: could not allocate shared buffers");
    }
    bg.in_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bg.in_buf
                    shape:shape4(batch, PLANES, B, B)
                 dataType:MPSDataTypeFloat32];
    bg.pol_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bg.pol_buf
                    shape:shape4(batch, POL_PLANES, B, B)
                 dataType:MPSDataTypeFloat32];
    bg.val_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bg.val_buf
                    shape:shape4(batch, VAL_PLANES, B, B)
                 dataType:MPSDataTypeFloat32];

    // The executable may order its results differently from targetTensors.
    NSArray<MPSGraphTensor*>* order = exe.targetTensors;
    if (order != nil && order.count == 2) {
        bg.pol_index = [order[0] isEqual:pol] ? 0 : 1;
        bg.val_index = 1 - bg.pol_index;
    }
    return bg;
}

MetalNetwork::MetalNetwork(const MetalContext& ctx, const int channels,
                           const int residual_blocks,
                           const ForwardPipe::ForwardPipeWeights& weights)
    : m_impl(std::make_unique<Impl>()) {
    @autoreleasepool {
        m_impl->device = ctx.impl().device;
        m_impl->queue = ctx.impl().queue;
        m_impl->channels = channels;
        m_impl->blocks = residual_blocks;
        m_impl->weights = weights;
        if (weights.m_conv_weights.size()
            != static_cast<std::size_t>(1 + 2 * residual_blocks)) {
            throw std::runtime_error("Metal: layer count does not match blocks");
        }
        // Build the batch-1 graph now so weight errors surface at load time.
        m_impl->graphs.emplace(1, m_impl->build(1));
    }
}

MetalNetwork::~MetalNetwork() = default;

void MetalNetwork::forward(const float* const in, const int batch,
                           float* const pol, float* const val) {
    @autoreleasepool {
        auto it = m_impl->graphs.find(batch);
        if (it == m_impl->graphs.end()) {
            it = m_impl->graphs.emplace(batch, m_impl->build(batch)).first;
        }
        auto& g = it->second;
        constexpr auto plane = BOARD_SIZE * BOARD_SIZE;

        std::memcpy([g.in_buf contents], in,
                    sizeof(float) * batch * PLANES * plane);

        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        NSMutableArray<MPSGraphTensorData*>* results =
            [NSMutableArray arrayWithObjects:g.pol_data, g.val_data, nil];
        results[g.pol_index] = g.pol_data;
        results[g.val_index] = g.val_data;
        [g.exe runWithMTLCommandQueue:m_impl->queue
                          inputsArray:@[ g.in_data ]
                         resultsArray:results
                  executionDescriptor:desc];

        std::memcpy(pol, [g.pol_buf contents],
                    sizeof(float) * batch * POL_PLANES * plane);
        std::memcpy(val, [g.val_buf contents],
                    sizeof(float) * batch * VAL_PLANES * plane);
    }
}
