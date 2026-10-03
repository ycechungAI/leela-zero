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
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "MetalContextImpl.h"
#include "Network.h"

namespace {

constexpr auto PLANES = Network::INPUT_CHANNELS;
constexpr auto POL_PLANES = Network::OUTPUTS_POLICY;
constexpr auto VAL_PLANES = Network::OUTPUTS_VALUE;
constexpr auto PLANE = BOARD_SIZE * BOARD_SIZE;

NSArray<NSNumber*>* shape4(const int a, const int b, const int c, const int d) {
    return @[ @(a), @(b), @(c), @(d) ];
}

NSData* float_data(const std::vector<float>& v) {
    return [NSData dataWithBytes:v.data() length:v.size() * sizeof(float)];
}

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

struct Compiled {
    MPSGraphExecutable* exe = nil;
    // Position of the policy result in the executable's result order.
    int pol_index = 0;
};

Compiled compile(id<MTLDevice> device, const int C, const int blocks,
                 const ForwardPipe::ForwardPipeWeights& weights,
                 const int batch) {
    constexpr auto B = BOARD_SIZE;
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
        MPSGraphTensor* w = [g constantWithData:float_data(f.w)
                                          shape:shape4(C, in_ch, 3, 3)
                                       dataType:MPSDataTypeFloat32];
        MPSGraphTensor* bias = [g constantWithData:float_data(f.b)
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
    Compiled c;
    c.exe = [g compileWithDevice:gdev
                           feeds:@{input : in_type}
                   targetTensors:@[ pol, val ]
                targetOperations:nil
           compilationDescriptor:nil];
    if (c.exe == nil) {
        throw std::runtime_error("Metal: MPSGraph compilation failed");
    }
    // The executable may order its results differently from targetTensors.
    NSArray<MPSGraphTensor*>* order = c.exe.targetTensors;
    if (order != nil && order.count == 2 && ![order[0] isEqual:pol]) {
        c.pol_index = 1;
    }
    return c;
}

} // namespace

struct MetalSlot::Impl {
    int batch = 0;
    id<MTLBuffer> in_buf = nil;
    id<MTLBuffer> pol_buf = nil;
    id<MTLBuffer> val_buf = nil;
    MPSGraphTensorData* in_data = nil;
    MPSGraphTensorData* pol_data = nil;
    MPSGraphTensorData* val_data = nil;
};

MetalSlot::MetalSlot() : m_impl(std::make_unique<Impl>()) {}
MetalSlot::~MetalSlot() = default;

int MetalSlot::batch() const {
    return m_impl->batch;
}

float* MetalSlot::input() {
    return static_cast<float*>([m_impl->in_buf contents]);
}

const float* MetalSlot::policy() const {
    return static_cast<const float*>([m_impl->pol_buf contents]);
}

const float* MetalSlot::value() const {
    return static_cast<const float*>([m_impl->val_buf contents]);
}

struct MetalNetwork::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::map<int, Compiled> graphs; // by batch size; fixed after construction
    // MPSGraphExecutable encoding is serialized; execution still overlaps.
    std::mutex encode_mutex;
    // Lazily created slots for forward() (tests only).
    std::map<int, std::unique_ptr<MetalSlot>> forward_slots;
};

MetalNetwork::MetalNetwork(const MetalContext& ctx, const int channels,
                           const int residual_blocks,
                           const ForwardPipe::ForwardPipeWeights& weights,
                           const std::vector<int>& batch_sizes)
    : m_impl(std::make_unique<Impl>()) {
    @autoreleasepool {
        m_impl->device = ctx.impl().device;
        m_impl->queue = ctx.impl().queue;
        if (weights.m_conv_weights.size()
            != static_cast<std::size_t>(1 + 2 * residual_blocks)) {
            throw std::runtime_error("Metal: layer count does not match blocks");
        }
        for (const auto batch : batch_sizes) {
            if (batch < 1) {
                throw std::runtime_error("Metal: invalid batch size");
            }
            if (m_impl->graphs.count(batch) == 0) {
                m_impl->graphs.emplace(
                    batch, compile(m_impl->device, channels, residual_blocks,
                                   weights, batch));
            }
        }
    }
}

MetalNetwork::~MetalNetwork() = default;

std::unique_ptr<MetalSlot> MetalNetwork::make_slot(const int batch) const {
    if (m_impl->graphs.count(batch) == 0) {
        throw std::logic_error("Metal: no graph compiled for batch size "
                               + std::to_string(batch));
    }
    @autoreleasepool {
        std::unique_ptr<MetalSlot> slot(new MetalSlot());
        auto& s = *slot->m_impl;
        auto* const dev = m_impl->device;
        constexpr auto B = BOARD_SIZE;
        const auto bytes = [batch](const int planes) {
            return static_cast<NSUInteger>(batch) * planes * PLANE
                   * sizeof(float);
        };
        s.batch = batch;
        s.in_buf = [dev newBufferWithLength:bytes(PLANES)
                                    options:MTLResourceStorageModeShared];
        s.pol_buf = [dev newBufferWithLength:bytes(POL_PLANES)
                                     options:MTLResourceStorageModeShared];
        s.val_buf = [dev newBufferWithLength:bytes(VAL_PLANES)
                                     options:MTLResourceStorageModeShared];
        if (s.in_buf == nil || s.pol_buf == nil || s.val_buf == nil) {
            throw std::runtime_error("Metal: could not allocate shared buffers");
        }
        s.in_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:s.in_buf
                        shape:shape4(batch, PLANES, B, B)
                     dataType:MPSDataTypeFloat32];
        s.pol_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:s.pol_buf
                        shape:shape4(batch, POL_PLANES, B, B)
                     dataType:MPSDataTypeFloat32];
        s.val_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:s.val_buf
                        shape:shape4(batch, VAL_PLANES, B, B)
                     dataType:MPSDataTypeFloat32];
        return slot;
    }
}

void MetalNetwork::run(MetalSlot& slot) const {
    @autoreleasepool {
        auto& s = *slot.m_impl;
        const auto it = m_impl->graphs.find(s.batch);
        if (it == m_impl->graphs.end()) {
            throw std::logic_error("Metal: slot batch size has no graph");
        }
        const auto& g = it->second;
        NSArray<MPSGraphTensorData*>* results =
            g.pol_index == 0 ? @[ s.pol_data, s.val_data ]
                             : @[ s.val_data, s.pol_data ];

        // The completion handler runs on a Metal thread; the semaphore hands
        // the finished slot (and any error) back to this thread.
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block NSString* failure = nil;
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.completionHandler =
            ^(NSArray<MPSGraphTensorData*>* /*r*/, NSError* error) {
                if (error != nil) {
                    failure = [error localizedDescription];
                }
                dispatch_semaphore_signal(done);
            };
        {
            std::lock_guard<std::mutex> lock(m_impl->encode_mutex);
            [g.exe runAsyncWithMTLCommandQueue:m_impl->queue
                                   inputsArray:@[ s.in_data ]
                                  resultsArray:results
                           executionDescriptor:desc];
        }
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (failure != nil) {
            throw std::runtime_error("Metal: GPU error: "
                                     + std::string([failure UTF8String]));
        }
    }
}

void MetalNetwork::forward(const float* const in, const int batch,
                           float* const pol, float* const val) {
    auto& slot = m_impl->forward_slots[batch];
    if (!slot) {
        slot = make_slot(batch);
    }
    std::memcpy(slot->input(), in, sizeof(float) * batch * PLANES * PLANE);
    run(*slot);
    std::memcpy(pol, slot->policy(), sizeof(float) * batch * POL_PLANES * PLANE);
    std::memcpy(val, slot->value(), sizeof(float) * batch * VAL_PLANES * PLANE);
}
