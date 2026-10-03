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

#ifndef METALNETWORK_H_INCLUDED
#define METALNETWORK_H_INCLUDED

#include <memory>

#include "ForwardPipe.h"
#include "MetalContext.h"

// The residual tower and both 1x1 head convolutions as one MPSGraph, with
// batch norm folded into the convolution weights. Plain C++ header; the
// graph lives in MetalNetwork.mm.
class MetalNetwork {
public:
    // Takes the raw (not Winograd) weights of ForwardPipe::ForwardPipeWeights.
    // Throws std::runtime_error with a readable message on failure.
    MetalNetwork(const MetalContext& ctx, int channels, int residual_blocks,
                 const ForwardPipe::ForwardPipeWeights& weights);
    ~MetalNetwork();
    MetalNetwork(const MetalNetwork&) = delete;
    MetalNetwork& operator=(const MetalNetwork&) = delete;

    // in: batch x 18 x 361; pol: batch x 2 x 361; val: batch x 1 x 361.
    // Synchronous. Not thread-safe: callers serialize.
    void forward(const float* in, int batch, float* pol, float* val);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif
