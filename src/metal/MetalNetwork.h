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
#include <vector>

#include "ForwardPipe.h"
#include "MetalContext.h"

class MetalNetwork;

// Precision of the residual tower. The 1x1 heads always run in fp32, so the
// policy and value logits cannot overflow; only the tower uses fp16. Inputs
// and outputs are fp32 in both modes (the cast is part of the graph).
enum class MetalPrecision { Single, Half };

// How the tower is computed: Graph is MPSGraph direct convolution; Winograd is
// the Winograd F(4x4, 3x3) kernels in MetalWinograd.mm (step 2.11).
enum class MetalEngine { Graph, Winograd };

// Shared-memory input and output buffers for one batch size. The CPU writes
// the inputs and reads the outputs in place; the GPU uses the same memory, so
// nothing is copied to or from the device. A slot must be used by one thread
// at a time, and not touched while MetalNetwork::run() is using it.
class MetalSlot {
public:
    ~MetalSlot();
    MetalSlot(const MetalSlot&) = delete;
    MetalSlot& operator=(const MetalSlot&) = delete;

    int batch() const;
    float* input();               // batch x 18 x 361
    const float* policy() const;  // batch x 2 x 361
    const float* value() const;   // batch x 1 x 361

private:
    friend class MetalNetwork;
    MetalSlot();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// The residual tower and both 1x1 head convolutions as compiled MPSGraphs
// (one per batch size), with batch norm folded into the convolution weights.
// Plain C++ header; the graphs live in MetalNetwork.mm.
class MetalNetwork {
public:
    // Compiles a graph for each of batch_sizes from the raw (not Winograd)
    // weights of ForwardPipe::ForwardPipeWeights. The weights are not kept.
    // With ane (fp16 only; ignored for fp32), the placement pass may run the
    // tower on the Neural Engine. MPSGraph then compiles each graph lazily on
    // its first run, which takes minutes for a network it has not seen before
    // (the system caches the result), so the constructor runs every graph once
    // and prints a notice to stderr. While it does, stdout is redirected to
    // /dev/null, because the compiler prints "error:" lines there; call this
    // only when nothing else is writing to stdout.
    // Throws std::runtime_error with a readable message on failure.
    MetalNetwork(const MetalContext& ctx, int channels, int residual_blocks,
                 const ForwardPipe::ForwardPipeWeights& weights,
                 const std::vector<int>& batch_sizes,
                 MetalPrecision precision = MetalPrecision::Single,
                 bool ane = false, MetalEngine engine = MetalEngine::Graph);
    ~MetalNetwork();
    MetalNetwork(const MetalNetwork&) = delete;
    MetalNetwork& operator=(const MetalNetwork&) = delete;

    // Buffers for one of the batch sizes given to the constructor.
    std::unique_ptr<MetalSlot> make_slot(int batch) const;

    // Evaluates the slot's inputs into its outputs and returns when the GPU
    // is done. Thread-safe: several threads may run their own slots at once;
    // encoding is serialized and the GPU work overlaps.
    // Throws std::runtime_error if the GPU reports an error.
    void run(MetalSlot& slot) const;

    // Evaluations per second with `streams` threads each running `runs`
    // back-to-back batches of this size on their own slot, after one warm-up
    // run each. Use the number of workers as `streams` to measure what the
    // scheduler will see. Not thread-safe. Throws like run().
    double benchmark(int batch, int runs, int streams = 1) const;

    // Convenience for tests: copies in, runs, copies out. Not thread-safe.
    // in: batch x 18 x 361; pol: batch x 2 x 361; val: batch x 1 x 361.
    void forward(const float* in, int batch, float* pol, float* val);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif
