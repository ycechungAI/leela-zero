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

#ifndef METALSCHEDULER_H_INCLUDED
#define METALSCHEDULER_H_INCLUDED

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "BatchQueue.h"
#include "ForwardPipe.h"
#include "MetalContext.h"
#include "MetalNetwork.h"

// Metal backend. Search threads submit evaluations to a BatchQueue; worker
// threads take whole batches, fill their own shared-memory slot, run it on
// the GPU and hand the outputs back.
//
// Each worker owns its slots for its whole life, so a slot's states (free,
// filling, submitted, done) are just that thread's program order: no slot
// state is shared between threads, and nothing needs a lock-free state
// machine. With W workers, up to W batches are in flight; encoding is
// serialized inside MetalNetwork::run() and GPU execution overlaps, so one
// worker fills or reads back its slot while another's batch runs.
class MetalScheduler : public ForwardPipe {
public:
    // Batches in flight. Two hide the CPU-side work behind the GPU; a third
    // measured no faster on the M4 (BENCHMARKS.md, step 2.4).
    static constexpr int DEFAULT_WORKERS = 2;

    explicit MetalScheduler(int max_batch, int workers = DEFAULT_WORKERS,
                            MetalPrecision precision = MetalPrecision::Single);
    ~MetalScheduler() override;

    // Throws std::runtime_error if Metal is unavailable.
    void initialize(int channels) override;
    // Compiles the graphs and starts the workers. Call once.
    void push_weights(
        unsigned int filter_size, unsigned int channels, unsigned int outputs,
        std::shared_ptr<const ForwardPipeWeights> weights) override;
    // Throws NetworkHaltException if the queue is drained while waiting.
    void forward(const std::vector<float>& input,
                 std::vector<float>& output_pol,
                 std::vector<float>& output_val) override;
    void drain() override;
    void resume() override;

    std::string describe() const;

    // GPU-only throughput in evaluations per second at the full batch size
    // with one stream per worker, measured on `runs` batches each. Call while
    // no search is running.
    double benchmark(int runs) const;

private:
    void worker();

    const int m_max_batch;
    const int m_workers;
    const MetalPrecision m_precision;
    std::unique_ptr<MetalContext> m_context;
    std::unique_ptr<MetalNetwork> m_network;
    BatchQueue m_queue;
    std::vector<std::thread> m_threads;
};

#endif
