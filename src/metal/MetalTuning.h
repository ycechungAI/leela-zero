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

#ifndef METALTUNING_H_INCLUDED
#define METALTUNING_H_INCLUDED

#include <functional>
#include <string>
#include <vector>

#include "ForwardPipe.h"
#include "MetalContext.h"
#include "MetalNetwork.h"

// Choosing the Metal batch size and precision by measurement, and remembering
// the measurements. Plain C++: only measure() touches the GPU, through
// MetalNetwork.
//
// The cache holds speed only, keyed by device and network shape. Speed does
// not depend on the weight values, so a new generation of the same network
// reuses it. Whether fp16 is accurate does depend on the weights, so that is
// checked at every start (Network::init_metal), never cached. The Neural
// Engine is part of the key, as it changes the fp16 speed completely.
namespace MetalTuning {

// Batch sizes tried, smallest first. On the Neural Engine only 8 and 16: each
// graph compiles for minutes the first time, and the time grows with the
// batch (330 s at 8, 680 s at 16 on a 15b x 192 net), so 32 and 64 would cost
// an hour for sizes that the GPU-bound plateau does not need.
inline const std::vector<int>& candidate_batches(const bool ane = false) {
    static const std::vector<int> gpu{8, 16, 32, 64};
    static const std::vector<int> engine{8, 16};
    return ane ? engine : gpu;
}

struct Key {
    std::string device;
    int channels;
    int blocks;
    // The fp16 rows were measured with the tower on the Neural Engine. The
    // fp32 rows do not depend on it, but are stored per key anyway.
    bool ane = false;
};

struct Measurement {
    MetalPrecision precision;
    int batch;
    double evals_per_sec;
};

struct Choice {
    MetalPrecision precision;
    int batch;
};

// The choice for the measurements. Per precision, the smallest batch within 5%
// of that precision's best throughput (fewer search threads search better);
// fp16 only if it is at least 5% faster than fp32 at their chosen batches.
// A precision not allowed is ignored. Throws std::runtime_error if nothing
// allowed was measured.
Choice choose(const std::vector<Measurement>& measurements, bool allow_single,
              bool allow_half);

// Measurements as a table, one row per batch size.
std::string format_table(const std::vector<Measurement>& measurements);

class Cache {
public:
    explicit Cache(std::string path) : m_path(std::move(path)) {}

    // $LZ_METAL_TUNING_FILE, else ~/Library/Application Support/leela-zero/
    // metal_tuning.
    static std::string default_path();

    // False if the key is not cached. Unreadable or malformed lines are
    // skipped, never fatal.
    bool load(const Key& key, std::vector<Measurement>& measurements) const;
    // Replaces the key's entry and keeps the others. Returns false (and does
    // not throw) if the file cannot be written.
    bool store(const Key& key, const std::vector<Measurement>& measurements) const;

private:
    std::string m_path;
};

// Times every candidate batch size in both precisions (fp16 on the Neural
// Engine if ane) with one stream per worker, alternating precisions and taking
// medians. Each measurement runs
// for about target_seconds. progress(batch_rows) is called after each batch
// size with the rows measured so far. Throws std::runtime_error on GPU
// failure.
std::vector<Measurement> measure(
    const MetalContext& context, int channels, int blocks,
    const ForwardPipe::ForwardPipeWeights& weights, int workers,
    double target_seconds, bool ane,
    const std::function<void(const std::vector<Measurement>&)>& progress);

} // namespace MetalTuning

#endif
