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

#include "MetalScheduler.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

#include "Network.h"
#include "Utils.h"

namespace {
constexpr auto PLANE = BOARD_SIZE * BOARD_SIZE;
constexpr auto IN_SIZE = Network::INPUT_CHANNELS * PLANE;
constexpr auto POL_SIZE = Network::OUTPUTS_POLICY * PLANE;
constexpr auto VAL_SIZE = Network::OUTPUTS_VALUE * PLANE;
} // namespace

MetalScheduler::MetalScheduler(const int max_batch, const int workers,
                               const MetalPrecision precision, const bool ane)
    : m_max_batch(std::max(1, max_batch)),
      m_workers(std::max(1, workers)),
      m_precision(precision),
      m_ane(ane && precision == MetalPrecision::Half) {}

MetalScheduler::~MetalScheduler() {
    m_queue.shutdown();
    for (auto& t : m_threads) {
        t.join();
    }
}

void MetalScheduler::initialize(const int /*channels*/) {
    std::string error;
    m_context = MetalContext::create(error);
    if (!m_context) {
        throw std::runtime_error("Metal: " + error);
    }
}

void MetalScheduler::push_weights(
    const unsigned int /*filter_size*/, const unsigned int /*channels*/,
    const unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {
    assert(m_threads.empty());
    // BatchQueue::pickup() returns either a full batch or a single entry, so
    // those are the only two graphs needed.
    const auto blocks =
        static_cast<int>((weights->m_conv_weights.size() - 1) / 2);
    m_network = std::make_unique<MetalNetwork>(
        *m_context, static_cast<int>(outputs), blocks, *weights,
        std::vector<int>{1, m_max_batch}, m_precision, m_ane);
    for (auto i = 0; i < m_workers; i++) {
        m_threads.emplace_back(&MetalScheduler::worker, this);
    }
}

void MetalScheduler::forward(const std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val) {
    if (!m_queue.submit(input, output_pol, output_val)) {
        throw NetworkHaltException();
    }
}

void MetalScheduler::drain() {
    m_queue.drain();
}

void MetalScheduler::resume() {
    // UCTSearch drains, waits for its threads to finish, then resumes.
    assert(m_queue.empty());
    m_queue.resume();
}

double MetalScheduler::benchmark(const int runs) const {
    return m_network->benchmark(m_max_batch, runs, m_workers);
}

std::string MetalScheduler::describe() const {
    return m_context->describe() + ", MPSGraph, "
           + (m_precision == MetalPrecision::Half ? "fp16" : "fp32")
           + (m_ane ? " (Neural Engine)" : "") + ", batch "
           + std::to_string(m_max_batch) + ", " + std::to_string(m_workers)
           + " workers";
}

void MetalScheduler::worker() {
    try {
        const auto single = m_network->make_slot(1);
        const auto full = m_network->make_slot(m_max_batch);
        while (true) {
            auto batch = m_queue.pickup(static_cast<size_t>(m_max_batch));
            if (batch.empty()) {
                return; // shutdown
            }
            // pickup() gives 1 or m_max_batch entries. Anything in between
            // would still work: the full slot's unused rows are ignored.
            auto& slot = batch.size() == 1 ? *single : *full;

            auto in = slot.input();
            for (const auto& entry : batch) {
                assert(entry->in.size() == IN_SIZE);
                std::copy(begin(entry->in), end(entry->in), in);
                in += IN_SIZE;
            }

            m_network->run(slot);

            auto pol = slot.policy();
            auto val = slot.value();
            for (const auto& entry : batch) {
                std::copy(pol, pol + POL_SIZE, begin(entry->out_p));
                std::copy(val, val + VAL_SIZE, begin(entry->out_v));
                pol += POL_SIZE;
                val += VAL_SIZE;
            }
            m_queue.complete(batch);
        }
    } catch (const std::exception& e) {
        // Search threads are blocked on this batch; there is no way to
        // recover them.
        Utils::myprintf_error("%s\n", e.what());
        std::abort();
    }
}
