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

#ifndef METALPIPE_H_INCLUDED
#define METALPIPE_H_INCLUDED

#include <memory>
#include <mutex>
#include <vector>

#include "ForwardPipe.h"
#include "MetalContext.h"
#include "MetalNetwork.h"

// Metal backend, batch size 1 and synchronous. Step 2.4 replaces it with a
// batching, asynchronous MetalScheduler; this exists so the network can be
// checked against the CPU (gate G2) first.
class MetalPipe : public ForwardPipe {
public:
    // Throws std::runtime_error if Metal is unavailable.
    void initialize(int channels) override;
    void forward(const std::vector<float>& input,
                 std::vector<float>& output_pol,
                 std::vector<float>& output_val) override;
    void push_weights(
        unsigned int filter_size, unsigned int channels, unsigned int outputs,
        std::shared_ptr<const ForwardPipeWeights> weights) override;

    std::string describe() const;

private:
    std::unique_ptr<MetalContext> m_context;
    std::unique_ptr<MetalNetwork> m_network;
    int m_channels{0};
    std::mutex m_mutex; // forward() is called from every search thread
};

#endif
