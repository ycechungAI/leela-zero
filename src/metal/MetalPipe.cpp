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

#include "MetalPipe.h"

#include <stdexcept>

void MetalPipe::initialize(const int channels) {
    std::string error;
    m_context = MetalContext::create(error);
    if (!m_context) {
        throw std::runtime_error("Metal: " + error);
    }
    m_channels = channels;
}

void MetalPipe::push_weights(
    const unsigned int /*filter_size*/, const unsigned int /*channels*/,
    const unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {
    // The tower is one input convolution plus two per residual block.
    const auto blocks = static_cast<int>((weights->m_conv_weights.size() - 1) / 2);
    m_network = std::make_unique<MetalNetwork>(*m_context, static_cast<int>(outputs),
                                               blocks, *weights);
}

void MetalPipe::forward(const std::vector<float>& input,
                        std::vector<float>& output_pol,
                        std::vector<float>& output_val) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_network->forward(input.data(), 1, output_pol.data(), output_val.data());
}

std::string MetalPipe::describe() const {
    return m_context->describe() + ", MPSGraph, fp32, batch 1 (synchronous)";
}
