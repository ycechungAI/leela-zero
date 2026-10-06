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

#include "MetalTuning.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace MetalTuning {

namespace {
constexpr auto SCHEMA = "leela-zero metal tuning v1";

const char* precision_name(const MetalPrecision precision) {
    return precision == MetalPrecision::Half ? "half" : "single";
}

// Splits on tabs, keeping empty fields.
std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

bool parse_row(const std::string& line, Key& key, Measurement& m) {
    const auto f = split_tabs(line);
    // Six fields, plus an optional seventh: "ane" (Neural Engine) or
    // "winograd" (the Winograd engine). Six means MPSGraph on the GPU, which
    // is what rows written before either existed are.
    if (f.size() != 6
        && !(f.size() == 7 && (f[6] == "ane" || f[6] == "winograd"))) {
        return false;
    }
    try {
        key = Key{f[0], std::stoi(f[1]), std::stoi(f[2]),
                  f.size() == 7 && f[6] == "ane",
                  f.size() == 7 && f[6] == "winograd" ? MetalEngine::Winograd
                                                      : MetalEngine::Graph};
        if (f[3] != "single" && f[3] != "half") {
            return false;
        }
        m = Measurement{f[3] == "half" ? MetalPrecision::Half
                                       : MetalPrecision::Single,
                        std::stoi(f[4]), std::stod(f[5])};
    } catch (const std::exception&) {
        return false;
    }
    return m.batch > 0 && std::isfinite(m.evals_per_sec) && m.evals_per_sec > 0;
}

bool same_key(const Key& a, const Key& b) {
    return a.device == b.device && a.channels == b.channels
           && a.blocks == b.blocks && a.ane == b.ane && a.engine == b.engine;
}

std::string format_row(const Key& key, const Measurement& m) {
    char speed[32];
    std::snprintf(speed, sizeof(speed), "%.1f", m.evals_per_sec);
    return key.device + "\t" + std::to_string(key.channels) + "\t"
           + std::to_string(key.blocks) + "\t" + precision_name(m.precision)
           + "\t" + std::to_string(m.batch) + "\t" + speed
           + (key.ane ? "\tane"
                      : key.engine == MetalEngine::Winograd ? "\twinograd" : "");
}
} // namespace

Choice choose(const std::vector<Measurement>& measurements,
              const bool allow_single, const bool allow_half) {
    struct Best {
        bool found = false;
        int batch = 0;
        double speed = 0.0;
    };
    const auto best_for = [&measurements](const MetalPrecision precision) {
        auto top = 0.0;
        for (const auto& m : measurements) {
            if (m.precision == precision) {
                top = std::max(top, m.evals_per_sec);
            }
        }
        Best best;
        for (const auto& m : measurements) {
            if (m.precision == precision && m.evals_per_sec >= 0.95 * top
                && (!best.found || m.batch < best.batch)) {
                best = Best{true, m.batch, m.evals_per_sec};
            }
        }
        return best;
    };

    const auto single = allow_single ? best_for(MetalPrecision::Single) : Best{};
    const auto half = allow_half ? best_for(MetalPrecision::Half) : Best{};
    if (!single.found && !half.found) {
        throw std::runtime_error("Metal tuning: no usable measurements");
    }
    if (half.found && (!single.found || half.speed > 1.05 * single.speed)) {
        return Choice{MetalPrecision::Half, half.batch, half.speed};
    }
    return Choice{MetalPrecision::Single, single.batch, single.speed};
}

Choice choose_engine(const Choice& graph, const Choice& winograd) {
    auto choice = graph;
    choice.engine = MetalEngine::Graph;
    if (winograd.evals_per_sec > 1.05 * graph.evals_per_sec) {
        choice = winograd;
        choice.engine = MetalEngine::Winograd;
    }
    return choice;
}

std::string format_table(const std::vector<Measurement>& measurements) {
    std::vector<int> batches;
    for (const auto& m : measurements) {
        if (std::find(batches.begin(), batches.end(), m.batch) == batches.end()) {
            batches.push_back(m.batch);
        }
    }
    std::sort(batches.begin(), batches.end());
    const auto speed = [&measurements](const int batch,
                                       const MetalPrecision precision) {
        for (const auto& m : measurements) {
            if (m.batch == batch && m.precision == precision) {
                char text[32];
                std::snprintf(text, sizeof(text), "%10.0f", m.evals_per_sec);
                return std::string(text);
            }
        }
        return std::string("         -");
    };
    std::string table = "batch    single (evals/s)    half (evals/s)\n";
    for (const auto batch : batches) {
        char row[96];
        std::snprintf(row, sizeof(row), "%5d  %18s  %16s\n", batch,
                      speed(batch, MetalPrecision::Single).c_str(),
                      speed(batch, MetalPrecision::Half).c_str());
        table += row;
    }
    return table;
}

std::string Cache::default_path() {
    if (const auto* const env = std::getenv("LZ_METAL_TUNING_FILE")) {
        if (*env != '\0') {
            return env;
        }
    }
    const auto* const home = std::getenv("HOME");
    const auto base = std::filesystem::path(home ? home : ".");
    return (base / "Library" / "Application Support" / "leela-zero"
            / "metal_tuning")
        .string();
}

bool Cache::load(const Key& key, std::vector<Measurement>& measurements) const {
    std::ifstream file(m_path);
    if (!file) {
        return false;
    }
    std::vector<Measurement> found;
    std::string line;
    while (std::getline(file, line)) {
        Key row_key{};
        Measurement m{};
        if (!line.empty() && line[0] != '#' && parse_row(line, row_key, m)
            && same_key(row_key, key)) {
            found.push_back(m);
        }
    }
    if (found.empty()) {
        return false;
    }
    measurements = std::move(found);
    return true;
}

bool Cache::store(const Key& key,
                  const std::vector<Measurement>& measurements) const {
    try {
        // Keep every other key's rows.
        std::vector<std::string> keep;
        {
            std::ifstream file(m_path);
            std::string line;
            while (std::getline(file, line)) {
                Key row_key{};
                Measurement m{};
                if (!line.empty() && line[0] != '#'
                    && parse_row(line, row_key, m) && !same_key(row_key, key)) {
                    keep.push_back(line);
                }
            }
        }
        const auto path = std::filesystem::path(m_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        // Write a temporary file and rename it, so a crash or a second
        // process never leaves a half-written cache.
        // Unique per process: std::rand() is never seeded, so it would give
        // concurrent processes the same name.
        const auto temp =
            path.string() + ".tmp." + std::to_string(std::random_device{}());
        {
            std::ofstream out(temp, std::ios::trunc);
            if (!out) {
                return false;
            }
            out << "# " << SCHEMA << "\n";
            for (const auto& line : keep) {
                out << line << "\n";
            }
            for (const auto& m : measurements) {
                out << format_row(key, m) << "\n";
            }
            out.flush();
            if (!out) {
                std::filesystem::remove(temp);
                return false;
            }
        }
        std::filesystem::rename(temp, path);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<Measurement> measure(
    const MetalContext& context, const int channels, const int blocks,
    const ForwardPipe::ForwardPipeWeights& weights, const int workers,
    const double target_seconds, const bool ane, const MetalEngine engine,
    const std::function<void(const std::vector<Measurement>&)>& progress) {
    constexpr auto rounds = 3;
    std::vector<Measurement> all;
    for (const auto batch : candidate_batches(ane)) {
        // Both precisions for this batch size, so they can be alternated.
        const MetalNetwork single(context, channels, blocks, weights, {batch},
                                  MetalPrecision::Single, false, engine);
        const MetalNetwork half(context, channels, blocks, weights, {batch},
                                MetalPrecision::Half, ane, engine);

        // Calibrate the number of runs so a measurement takes about
        // target_seconds, whatever the network size.
        const auto probe = single.benchmark(batch, 2, workers);
        const auto seconds_per_run = static_cast<double>(batch) * workers / probe;
        const auto runs = static_cast<int>(
            std::clamp(target_seconds / seconds_per_run, 3.0, 40.0));

        std::vector<double> single_speeds, half_speeds;
        for (auto round = 0; round < rounds; round++) {
            single_speeds.push_back(single.benchmark(batch, runs, workers));
            half_speeds.push_back(half.benchmark(batch, runs, workers));
        }
        const auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        all.push_back({MetalPrecision::Single, batch, median(single_speeds)});
        all.push_back({MetalPrecision::Half, batch, median(half_speeds)});
        if (progress) {
            progress(all);
        }
    }
    return all;
}

} // namespace MetalTuning
