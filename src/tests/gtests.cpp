/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Gian-Carlo Pascutto and contributors

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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/
#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#include "GTP.h"
#include "GameState.h"
#include "NNCache.h"
#include "Platform.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

void expect_regex(const std::string& s, const std::string& re,
                  const bool positive = true) {
    auto m = std::regex_search(s, std::regex(re));
    if (positive && !m) {
        FAIL() << "Output:" << std::endl
               << s << "Does not contain:" << std::endl
               << re << std::endl;
    } else if (!positive && m) {
        FAIL() << "output:" << std::endl
               << s << "Should not contain:" << std::endl
               << re << std::endl;
    }
}

class LeelaEnv : public ::testing::Environment {
public:
    ~LeelaEnv() {}
    void SetUp() {
        GTP::setup_default_parameters();
        cfg_gtp_mode = true;

        // Setup global objects after command line has been parsed
        thread_pool.initialize(cfg_num_threads);

        // Use deterministic random numbers for hashing
        auto rng = std::make_unique<Random>(5489);
        Zobrist::init_zobrist(*rng);

        // Initialize the main thread RNG.
        // Doing this here avoids mixing in the thread_id, which
        // improves reproducibility across platforms.
        Random::get_Rng().seedrandom(cfg_rng_seed);

        cfg_weightsfile = "../src/tests/0k.txt";

        auto playouts = std::min(cfg_max_playouts, cfg_max_visits);
        auto network = std::make_unique<Network>();
        network->initialize(playouts, cfg_weightsfile);
        GTP::initialize(std::move(network));
    }
    void TearDown() {}
};

::testing::Environment* const leela_env =
    ::testing::AddGlobalTestEnvironment(new LeelaEnv);

class LeelaTest : public ::testing::Test {
public:
    LeelaTest() {
        // Reset engine parameters
        GTP::setup_default_parameters();
        cfg_max_playouts = 1;
        cfg_gtp_mode = true;

        m_gamestate = std::make_unique<GameState>();
        m_gamestate->init_game(19, 7.5f);
    }

    GameState& get_gamestate() {
        return *m_gamestate;
    }
    std::pair<std::string, std::string> gtp_execute(const std::string& cmd) {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        GTP::execute(get_gamestate(), cmd);
        return std::make_pair(testing::internal::GetCapturedStdout(),
                              testing::internal::GetCapturedStderr());
    }
    void test_analyze_cmd(const std::string& cmd, bool valid, int who,
                          int interval, int avoidlen, int avoidcolor,
                          int avoiduntil);

private:
    std::unique_ptr<GameState> m_gamestate;
};

TEST_F(LeelaTest, Startup) {
    auto maingame = get_gamestate();
}

TEST_F(LeelaTest, DefaultHash) {
    auto maingame = get_gamestate();
    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    EXPECT_EQ(hash, 0x9A930BE1616C538E);
    EXPECT_EQ(ko_hash, 0xA14C933E7669946D);
}

TEST_F(LeelaTest, Transposition) {
    auto maingame = get_gamestate();

    testing::internal::CaptureStdout();
    GTP::execute(maingame, "play b Q16");
    GTP::execute(maingame, "play w D16");
    GTP::execute(maingame, "play b D4");

    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    GTP::execute(maingame, "clear_board");

    GTP::execute(maingame, "play b D4");
    GTP::execute(maingame, "play w D16");
    GTP::execute(maingame, "play b Q16");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(hash, maingame.board.get_hash());
    EXPECT_EQ(ko_hash, maingame.board.get_ko_hash());
}

TEST_F(LeelaTest, KoPntNotSame) {
    auto maingame = get_gamestate();

    testing::internal::CaptureStdout();
    GTP::execute(maingame, "play b E6");
    GTP::execute(maingame, "play w F6");
    GTP::execute(maingame, "play b E5");
    GTP::execute(maingame, "play w F5");
    GTP::execute(maingame, "play b D4");
    GTP::execute(maingame, "play w E4");
    GTP::execute(maingame, "play b E3");
    GTP::execute(maingame, "play w G4");
    GTP::execute(maingame, "play b F4"); // capture
    GTP::execute(maingame, "play w F3");
    GTP::execute(maingame, "play b D3");

    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    GTP::execute(maingame, "clear_board");

    GTP::execute(maingame, "play b E6");
    GTP::execute(maingame, "play w F6");
    GTP::execute(maingame, "play b E5");
    GTP::execute(maingame, "play w F5");
    GTP::execute(maingame, "play b D4");
    GTP::execute(maingame, "play w E4");
    GTP::execute(maingame, "play b E3");
    GTP::execute(maingame, "play w G4");
    GTP::execute(maingame, "play b D3");
    GTP::execute(maingame, "play w F3");
    GTP::execute(maingame, "play b F4"); // capture
    std::string output = testing::internal::GetCapturedStdout();

    // Board position is the same
    EXPECT_EQ(ko_hash, maingame.board.get_ko_hash());
    // But ko (intersection) is not
    EXPECT_NE(hash, maingame.board.get_hash());
}

TEST_F(LeelaTest, MoveOnOccupiedPnt) {
    auto maingame = get_gamestate();
    std::string output;

    {
        testing::internal::CaptureStdout();
        GTP::execute(maingame, "play b D4");
        GTP::execute(maingame, "play b D4");
        output = testing::internal::GetCapturedStdout();
    }

    // Find this error in the output
    EXPECT_NE(output.find("illegal move"), std::string::npos);

    {
        testing::internal::CaptureStdout();
        GTP::execute(maingame, "play w Q16");
        GTP::execute(maingame, "play b Q16");
        output = testing::internal::GetCapturedStdout();
    }

    // Find this error in the output
    EXPECT_NE(output.find("illegal move"), std::string::npos);
}

// Basic TimeControl test
TEST_F(LeelaTest, TimeControl) {
    std::pair<std::string, std::string> result;

    // clear_board to force GTP to make a new UCTSearch.
    // This will pickup our new cfg_* settings.
    result = gtp_execute("clear_board");

    result = gtp_execute("kgs-time_settings canadian 0 120 25");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: 00:02:00, 25 stones left");
    expect_regex(result.second, "White time: 00:02:00, 25 stones left");

    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: \\S*, 24 stones left");
    expect_regex(result.second, "White time: \\S*, 25 stones left");

    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: \\S*, 24 stones left");
    expect_regex(result.second, "White time: \\S*, 24 stones left");
}

// Test changing TimeControl during game
TEST_F(LeelaTest, TimeControl2) {
    std::pair<std::string, std::string> result;

    // clear_board to force GTP to make a new UCTSearch.
    // This will pickup our new cfg_* settings.
    result = gtp_execute("clear_board");

    result = gtp_execute("kgs-time_settings byoyomi 0 100 1");
    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second,
                 "Black time: 00:01:40, 1 period\\(s\\) of 100 seconds left");
    expect_regex(result.second,
                 "White time: 00:01:40, 1 period\\(s\\) of 100 seconds left");

    result = gtp_execute("kgs-time_settings byoyomi 0 120 1");
    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second,
                 "Black time: 00:02:00, 1 period\\(s\\) of 120 seconds left");
    expect_regex(result.second,
                 "White time: 00:02:00, 1 period\\(s\\) of 120 seconds left");
}

void LeelaTest::test_analyze_cmd(const std::string& cmd, const bool valid,
                                 const int who, const int interval,
                                 const int avoidlen, const int avoidcolor,
                                 const int avoiduntil) {
    // std::cout << "testing " << cmd << std::endl;
    // avoid_until checks against the absolute game move number, indexed from 0
    std::istringstream cmdstream(cmd);
    auto maingame = get_gamestate();
    AnalyzeTags result{cmdstream, maingame};
    EXPECT_EQ(result.m_invalid, !valid);
    if (!valid) return;
    EXPECT_EQ(result.m_who, who);
    EXPECT_EQ(result.m_interval_centis, interval);
    EXPECT_EQ(result.m_moves_to_avoid.size(), avoidlen);
    if (avoidlen) {
        EXPECT_EQ(result.m_moves_to_avoid[0].color, avoidcolor);
        EXPECT_EQ(result.m_moves_to_avoid[0].until_move, avoiduntil);
    }
}

// Test parsing the lz-analyze command line
TEST_F(LeelaTest, AnalyzeParse) {
    gtp_execute("clear_board");

    test_analyze_cmd("b 50", true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("50 b", true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("b interval 50", true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("interval 50 b", true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("b interval", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("42 w", true, FastBoard::WHITE, 42, 0, -1, -1);
    test_analyze_cmd("1234", true, FastBoard::BLACK, 1234, 0, -1, -1);
    gtp_execute("play b q16");
    test_analyze_cmd("1234", true, FastBoard::WHITE, 1234, 0, -1, -1);
    test_analyze_cmd("b 100 avoid b k10 1", true, FastBoard::BLACK, 100, 1,
                     FastBoard::BLACK, 1);
    test_analyze_cmd("b 100 avoid b k10 1 avoid b a1 1", true, FastBoard::BLACK,
                     100, 2, FastBoard::BLACK, 1);
    test_analyze_cmd("b 100 avoid w k10 8", true, FastBoard::BLACK, 100, 1,
                     FastBoard::WHITE, 8);
    gtp_execute("play w q4");
    test_analyze_cmd("b 100 avoid b k10 8", true, FastBoard::BLACK, 100, 1,
                     FastBoard::BLACK, 9);
    test_analyze_cmd("100 b avoid b k10 8", true, FastBoard::BLACK, 100, 1,
                     FastBoard::BLACK, 9);
    test_analyze_cmd("b avoid b k10 8 100", true, FastBoard::BLACK, 100, 1,
                     FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b k10 8 100 b", true, FastBoard::BLACK, 100, 1,
                     FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b k10 8 100 w", true, FastBoard::WHITE, 100, 1,
                     FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b z10 8 100 w", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("avoid b k10 8 100 w bogus", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("avoid b k10 8 100 w avoid b pass 17", true,
                     FastBoard::WHITE, 100, 2, FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b k10 8 w avoid b pass 17", true, FastBoard::WHITE,
                     0, 2, FastBoard::BLACK, 9);

    gtp_execute("clear_board");
    test_analyze_cmd("b avoid b a1 10 allow b t1 1", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid w a1 10 allow b t1 1", true, FastBoard::BLACK, 0,
                     1, FastBoard::WHITE, 9);
    test_analyze_cmd("b avoid b pass 10 allow b t1 1", true, FastBoard::BLACK,
                     0, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("b avoid b resign 10 allow b t1 1", true, FastBoard::BLACK,
                     0, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("b avoid w c3,c4,d3,d4 2 avoid b pass 50", true,
                     FastBoard::BLACK, 0, 5, FastBoard::WHITE, 1);
    test_analyze_cmd("b avoid w c3,c4,d3,d4, 2 avoid b pass 50", false, -1, -1,
                     -1, -1, -1);

    gtp_execute("clear_board");
    test_analyze_cmd("b avoid b q16 1", true, FastBoard::BLACK, 0, 1,
                     FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b : 1", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b d4: 1", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b d14: 1", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b :e3 1", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b d:e3 1", false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b q16:q16 20", true, FastBoard::BLACK, 0, 1,
                     FastBoard::BLACK, 19);
    test_analyze_cmd("b avoid b q16:t19 1", true, FastBoard::BLACK, 0, 16,
                     FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b t19:q16 1", true, FastBoard::BLACK, 0, 16,
                     FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b t16:q19 1", true, FastBoard::BLACK, 0, 16,
                     FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b q19:t16 1", true, FastBoard::BLACK, 0, 16,
                     FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b a1:t19 1", true, FastBoard::BLACK, 0, 361,
                     FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b a1:t19 1 avoid w pass 1 avoid w resign 1", true,
                     FastBoard::BLACK, 0, 363, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b a1:t19,pass,resign 1", true, FastBoard::BLACK,
                     0, 363, FastBoard::BLACK, 0);
}

TEST_F(LeelaTest, AnalyzeParseMinmoves) {
    gtp_execute("clear_board");
    gtp_execute("lz-setoption name pondering value false");
    gtp_execute("lz-setoption name playouts value 1");
    auto result = gtp_execute("lz-analyze b interval 1 minmoves 5");
    // Expect to see at least 5 move priors. Counting flat matches avoids
    // nested lazy quantifiers, which backtrack heavily on long output.
    expect_regex(result.first, "info");
    const auto prior_re = std::regex("prior\\s+\\d+");
    const auto priors = std::distance(
        std::sregex_iterator(result.first.begin(), result.first.end(), prior_re),
        std::sregex_iterator());
    EXPECT_GE(priors, 5) << result.first;
}

TEST(PlatformTest, CoreCountsAreSane) {
    EXPECT_GE(Platform::num_perf_cores(), 1u);
    // Performance + efficiency cores never exceed the logical CPUs.
    EXPECT_LE(Platform::num_perf_cores() + Platform::num_eff_cores(),
              Platform::num_cpus());
#ifndef __APPLE__
    EXPECT_EQ(Platform::num_eff_cores(), 0u);
    EXPECT_FALSE(Platform::set_thread_qos_interactive());
#endif
}

#ifdef USE_METAL
#include <random>

#include "CPUPipe.h"
#include "MetalContext.h"
#include "MetalNetwork.h"
#include "MetalScheduler.h"
#include "MetalTuning.h"
#include "Network.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

// GitHub's macOS runners are virtual machines and may have no usable Metal
// device. Skip there instead of failing; on a real Mac these always run.
static bool metal_available(std::string& why) {
    std::string error;
    if (MetalContext::create(error)) {
        return true;
    }
    why = error;
    return false;
}
#define SKIP_WITHOUT_METAL()                                                   \
    do {                                                                       \
        std::string why_;                                                      \
        if (!metal_available(why_)) {                                          \
            GTEST_SKIP() << "no Metal device: " << why_;                       \
        }                                                                      \
    } while (0)

TEST(MetalContextTest, DeviceAndSelfTest) {
    SKIP_WITHOUT_METAL();
    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;
    EXPECT_FALSE(ctx->device_name().empty());
    EXPECT_TRUE(ctx->has_unified_memory());
    // Informational: GitHub's virtualized runners report an "Apple
    // Paravirtual device" that is not in an Apple GPU family, yet runs the
    // MPSGraph network correctly. Nothing requires the family yet.
    std::cout << "Apple GPU family (Apple7+): "
              << (ctx->supports_apple_gpu_family() ? "yes" : "no") << std::endl;
    EXPECT_GE(ctx->max_threads_per_threadgroup(), 256u);
    std::cout << ctx->describe() << std::endl;

    EXPECT_TRUE(ctx->self_test(error)) << error;

    // A broken shader must fail cleanly with a message, not crash.
    EXPECT_FALSE(ctx->compile_check("kernel void broken( {", error));
    EXPECT_FALSE(error.empty());
}

// A tiny random network with non-trivial batch norm (non-zero means, scales
// away from 1), so a folding mistake cannot cancel out.
static std::shared_ptr<ForwardPipe::ForwardPipeWeights> make_test_weights(
    const int C, const int blocks, std::mt19937& rng) {
    auto w = std::make_shared<ForwardPipe::ForwardPipeWeights>();
    const auto uniform = [&rng](const float lo, const float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    const auto fill = [&](const size_t n, const float lo, const float hi) {
        std::vector<float> v(n);
        for (auto& x : v) {
            x = uniform(lo, hi);
        }
        return v;
    };
    for (auto layer = 0; layer < 1 + 2 * blocks; layer++) {
        const auto inputs = layer == 0 ? Network::INPUT_CHANNELS : C;
        const auto amp = std::sqrt(2.0f / (inputs * 9));
        w->m_conv_weights.push_back(fill(C * inputs * 9, -amp, amp));
        w->m_conv_biases.push_back(std::vector<float>(C, 0.0f));
        w->m_batchnorm_means.push_back(fill(C, -0.5f, 0.5f));
        w->m_batchnorm_stddevs.push_back(fill(C, 0.5f, 2.0f));
    }
    const auto amp = std::sqrt(1.0f / C);
    w->m_conv_pol_w = fill(Network::OUTPUTS_POLICY * C, -amp, amp);
    w->m_conv_pol_b = std::vector<float>(Network::OUTPUTS_POLICY, 0.0f);
    w->m_conv_val_w = fill(Network::OUTPUTS_VALUE * C, -amp, amp);
    w->m_conv_val_b = std::vector<float>(Network::OUTPUTS_VALUE, 0.0f);
    return w;
}

TEST(MetalNetworkTest, MatchesCpuAndBatchesConsistently) {
    SKIP_WITHOUT_METAL();
    constexpr auto C = 8;
    constexpr auto blocks = 2;
    constexpr auto N = 4;
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    std::mt19937 rng(1234);
    const auto weights = make_test_weights(C, blocks, rng);

    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;
    MetalNetwork metal(*ctx, C, blocks, *weights, {1, N});

    CPUPipe cpu;
    cpu.initialize(C);
    cpu.push_weights(WINOGRAD_ALPHA, Network::INPUT_CHANNELS, C, weights);

    std::vector<float> in(N * Network::INPUT_CHANNELS * plane);
    for (auto& x : in) {
        x = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    }
    constexpr auto pol_size = Network::OUTPUTS_POLICY * plane;
    constexpr auto val_size = Network::OUTPUTS_VALUE * plane;
    const auto in_size = Network::INPUT_CHANNELS * plane;

    std::vector<float> pol_b(N * pol_size), val_b(N * val_size);
    metal.forward(in.data(), N, pol_b.data(), val_b.data());

    float worst_cpu = 0.0f, worst_batch = 0.0f, magnitude = 0.0f;
    for (auto i = 0; i < N; i++) {
        const std::vector<float> one(in.begin() + i * in_size,
                                     in.begin() + (i + 1) * in_size);
        std::vector<float> pol_c(pol_size), val_c(val_size);
        cpu.forward(one, pol_c, val_c);

        std::vector<float> pol_1(pol_size), val_1(val_size);
        metal.forward(one.data(), 1, pol_1.data(), val_1.data());

        for (auto j = 0; j < pol_size; j++) {
            worst_cpu = std::max(worst_cpu, std::abs(pol_1[j] - pol_c[j]));
            worst_batch = std::max(
                worst_batch, std::abs(pol_b[i * pol_size + j] - pol_1[j]));
            magnitude = std::max(magnitude, std::abs(pol_c[j]));
        }
        for (auto j = 0; j < val_size; j++) {
            worst_cpu = std::max(worst_cpu, std::abs(val_1[j] - val_c[j]));
            worst_batch = std::max(
                worst_batch, std::abs(val_b[i * val_size + j] - val_1[j]));
        }
    }
    // The outputs must be big enough for the comparison to mean something.
    EXPECT_GT(magnitude, 0.05f);
    EXPECT_LE(worst_cpu, 1e-4f) << "Metal vs CPU";
    EXPECT_LE(worst_batch, 1e-6f) << "batch 4 vs batch 1";
    std::cout << "Metal vs CPU max diff " << worst_cpu << ", batch vs single "
              << worst_batch << ", max |output| " << magnitude << std::endl;
}

// Reference outputs from the CPU backend, one per input.
struct EvalCase {
    std::vector<float> in, pol, val;
};

static std::vector<EvalCase> make_cases(
    const std::shared_ptr<ForwardPipe::ForwardPipeWeights>& weights,
    const int C, const int count, std::mt19937& rng) {
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    CPUPipe cpu;
    cpu.initialize(C);
    cpu.push_weights(WINOGRAD_ALPHA, Network::INPUT_CHANNELS, C, weights);
    std::vector<EvalCase> cases(count);
    for (auto& c : cases) {
        c.in.resize(Network::INPUT_CHANNELS * plane);
        for (auto& x : c.in) {
            x = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
        }
        c.pol.resize(Network::OUTPUTS_POLICY * plane);
        c.val.resize(Network::OUTPUTS_VALUE * plane);
        cpu.forward(c.in, c.pol, c.val);
    }
    return cases;
}

static float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    auto worst = 0.0f;
    for (auto i = size_t{0}; i < a.size(); i++) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

// fp16 tower, fp32 heads: results stay close to the fp32 CPU reference. The
// bound is relative to the output scale and loose enough for any plausible
// rounding, but a wrong cast or a broken fold is off by order 1.
TEST(MetalNetworkTest, HalfPrecisionIsCloseToCpu) {
    SKIP_WITHOUT_METAL();
    constexpr auto C = 32;
    constexpr auto blocks = 3;
    constexpr auto N = 4;
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    std::mt19937 rng(4321);
    const auto weights = make_test_weights(C, blocks, rng);
    const auto cases = make_cases(weights, C, N, rng);

    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;
    MetalNetwork half(*ctx, C, blocks, *weights, {N}, MetalPrecision::Half);

    std::vector<float> in;
    for (const auto& c : cases) {
        in.insert(in.end(), c.in.begin(), c.in.end());
    }
    std::vector<float> pol(N * Network::OUTPUTS_POLICY * plane);
    std::vector<float> val(N * Network::OUTPUTS_VALUE * plane);
    half.forward(in.data(), N, pol.data(), val.data());

    auto worst = 0.0f, scale = 0.0f;
    for (auto i = 0; i < N; i++) {
        for (auto j = 0; j < Network::OUTPUTS_POLICY * plane; j++) {
            worst = std::max(worst, std::abs(pol[i * Network::OUTPUTS_POLICY * plane + j]
                                             - cases[i].pol[j]));
            scale = std::max(scale, std::abs(cases[i].pol[j]));
        }
        for (auto j = 0; j < Network::OUTPUTS_VALUE * plane; j++) {
            worst = std::max(worst, std::abs(val[i * Network::OUTPUTS_VALUE * plane + j]
                                             - cases[i].val[j]));
            scale = std::max(scale, std::abs(cases[i].val[j]));
        }
    }
    std::cout << "fp16 vs CPU max diff " << worst << " (output scale " << scale
              << ")" << std::endl;
    EXPECT_GT(scale, 0.05f);
    EXPECT_LE(worst, 0.02f * scale);
    // fp16 must actually differ from fp32, or the test proves nothing.
    EXPECT_GT(worst, 0.0f);
}

// The Neural Engine compiles lazily; the constructor must do it all up front,
// keep stdout clean, and give results within fp16 tolerance. MPSGraph places
// nothing on the ANE for small towers (32x3, 64x3 and 128x3 gave results
// identical to fp16 on the GPU on an M4), so this uses 128x6, which it does
// place; the test checks that, so it cannot pass on the GPU by accident.
// The first compile on a machine can be slow, so it only runs with
// LZ_TEST_ANE=1.
TEST(MetalNetworkTest, NeuralEngineIsCloseToCpuAndKeepsStdoutClean) {
    SKIP_WITHOUT_METAL();
    if (std::getenv("LZ_TEST_ANE") == nullptr) {
        GTEST_SKIP() << "set LZ_TEST_ANE=1 (compiles for the Neural Engine)";
    }
    constexpr auto C = 128;
    constexpr auto blocks = 6;
    constexpr auto N = 4;
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    std::mt19937 rng(4321);
    const auto weights = make_test_weights(C, blocks, rng);
    const auto cases = make_cases(weights, C, N, rng);

    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;
    MetalNetwork gpu(*ctx, C, blocks, *weights, {N}, MetalPrecision::Half);

    testing::internal::CaptureStdout();
    std::unique_ptr<MetalNetwork> ane;
    ASSERT_NO_THROW(ane = std::make_unique<MetalNetwork>(
                        *ctx, C, blocks, *weights, std::vector<int>{1, N},
                        MetalPrecision::Half, true));
    std::printf("after-compile marker\n");
    const auto out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(out, "after-compile marker\n") << "nothing else on stdout";

    std::vector<float> in;
    for (const auto& c : cases) {
        in.insert(in.end(), c.in.begin(), c.in.end());
    }
    std::vector<float> pol(N * Network::OUTPUTS_POLICY * plane);
    std::vector<float> val(N * Network::OUTPUTS_VALUE * plane);
    ane->forward(in.data(), N, pol.data(), val.data());
    std::vector<float> gpu_pol(pol.size()), gpu_val(val.size());
    gpu.forward(in.data(), N, gpu_pol.data(), gpu_val.data());
    const auto vs_gpu = std::max(max_diff(pol, gpu_pol), max_diff(val, gpu_val));
    std::cout << "ANE vs GPU fp16 max diff " << vs_gpu << std::endl;
    EXPECT_GT(vs_gpu, 0.0f) << "identical to the GPU: not placed on the ANE";
    auto worst = 0.0f, scale = 0.0f;
    for (auto i = 0; i < N; i++) {
        for (auto j = 0; j < Network::OUTPUTS_POLICY * plane; j++) {
            worst = std::max(worst, std::abs(pol[i * Network::OUTPUTS_POLICY * plane + j]
                                             - cases[i].pol[j]));
            scale = std::max(scale, std::abs(cases[i].pol[j]));
        }
        for (auto j = 0; j < Network::OUTPUTS_VALUE * plane; j++) {
            worst = std::max(worst, std::abs(val[i * Network::OUTPUTS_VALUE * plane + j]
                                             - cases[i].val[j]));
            scale = std::max(scale, std::abs(cases[i].val[j]));
        }
    }
    std::cout << "ANE vs CPU max diff " << worst << " (output scale " << scale
              << ")" << std::endl;
    EXPECT_GT(scale, 0.05f);
    EXPECT_LE(worst, 0.02f * scale);
}

// ane is ignored for fp32.
TEST(MetalNetworkTest, AneFlagIsIgnoredForSinglePrecision) {
    SKIP_WITHOUT_METAL();
    constexpr auto C = 8;
    constexpr auto blocks = 1;
    std::mt19937 rng(11);
    const auto weights = make_test_weights(C, blocks, rng);
    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;
    testing::internal::CaptureStderr();
    MetalNetwork single(*ctx, C, blocks, *weights, {1}, MetalPrecision::Single,
                        true);
    const auto err = testing::internal::GetCapturedStderr();
    EXPECT_EQ(err.find("Neural Engine"), std::string::npos);
}

// The Winograd engine against the CPU reference, with non-trivial batch norm,
// for several channel counts (odd ones too) and batch 1 and 4. Batch rows must
// not leak into each other: the batch-4 result equals four batch-1 results.
static void check_winograd(const int C, const int blocks, const int N,
                           const unsigned seed, const MetalPrecision precision,
                           const float tolerance) {
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    std::mt19937 rng(seed);
    const auto weights = make_test_weights(C, blocks, rng);
    const auto cases = make_cases(weights, C, N, rng);

    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;
    MetalNetwork net(*ctx, C, blocks, *weights, {1, N}, precision, false,
                     MetalEngine::Winograd);

    std::vector<float> in;
    for (const auto& c : cases) {
        in.insert(in.end(), c.in.begin(), c.in.end());
    }
    constexpr auto pol_size = Network::OUTPUTS_POLICY * plane;
    constexpr auto val_size = Network::OUTPUTS_VALUE * plane;
    std::vector<float> pol(N * pol_size), val(N * val_size);
    net.forward(in.data(), N, pol.data(), val.data());

    auto worst_cpu = 0.0f, worst_batch = 0.0f, scale = 0.0f;
    for (auto i = 0; i < N; i++) {
        std::vector<float> pol1(pol_size), val1(val_size);
        net.forward(cases[i].in.data(), 1, pol1.data(), val1.data());
        for (auto j = 0; j < pol_size; j++) {
            worst_cpu = std::max(worst_cpu, std::abs(pol1[j] - cases[i].pol[j]));
            worst_batch = std::max(worst_batch,
                                   std::abs(pol[i * pol_size + j] - pol1[j]));
            scale = std::max(scale, std::abs(cases[i].pol[j]));
        }
        for (auto j = 0; j < val_size; j++) {
            worst_cpu = std::max(worst_cpu, std::abs(val1[j] - cases[i].val[j]));
            worst_batch = std::max(worst_batch,
                                   std::abs(val[i * val_size + j] - val1[j]));
            scale = std::max(scale, std::abs(cases[i].val[j]));
        }
    }
    std::cout << "Winograd C=" << C << " blocks=" << blocks << ": vs CPU "
              << worst_cpu << ", batch vs single " << worst_batch
              << ", output scale " << scale << std::endl;
    EXPECT_GT(scale, 0.05f) << "outputs too small to mean anything";
    EXPECT_LE(worst_cpu, tolerance * scale) << "Winograd vs CPU, C=" << C;
    EXPECT_LE(worst_batch, 1e-5f * scale) << "batch vs single, C=" << C;
}

TEST(MetalWinogradTest, MatchesCpuFp32) {
    SKIP_WITHOUT_METAL();
    check_winograd(8, 1, 4, 11, MetalPrecision::Single, 1e-4f);
    check_winograd(32, 3, 4, 12, MetalPrecision::Single, 1e-4f);
    check_winograd(17, 2, 3, 13, MetalPrecision::Single, 1e-4f); // odd channels
    check_winograd(64, 2, 8, 14, MetalPrecision::Single, 1e-4f);
}

// Many search threads, each input distinct: any mix-up between batch rows,
// slots or waiting threads shows up as a wrong answer.
static void run_concurrency_test(const MetalPrecision precision,
                                 const float tolerance) {
    constexpr auto C = 32;
    constexpr auto blocks = 2;
    constexpr auto threads = 12;
    constexpr auto iterations = 150;
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    std::mt19937 rng(99);
    const auto weights = make_test_weights(C, blocks, rng);
    const auto cases = make_cases(weights, C, 37, rng);

    MetalScheduler metal(4, 3, precision);
    metal.initialize(C);
    metal.push_weights(WINOGRAD_ALPHA, Network::INPUT_CHANNELS, C, weights);

    std::atomic<int> wrong{0};
    std::atomic<int> done{0};
    std::vector<float> worst(threads, 0.0f);
    std::vector<std::thread> pool;
    for (auto t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            std::vector<float> pol(Network::OUTPUTS_POLICY * plane);
            std::vector<float> val(Network::OUTPUTS_VALUE * plane);
            for (auto i = 0; i < iterations; i++) {
                const auto& c = cases[(t * 7 + i * 3) % cases.size()];
                metal.forward(c.in, pol, val);
                const auto d = std::max(max_diff(pol, c.pol), max_diff(val, c.val));
                worst[t] = std::max(worst[t], d);
                if (d > tolerance) {
                    wrong++;
                }
                done++;
            }
        });
    }
    for (auto& t : pool) {
        t.join();
    }
    EXPECT_EQ(done.load(), threads * iterations);
    EXPECT_EQ(wrong.load(), 0);
    std::cout << "max diff vs CPU over " << done.load() << " evals: "
              << *std::max_element(worst.begin(), worst.end()) << std::endl;
}

TEST(MetalSchedulerTest, ConcurrentEvaluationsGetTheirOwnResults) {
    SKIP_WITHOUT_METAL();
    run_concurrency_test(MetalPrecision::Single, 1e-4f);
}

// fp16: the same mix-up check with a tolerance that fp16 rounding fits in but
// a wrong row or slot (order-1 errors) does not.
TEST(MetalSchedulerTest, ConcurrentEvaluationsInHalfPrecision) {
    SKIP_WITHOUT_METAL();
    run_concurrency_test(MetalPrecision::Half, 0.05f);
}

// drain() releases every waiting thread with NetworkHaltException, and the
// scheduler works again after resume().
TEST(MetalSchedulerTest, DrainReleasesWaitersAndResumeRestarts) {
    SKIP_WITHOUT_METAL();
    constexpr auto C = 32;
    constexpr auto blocks = 2;
    constexpr auto threads = 10;
    constexpr auto plane = BOARD_SIZE * BOARD_SIZE;
    std::mt19937 rng(7);
    const auto weights = make_test_weights(C, blocks, rng);
    const auto cases = make_cases(weights, C, 8, rng);

    MetalScheduler metal(4, 3);
    metal.initialize(C);
    metal.push_weights(WINOGRAD_ALPHA, Network::INPUT_CHANNELS, C, weights);

    std::atomic<int> halted{0};
    std::atomic<int> evals{0};
    std::vector<std::thread> pool;
    for (auto t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            std::vector<float> pol(Network::OUTPUTS_POLICY * plane);
            std::vector<float> val(Network::OUTPUTS_VALUE * plane);
            try {
                for (auto i = 0;; i++) {
                    metal.forward(cases[(t + i) % cases.size()].in, pol, val);
                    evals++;
                }
            } catch (const NetworkHaltException&) {
                halted++;
            }
        });
    }
    while (evals.load() < 200) {
        std::this_thread::yield();
    }
    metal.drain();
    for (auto& t : pool) {
        t.join(); // hangs here if drain() misses a waiter
    }
    EXPECT_EQ(halted.load(), threads);

    metal.resume();
    std::vector<float> pol(Network::OUTPUTS_POLICY * plane);
    std::vector<float> val(Network::OUTPUTS_VALUE * plane);
    metal.forward(cases[0].in, pol, val);
    EXPECT_LE(max_diff(pol, cases[0].pol), 1e-4f);
    EXPECT_LE(max_diff(val, cases[0].val), 1e-4f);
}

// --- autotune: the chooser and the cache need no GPU ---

using MetalTuning::Measurement;
static Measurement S(const int batch, const double speed) {
    return {MetalPrecision::Single, batch, speed};
}
static Measurement H(const int batch, const double speed) {
    return {MetalPrecision::Half, batch, speed};
}

TEST(MetalTuningTest, ChoosesSmallestBatchWithinFivePercentOfTheBest) {
    // Plateau from batch 16: 16 is within 5% of 32's best, 8 is not.
    const std::vector<Measurement> m{S(8, 900),   S(16, 970), S(32, 1000),
                                     S(64, 990),  H(8, 900),  H(16, 970),
                                     H(32, 1000), H(64, 990)};
    const auto single_only = MetalTuning::choose(m, true, false);
    EXPECT_EQ(single_only.precision, MetalPrecision::Single);
    EXPECT_EQ(single_only.batch, 16);
    // Both allowed, equal speeds: fp16 is not 5% faster, so fp32 wins.
    EXPECT_EQ(MetalTuning::choose(m, true, true).precision,
              MetalPrecision::Single);
}

TEST(MetalTuningTest, HalfNeedsToBeAtLeastFivePercentFaster) {
    const std::vector<Measurement> slower{S(8, 1000), H(8, 1049)};
    EXPECT_EQ(MetalTuning::choose(slower, true, true).precision,
              MetalPrecision::Single);
    const std::vector<Measurement> faster{S(8, 1000), H(8, 1060)};
    EXPECT_EQ(MetalTuning::choose(faster, true, true).precision,
              MetalPrecision::Half);
    // Each precision keeps its own best batch.
    const std::vector<Measurement> mixed{S(8, 500),  S(16, 1000), H(8, 1500),
                                         H(16, 1550)};
    const auto c = MetalTuning::choose(mixed, true, true);
    EXPECT_EQ(c.precision, MetalPrecision::Half);
    EXPECT_EQ(c.batch, 8); // 1500 is within 5% of 1550
}

TEST(MetalTuningTest, HonorsTheAllowedPrecisions) {
    const std::vector<Measurement> m{S(8, 1000), H(8, 2000)};
    EXPECT_EQ(MetalTuning::choose(m, true, false).precision,
              MetalPrecision::Single);
    EXPECT_EQ(MetalTuning::choose(m, false, true).precision,
              MetalPrecision::Half);
    EXPECT_THROW(MetalTuning::choose(m, false, false), std::runtime_error);
    EXPECT_THROW(MetalTuning::choose({}, true, true), std::runtime_error);
    // Allowed but never measured.
    EXPECT_THROW(MetalTuning::choose({S(8, 1000)}, false, true),
                 std::runtime_error);
}

static std::string temp_file(const char* const name) {
    const auto dir = std::filesystem::temp_directory_path()
                     / ("lz-tuning-test-" + std::to_string(std::rand()));
    std::filesystem::create_directories(dir);
    return (dir / name).string();
}

TEST(MetalTuningTest, CacheRoundTripAndKeyIsolation) {
    const auto path = temp_file("metal_tuning");
    const MetalTuning::Cache cache(path);
    const MetalTuning::Key a{"Apple M4", 192, 15};
    const MetalTuning::Key b{"Apple M4", 256, 40};
    const MetalTuning::Key other_device{"Apple M5", 192, 15};

    std::vector<Measurement> out;
    EXPECT_FALSE(cache.load(a, out)) << "no file yet";

    ASSERT_TRUE(cache.store(a, {S(8, 357.4), H(8, 396.6)}));
    ASSERT_TRUE(cache.store(b, {S(16, 120.0)}));
    ASSERT_TRUE(cache.load(a, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].precision, MetalPrecision::Single);
    EXPECT_EQ(out[0].batch, 8);
    EXPECT_NEAR(out[0].evals_per_sec, 357.4, 0.06);
    EXPECT_EQ(out[1].precision, MetalPrecision::Half);
    ASSERT_TRUE(cache.load(b, out));
    EXPECT_EQ(out.size(), 1u);
    EXPECT_FALSE(cache.load(other_device, out)) << "device is part of the key";

    // The Neural Engine is a separate key.
    const MetalTuning::Key a_ane{"Apple M4", 192, 15, true};
    EXPECT_FALSE(cache.load(a_ane, out)) << "ane is part of the key";
    ASSERT_TRUE(cache.store(a_ane, {H(8, 760.0)}));
    ASSERT_TRUE(cache.load(a_ane, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].precision, MetalPrecision::Half);
    ASSERT_TRUE(cache.load(a, out));
    EXPECT_EQ(out.size(), 2u) << "the non-ANE rows are untouched";

    // Storing a key again replaces its rows and keeps the others.
    ASSERT_TRUE(cache.store(a, {S(32, 400.0)}));
    ASSERT_TRUE(cache.load(a, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].batch, 32);
    EXPECT_TRUE(cache.load(b, out)) << "other keys survive";
    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(MetalTuningTest, CacheToleratesGarbage) {
    const auto path = temp_file("metal_tuning");
    {
        std::ofstream f(path);
        f << "# some header\n"
             "not a row\n"
             "Apple M4\t192\t15\tsingle\t8\t350.0\n"
             "Apple M4\t192\t15\thalf\tEIGHT\t400.0\n"   // bad number
             "Apple M4\t192\t15\tquarter\t8\t400.0\n"     // bad precision
             "Apple M4\t192\t15\thalf\t8\t-5\n"           // bad speed
             "Apple M4\t192\t15\thalf\t16\t410.0\t\textra\n"
             "Apple M4\t192\t15\thalf\t16\t410.0\n";
    }
    const MetalTuning::Cache cache(path);
    std::vector<Measurement> out;
    ASSERT_TRUE(cache.load({"Apple M4", 192, 15}, out));
    EXPECT_EQ(out.size(), 2u) << "only the two well-formed rows";
    // A directory that cannot be created is a quiet failure, not an exception.
    const MetalTuning::Cache unwritable("/dev/null/nope/metal_tuning");
    EXPECT_FALSE(unwritable.store({"Apple M4", 1, 1}, {S(8, 1.0)}));
    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

// Rows written before the Neural Engine existed (six fields) are GPU rows.
TEST(MetalTuningTest, LegacyRowsAreNotAneRows) {
    const auto path = temp_file("metal_tuning");
    {
        std::ofstream f(path);
        f << "Apple M4\t192\t15\thalf\t8\t400.0\n"
             "Apple M4\t192\t15\thalf\t8\t760.0\tane\n"
             "Apple M4\t192\t15\thalf\t16\t760.0\tgpu\n"; // bad flag
    }
    const MetalTuning::Cache cache(path);
    std::vector<Measurement> out;
    ASSERT_TRUE(cache.load({"Apple M4", 192, 15, false}, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].evals_per_sec, 400.0, 0.01);
    ASSERT_TRUE(cache.load({"Apple M4", 192, 15, true}, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].evals_per_sec, 760.0, 0.01);
    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST(MetalTuningTest, NeuralEngineTriesFewerBatchSizes) {
    EXPECT_EQ(MetalTuning::candidate_batches(false),
              (std::vector<int>{8, 16, 32, 64}));
    EXPECT_EQ(MetalTuning::candidate_batches(true), (std::vector<int>{8, 16}));
}

// The GPU part: every candidate batch size, both precisions, plausible numbers.
TEST(MetalTuningTest, MeasuresEveryCandidateOnTheGpu) {
    SKIP_WITHOUT_METAL();
    constexpr auto C = 16;
    constexpr auto blocks = 2;
    std::mt19937 rng(5);
    const auto weights = make_test_weights(C, blocks, rng);
    std::string error;
    const auto ctx = MetalContext::create(error);
    ASSERT_NE(ctx, nullptr) << error;

    auto reports = 0;
    const auto m = MetalTuning::measure(
        *ctx, C, blocks, *weights, 2, 0.01, false,
        [&reports](const std::vector<Measurement>&) { reports++; });
    EXPECT_EQ(reports, 4) << "one progress report per batch size";
    EXPECT_EQ(m.size(), 8u);
    for (const auto& r : m) {
        EXPECT_GT(r.evals_per_sec, 0.0);
    }
    EXPECT_NO_THROW(MetalTuning::choose(m, true, true));
}
#endif
