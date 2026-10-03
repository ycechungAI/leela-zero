/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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
#include <array>
#include <cstring>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#include "CPUPipe.h"
#include "Im2Col.h"
#include "Network.h"

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

void CPUPipe::initialize(int channels) {
    m_input_channels = channels;
}

namespace {

// The Winograd transforms are written once, generic over T, and run on
// several independent values at a time: T = vecf works on VEC_LANES of them
// per instruction, T = float on one. Each lane performs the scalar
// operations in the same order; only FMA contraction may differ, which moves
// results by a few ulps (well inside the G1 parity tolerance).
#if defined(__GNUC__) || defined(__clang__)
// 4 floats: NEON on arm64, SSE on x86-64.
typedef float vecf __attribute__((vector_size(16)));
#else
using vecf = float;
#endif
constexpr auto VEC_LANES = int{sizeof(vecf) / sizeof(float)};

// Unaligned load/store between T and VEC_LANES-wide float runs. The memcpy
// compiles to a single vector load/store.
template <typename T>
T load(const float* const src) {
    T v;
    std::memcpy(&v, src, sizeof(T));
    return v;
}

template <typename T>
void store(float* const dst, const T& v) {
    std::memcpy(dst, &v, sizeof(T));
}

// multiple vector [i0..i5] by Bt and produce [o0..o5]
// const auto Bt = std::array<float, WINOGRAD_TILE>{
//     1.0f,  0.0f,       -5.0f / 2.0f,  0.0f,        1.0f, 0.0f,
//     0.0f, -SQ2,        -2.0f,         SQ2 / 2.0f,  1.0f, 0.0f,
//     0.0f,  SQ2,        -2.0f,        -SQ2 / 2.0f,  1.0f, 0.0f,
//     0.0f, -SQ2 / 2.0f, -1.0f / 2.0f,  SQ2,         1.0f, 0.0f,
//     0.0f,  SQ2 / 2.0f, -1.0f / 2.0f, -SQ2,         1.0f, 0.0f,
//     0.0f,  1.0f,        0.0f,        -5.0f / 2.0f, 0.0f, 1.0f};
template <typename T>
inline void multiply_bt(T& o0, T& o1, T& o2, T& o3, T& o4, T& o5,
                        const T i0, const T i1, const T i2,
                        const T i3, const T i4, const T i5) {
    const T i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
    const T i4m2 = i2 * -2.0f + i4 * 1.0f;

    o0 = i0 + i2 * (-5.0f / 2.0f) + i4;
    o1 = i3m1 + i4m2;
    o2 = -i3m1 + i4m2;

    const T i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
    const T i4m2_2 = i2 * (-1.0f / 2.0f) + i4;

    o3 = i3m1_2 + i4m2_2;
    o4 = -i3m1_2 + i4m2_2;

    o5 = i1 + i3 * (-5.0f / 2.0f) + i5;
}

// multiple vector [i0..i5] by At and produce [o0..o3]
// const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>{
//     1.0f, 1.0f,        1.0f,        1.0f,        1.0f,       0.0f,
//     0.0f, SQ2 / 2.0f, -SQ2 / 2.0f,  SQ2,        -SQ2,        0.0f,
//     0.0f, 1.0f / 2.0f, 1.0f / 2.0f, 2.0f,        2.0f,       0.0f,
//     0.0f, SQ2 / 4.0f, -SQ2 / 4.0f,  2.0f * SQ2, -2.0f * SQ2, 1.0f};
template <typename T>
inline void multiply_at(T& o0, T& o1, T& o2, T& o3,
                        const T i0, const T i1, const T i2,
                        const T i3, const T i4, const T i5) {
    const T t1p2 = (i1 + i2) * (1.0f / 2.0f);
    const T t1m2 = (i1 - i2) * (SQ2 / 4.0f);
    const T t3p4 = i3 + i4;
    const T t3m4 = (i3 - i4) * (SQ2);

    o0 = i0 + t1p2 + t1p2 + t3p4;
    o1 = t1m2 + t1m2 + t3m4;
    o2 = t1p2 + t3p4 + t3p4;
    o3 = t1m2 + t3m4 + t3m4 + i5;
}

// Output transform of the consecutive tiles [b, b + lanes) of one output
// channel. M points at the channel's first tile in the first Winograd
// element; elements are stride apart. Y points at the channel's output plane.
// Each output is batch-normalized and passed through ReLU on its way out:
// max(0, scale * (x - mean) [+ res]), where res is the channel's residual
// plane or null.
template <typename T>
void transform_out_tiles(const float* const M, const int stride, const int b,
                         const float mean, const float scale,
                         const float* const res, float* const Y) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto lanes = int{sizeof(T) / sizeof(float)};

    T temp_m[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            temp_m[xi][nu] =
                load<T>(M + (xi * WINOGRAD_ALPHA + nu) * stride + b);
        }
    }

    // Calculates transpose(A).temp_m.A
    T temp[WINOGRAD_M][WINOGRAD_ALPHA];
    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
        multiply_at(temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                    temp_m[0][j], temp_m[1][j], temp_m[2][j],
                    temp_m[3][j], temp_m[4][j], temp_m[5][j]);
    }

    float o[WINOGRAD_M][WINOGRAD_M][lanes];
    for (auto i = 0; i < WINOGRAD_M; i++) {
        T o0, o1, o2, o3;
        multiply_at(o0, o1, o2, o3,
                    temp[i][0], temp[i][1], temp[i][2],
                    temp[i][3], temp[i][4], temp[i][5]);
        store(o[i][0], o0);
        store(o[i][1], o1);
        store(o[i][2], o2);
        store(o[i][3], o3);
    }

    for (auto lane = 0; lane < lanes; lane++) {
        const auto tile = b + lane;
        const auto y = WINOGRAD_M * (tile / WINOGRAD_WTILES);
        const auto x = WINOGRAD_M * (tile % WINOGRAD_WTILES);
        // The last row and column of tiles hang over the board edge.
        const auto rows = std::min(WINOGRAD_M, H - y);
        const auto cols = std::min(WINOGRAD_M, W - x);
        for (auto i = 0; i < rows; i++) {
            for (auto j = 0; j < cols; j++) {
                const auto idx = (y + i) * W + x + j;
                auto v = scale * (o[i][j][lane] - mean);
                if (res != nullptr) {
                    v += res[idx];
                }
                Y[idx] = std::max(0.0f, v);
            }
        }
    }
}

} // namespace

void CPUPipe::winograd_transform_in(const std::vector<float>& in,
                                    std::vector<float>& V, const int C) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;
    constexpr auto lanes = VEC_LANES;

    constexpr auto Wpad = 2 + WINOGRAD_M * WTILES;

    // Padded input of `lanes` channels, interleaved so that one vecf holds
    // the same point of every channel. The border stays zero.
    std::array<float, Wpad * Wpad * lanes> in_pad{};
    const auto pad = [&in_pad](const int y, const int x) {
        return &in_pad[(y * Wpad + x) * lanes];
    };

    // Vectorized across channels: each pass transforms channels
    // [c0, c0 + lanes); lanes past C compute zeros that are never stored.
    for (auto c0 = 0; c0 < C; c0 += lanes) {
        const auto valid = std::min(lanes, C - c0);
        for (auto lane = 0; lane < lanes; lane++) {
            const auto src = &in[(c0 + lane) * (W * H)];
            for (auto yin = 0; yin < H; yin++) {
                for (auto xin = 0; xin < W; xin++) {
                    pad(yin + 1, xin + 1)[lane] =
                        lane < valid ? src[yin * W + xin] : 0.0f;
                }
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = WINOGRAD_M * block_x;
                const auto tile = block_y * WTILES + block_x;

                // Calculates transpose(B).x.B
                vecf T1[WINOGRAD_ALPHA][WINOGRAD_ALPHA];
                for (auto xx = 0; xx < WINOGRAD_ALPHA; xx++) {
                    multiply_bt(T1[0][xx], T1[1][xx], T1[2][xx], T1[3][xx],
                                T1[4][xx], T1[5][xx],
                                load<vecf>(pad(yin + 0, xin + xx)),
                                load<vecf>(pad(yin + 1, xin + xx)),
                                load<vecf>(pad(yin + 2, xin + xx)),
                                load<vecf>(pad(yin + 3, xin + xx)),
                                load<vecf>(pad(yin + 4, xin + xx)),
                                load<vecf>(pad(yin + 5, xin + xx)));
                }

                for (auto xx = 0; xx < WINOGRAD_ALPHA; xx++) {
                    vecf o[WINOGRAD_ALPHA];
                    multiply_bt(o[0], o[1], o[2], o[3], o[4], o[5],
                                T1[xx][0], T1[xx][1], T1[xx][2],
                                T1[xx][3], T1[xx][4], T1[xx][5]);

                    // V is [element][channel][tile]. Consecutive tiles fill
                    // each run in turn, so the stores stay in L1.
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        float out[lanes];
                        store(out, o[nu]);
                        const auto dst = &V[(xx * WINOGRAD_ALPHA + nu) * C * P
                                            + c0 * P + tile];
                        for (auto lane = 0; lane < valid; lane++) {
                            dst[lane * P] = out[lane];
                        }
                    }
                }
            }
        }
    }
}

void CPUPipe::winograd_sgemm(const std::vector<float>& U,
                             const std::vector<float>& V,
                             std::vector<float>& M,
                             const int C, const int K) {
    constexpr auto P = WINOGRAD_P;

    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * P;
        const auto offset_m = b * K * P;
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
#else
        auto C_mat = EigenMatrixMap<float>(M.data() + offset_m, P, K);
        C_mat.noalias() =
            ConstEigenMatrixMap<float>(V.data() + offset_v, P, C)
            * ConstEigenMatrixMap<float>(U.data() + offset_u, K, C).transpose();
#endif
    }
}

void CPUPipe::winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y, const int K,
                                     const float* const means,
                                     const float* const stddevs,
                                     const float* const eltwise) {
    constexpr auto P = WINOGRAD_P;

    // Vectorized across tiles: a channel's P tiles are contiguous in M, so
    // VEC_LANES of them load as one vecf. The P % VEC_LANES leftover tiles
    // take the scalar path; a vector load there would read past the channel.
    for (auto k = 0; k < K; k++) {
        const auto src = &M[k * P];
        const auto dst = &Y[k * NUM_INTERSECTIONS];
        const auto res =
            eltwise == nullptr ? nullptr : &eltwise[k * NUM_INTERSECTIONS];
        auto b = 0;
        for (; b + VEC_LANES <= P; b += VEC_LANES) {
            transform_out_tiles<vecf>(src, K * P, b, means[k], stddevs[k], res,
                                      dst);
        }
        for (; b < P; b++) {
            transform_out_tiles<float>(src, K * P, b, means[k], stddevs[k], res,
                                       dst);
        }
    }
}

void CPUPipe::winograd_convolve3(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output,
                                 const float* const means,
                                 const float* const stddevs,
                                 const float* const eltwise) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs, means, stddevs, eltwise);
}

template <unsigned int filter_size>
void convolve(const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // The size of the board is defined at compile time
    constexpr unsigned int width = BOARD_SIZE;
    constexpr unsigned int height = BOARD_SIZE;
    constexpr auto num_intersections = width * height;
    constexpr auto filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * num_intersections == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, num_intersections, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], num_intersections,
                0.0f, &output[0], num_intersections);
#else
    auto C_mat =
        EigenMatrixMap<float>(output.data(), num_intersections, outputs);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(col.data(), num_intersections, filter_dim)
        * ConstEigenMatrixMap<float>(weights.data(), filter_dim, outputs);
#endif

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < num_intersections; b++) {
            output[(o * num_intersections) + b] += biases[o];
        }
    }
}

void CPUPipe::forward(const std::vector<float>& input,
                      std::vector<float>& output_pol,
                      std::vector<float>& output_val) {
    // Input convolution
    constexpr auto P = WINOGRAD_P;
    // Calculate output channels
    const auto output_channels = m_input_channels;
    // input_channels is the maximum number of input channels of any
    // convolution. Residual blocks are identical, but the first convolution
    // might be bigger when the network has very few filters
    const auto input_channels =
        std::max(static_cast<size_t>(output_channels),
                 static_cast<size_t>(Network::INPUT_CHANNELS));
    auto conv_out = std::vector<float>(output_channels * NUM_INTERSECTIONS);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * P);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * P);

    // Each convolution applies its batch norm, ReLU and (for the second
    // convolution of a block) the residual add as it writes its output.
    winograd_convolve3(output_channels, input, m_weights->m_conv_weights[0], V,
                       M, conv_out, m_weights->m_batchnorm_means[0].data(),
                       m_weights->m_batchnorm_stddevs[0].data());

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * NUM_INTERSECTIONS);
    auto res = std::vector<float>(output_channels * NUM_INTERSECTIONS);
    for (auto i = size_t{1}; i < m_weights->m_conv_weights.size(); i += 2) {
        auto output_channels = m_input_channels;
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           m_weights->m_conv_weights[i], V, M, conv_out,
                           m_weights->m_batchnorm_means[i].data(),
                           m_weights->m_batchnorm_stddevs[i].data());

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           m_weights->m_conv_weights[i + 1], V, M, conv_out,
                           m_weights->m_batchnorm_means[i + 1].data(),
                           m_weights->m_batchnorm_stddevs[i + 1].data(),
                           res.data());
    }
    convolve<1>(Network::OUTPUTS_POLICY, conv_out, m_conv_pol_w, m_conv_pol_b,
                output_pol);
    convolve<1>(Network::OUTPUTS_VALUE, conv_out, m_conv_val_w, m_conv_val_b,
                output_val);
}

void CPUPipe::push_weights(const unsigned int /*filter_size*/,
                           const unsigned int /*channels*/,
                           const unsigned int outputs,
                           std::shared_ptr<const ForwardPipeWeights> weights) {

    m_weights = weights;

    // Output head convolutions
    m_conv_pol_w = weights->m_conv_pol_w;
    m_conv_pol_b.resize(m_conv_pol_w.size() / outputs, 0.0f);
    m_conv_val_w = weights->m_conv_val_w;
    m_conv_val_b.resize(m_conv_val_w.size() / outputs, 0.0f);
}
