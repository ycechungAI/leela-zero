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

#ifndef WINOGRADKERNELS_H_INCLUDED
#define WINOGRADKERNELS_H_INCLUDED

// Metal Shading Language source for the Winograd F(4x4, 3x3) network, compiled
// at runtime by MetalWinograd.mm after it prepends
//   #define store_t float   (or half)
// Layouts, per Winograd element e = 0..35 (row-major matrices):
//   V[e]: N x C   (N = batch * 25 tiles)   input transform output
//   U[e]: C x K   weights (Network::winograd_transform_f)
//   M[e]: N x K   = V[e] * U[e]            (a plain, untransposed GEMM)
// Tile n = batch * 25 + tile_y * 5 + tile_x; its 6x6 input starts at pixel
// (4 * tile_y - 1, 4 * tile_x - 1). The transforms are literal ports of
// multiply_bt / multiply_at in CPUPipe.cpp, which are the reference.
static const char* const WINOGRAD_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float SQ2 = 1.4142135623730951f;
constant int BOARD = 19;
constant int PLANE = 361;
constant int WTILES = 5;
constant int TILES = 25;

struct Params {
    int C;   // input channels of this layer
    int K;   // output channels
    int N;   // batch * TILES
};

// Bt applied to a vector: o[0..5] from i0..i5.
inline void bt6(float i0, float i1, float i2, float i3, float i4, float i5,
                thread float* o) {
    const float i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
    const float i4m2 = i2 * -2.0f + i4 * 1.0f;
    o[0] = i0 + i2 * (-5.0f / 2.0f) + i4;
    o[1] = i3m1 + i4m2;
    o[2] = -i3m1 + i4m2;
    const float i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
    const float i4m2_2 = i2 * (-1.0f / 2.0f) + i4;
    o[3] = i3m1_2 + i4m2_2;
    o[4] = -i3m1_2 + i4m2_2;
    o[5] = i1 + i3 * (-5.0f / 2.0f) + i5;
}

// At applied to a vector: o[0..3] from i0..i5.
inline void at6(float i0, float i1, float i2, float i3, float i4, float i5,
                thread float* o) {
    const float t1p2 = (i1 + i2) * (1.0f / 2.0f);
    const float t1m2 = (i1 - i2) * (SQ2 / 4.0f);
    const float t3p4 = i3 + i4;
    const float t3m4 = (i3 - i4) * (SQ2);
    o[0] = i0 + t1p2 + t1p2 + t3p4;
    o[1] = t1m2 + t1m2 + t3m4;
    o[2] = t1p2 + t3p4 + t3p4;
    o[3] = t1m2 + t3m4 + t3m4 + i5;
}

// src: [batch][C][361]; the network input is float, activations are store_t.
// One thread per (channel, tile); channel is the fastest index so that the
// writes to V are coalesced.
// Tag only keeps the two instantiations distinct when store_t is float.
template <typename SrcT, int Tag>
kernel void in_transform(device const SrcT* src [[buffer(0)]],
                         device store_t* V [[buffer(1)]],
                         constant Params& p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const int n = int(gid.y);
    if (c >= p.C || n >= p.N) {
        return;
    }
    const int b = n / TILES;
    const int t = n - b * TILES;
    const int y0 = 4 * (t / WTILES) - 1;
    const int x0 = 4 * (t % WTILES) - 1;

    device const SrcT* plane = src + (b * p.C + c) * PLANE;
    float x[6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            const int yy = y0 + i;
            const int xx = x0 + j;
            const bool inside = yy >= 0 && yy < BOARD && xx >= 0 && xx < BOARD;
            x[i][j] = inside ? float(plane[yy * BOARD + xx]) : 0.0f;
        }
    }
    // T1[i][col]: Bt over the rows of each column.
    float T1[6][6];
    for (int col = 0; col < 6; col++) {
        float o[6];
        bt6(x[0][col], x[1][col], x[2][col], x[3][col], x[4][col], x[5][col], o);
        for (int i = 0; i < 6; i++) {
            T1[i][col] = o[i];
        }
    }
    // Element (xx, nu): Bt over each row of T1.
    for (int xx = 0; xx < 6; xx++) {
        float o[6];
        bt6(T1[xx][0], T1[xx][1], T1[xx][2], T1[xx][3], T1[xx][4], T1[xx][5], o);
        for (int nu = 0; nu < 6; nu++) {
            V[(long(xx * 6 + nu) * p.N + n) * p.C + c] = store_t(o[nu]);
        }
    }
}

template [[host_name("in_transform_float")]] kernel void in_transform<float, 0>(
    device const float*, device store_t*, constant Params&, uint2);
template [[host_name("in_transform_act")]] kernel void in_transform<store_t, 1>(
    device const store_t*, device store_t*, constant Params&, uint2);

// y = max(0, scale * (At M A - mean) [+ residual]). Y may be the residual
// buffer itself: each output pixel is read and written by one thread only.
kernel void out_transform(device const store_t* M [[buffer(0)]],
                          device store_t* Y [[buffer(1)]],
                          device const float* means [[buffer(2)]],
                          device const float* scales [[buffer(3)]],
                          constant Params& p [[buffer(4)]],
                          constant int& add_residual [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const int k = int(gid.x);
    const int n = int(gid.y);
    if (k >= p.K || n >= p.N) {
        return;
    }
    const int b = n / TILES;
    const int t = n - b * TILES;
    const int y0 = 4 * (t / WTILES);
    const int x0 = 4 * (t % WTILES);

    float m[6][6];
    for (int e = 0; e < 36; e++) {
        m[e / 6][e % 6] = float(M[(long(e) * p.N + n) * p.K + k]);
    }
    // temp[i][j]: At over the first index of each column j.
    float temp[4][6];
    for (int j = 0; j < 6; j++) {
        float o[4];
        at6(m[0][j], m[1][j], m[2][j], m[3][j], m[4][j], m[5][j], o);
        for (int i = 0; i < 4; i++) {
            temp[i][j] = o[i];
        }
    }
    const float mean = means[k];
    const float scale = scales[k];
    for (int i = 0; i < 4; i++) {
        float o[4];
        at6(temp[i][0], temp[i][1], temp[i][2], temp[i][3], temp[i][4],
            temp[i][5], o);
        const int y = y0 + i;
        for (int j = 0; j < 4; j++) {
            const int x = x0 + j;
            // The last row and column of tiles hang over the board.
            if (y < BOARD && x < BOARD) {
                const long idx = (long(b) * p.K + k) * PLANE + y * BOARD + x;
                float v = scale * (o[j] - mean);
                if (add_residual != 0) {
                    v += float(Y[idx]);
                }
                Y[idx] = store_t(max(0.0f, v));
            }
        }
    }
}

// The 1x1 heads: pol[b][0..1][pos] and val[b][0][pos] from act[b][C][pos].
// w is [3][C]: two policy rows, then the value row. No BN, no ReLU.
kernel void heads(device const store_t* act [[buffer(0)]],
                  device const float* w [[buffer(1)]],
                  device float* pol [[buffer(2)]],
                  device float* val [[buffer(3)]],
                  constant Params& p [[buffer(4)]],
                  uint2 gid [[thread_position_in_grid]]) {
    const int pos = int(gid.x);
    const int b = int(gid.y);
    if (pos >= PLANE) {
        return;
    }
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    device const store_t* in = act + long(b) * p.K * PLANE + pos;
    for (int c = 0; c < p.K; c++) {
        const float x = float(in[long(c) * PLANE]);
        a0 += w[c] * x;
        a1 += w[p.K + c] * x;
        a2 += w[2 * p.K + c] * x;
    }
    pol[(b * 2 + 0) * PLANE + pos] = a0;
    pol[(b * 2 + 1) * PLANE + pos] = a1;
    val[b * PLANE + pos] = a2;
}
)MSL";

#endif
