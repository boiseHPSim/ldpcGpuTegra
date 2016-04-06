/**
  Copyright (c) 2012-2015 "Bordeaux INP, Bertrand LE GAL"
  [bertrand.legal@ims-bordeaux.fr     ]
  [http://legal.vvv.enseirb-matmeca.fr]

  This file is part of Fast_LDPC_C_decoder_for_ARM15.

  Fast_LDPC_C_decoder_for_ARM15 is free software: you can redistribute it and/or modify

  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>

#include "transpose_neon.hpp"

#define DATA_TYPE  trans_TYPE

#define t_u8x16    uint8x16_t
#define t_u8x16x2  uint8x16x2_t
#define t_u16x8x2  uint16x8x2_t
#define t_u32x4x2  uint32x4x2_t
#define t_u64x2x2  uint64x2x2_t

#define DATA_dTYPE uint8x16x2_t
#define DATA_qTYPE uint16x8x2_t
#define DATA_oTYPE uint32x4x2_t
//  vzip_u8(uint8x8_t a, uint8x8_t b);       // VZIP.8 d0,d0
/*
#define SSE_16S_ADD(a,b)             (vqaddq_s8(a,b)) //
#define SSE_16S_SUB(a,b)             (vqsubq_s8(a,b)) //
#define SSE_16S_ABS(a)               (vabsq_s8(a))    //
#define SSE_16S_MAX(a,b)             (vmaxq_s8(a,b))  //
#define SSE_16S_MIN(a,b)             (vminq_s8(a,b))  //
#define SSE_16S_XOR(a,b)             (veorq_s8(a,b))  //
#define SSE_16S_OR(a,b)              (vorrq_s8(a,b))   //
#define SSE_16S_AND(a,b)             (vandq_s8(a,b))  //
#define SSE_16S_SET1u(a)             (vdupq_n_u8(a))  //
#define SSE_16S_SET1i(a)             (vdupq_n_s8(a))  //
#define SSE_16S_SHR_BY_1(a,b)        (vshrq_n_s8(a,b))
#define SSE_16S_SHL_BY_1(a,b)        (vshlq_n_s8(a,b))
#define SSE_16S_SHR_BY_8(a)          (vshrq_n_s8(a,8))
#define SSE_16S_NEG(a)               (vqnegq_s8 (a,8))
#define SSE_16S_CMP_EQ(a,b)          (vceqq_s8(a,b))
#define SSE_16S_CMP_LTEQ(a,b)        (vcleq_s8(a,b))
#define SSE_16S_NOT(a)               (vmvnq_s8(a))
#define SSE_16S_GET_SIGN_BIT(a,b)    (SSE_16S_AND(a, b))
*/
#define cast_16_to_8(a) vreinterpretq_u8_u16(a)
#define cast_32_to_8(a) vreinterpretq_u8_u32(a)
#define cast_64_to_8(a) vreinterpretq_u8_u64(a)
#define cast_8_to_16(a) vreinterpretq_u16_u8(a)
#define cast_8_to_32(a) vreinterpretq_u32_u8(a)
#define cast_8_to_64(a) vreinterpretq_u64_u8(a)

#define LOAD_SIMD_FX(a)          (vld1q_u8((const uint8_t*)a))
#define STORE_SIMD_FX(a,b)       (vst1q_u8((uint8_t*)a,b))
#define _mm_unpacklo_epi8(a,b)   (vzipq_u8(a,b))
#define _mm_unpacklo_epi16(a,b)  (vzipq_u16(cast_8_to_16(a),cast_8_to_16(b)))
#define _mm_unpacklo_epi32(a,b)  (vzipq_u32(cast_8_to_32(a),cast_8_to_32(b)))
#define _mm_unpacklo_epi64(a,b)  (vcombine_u64(cast_8_to_64(a),cast_8_to_64(b)))
#define _mm_unpackhi_epi8(a,b)   (_mm_unpacklo_epi8 (vshrq_n_u64(a,8), vshrq_n_u64(b,8)))
#define _mm_unpackhi_epi16(a,b)  (_mm_unpacklo_epi16(vshrq_n_u64(a,8), vshrq_n_u64(b,8)))
#define _mm_unpackhi_epi32(a,b)  (_mm_unpacklo_epi32(vshrq_n_u64(a,8), vshrq_n_u64(b,8)))
#define _mm_unpackhi_epi64(a,b)  (_mm_unpacklo_epi64(vshrq_n_u64(a,8), vshrq_n_u64(b,8)))

/*
#define MY_STORE_SIMD_FX(a,b,c) \
    { \
        uint64x1_t w = vget_low_u64 (cast_8_to_64(b)); \
        uint64x1_t x = vget_high_u64(cast_8_to_64(b)); \
        uint64x1_t y = vget_low_u64 (cast_8_to_64(c)); \
        uint64x1_t z = vget_high_u64(cast_8_to_64(c)); \
        STORE_SIMD_FX(a, cast_64_to_8( vcombine_u64(w,y) )); \
        a += 1; \
        STORE_SIMD_FX(a, cast_64_to_8( vcombine_u64(x,z) )); \
        a += 1; \
    }
*/
#define COMBINE_LOW(b,c)   (cast_64_to_8( vcombine_u64(vget_low_u64 (cast_8_to_64(b)),  vget_low_u64 (cast_8_to_64(c)))))
#define COMBINE_HIGH(b,c)  (cast_64_to_8( vcombine_u64(vget_high_u64 (cast_8_to_64(b)), vget_high_u64 (cast_8_to_64(c)))))
/*
#define MY_STORE_SIMD_FX2(a,b,c) \
    { \
        uint64x1_t w = vget_low_u64 (cast_8_to_64(b)); \
        uint64x1_t x = vget_high_u64(cast_8_to_64(b)); \
        uint64x1_t y = vget_low_u64 (cast_8_to_64(c)); \
        uint64x1_t z = vget_high_u64(cast_8_to_64(c)); \
        STORE_SIMD_FX(a, cast_64_to_8( vcombine_u64(w,y) )); \
        a += N; \
        STORE_SIMD_FX(a, cast_64_to_8( vcombine_u64(x,z) )); \
        a += N; \
    }
*/
void uchar_transpose_neon(DATA_TYPE *src, DATA_TYPE *dst, int n)
{
    const int N = n / 16; // NOMBRE DE PAQUET (128 bits) PAR TRAME
    t_u8x16 *p_input  = src;
    t_u8x16 *p_output = dst;
    int loop = N;

    while( loop-- ){
//        IACA_START
        t_u8x16 *copy_p_input  = p_input;

        // STAGE 2 DU BUTTERFLY
        t_u8x16 a               = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 b               = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_1        = _mm_unpacklo_epi8(a, b);
        t_u8x16 a_0_7_b_0_7   = regs_1.val[0];
        t_u8x16 a_8_15_b_8_15 = regs_1.val[1];

        t_u8x16 c             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 d             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_2        = _mm_unpacklo_epi8(c, d);
        t_u8x16 c_0_7_d_0_7   = regs_2.val[0];
        t_u8x16 c_8_15_d_8_15 = regs_2.val[1];

        t_u16x8x2 regs_1_16     = _mm_unpacklo_epi16(a_0_7_b_0_7, c_0_7_d_0_7);
        t_u8x16 a_0_3_b_0_3_c_0_3_d_0_3 = cast_16_to_8( regs_1_16.val[0] );
        t_u8x16 a_4_7_b_4_7_c_4_7_d_4_7 = cast_16_to_8( regs_1_16.val[1] );

        t_u16x8x2 regs_2_16                     = _mm_unpacklo_epi16(a_8_15_b_8_15, c_8_15_d_8_15);
        t_u8x16 a_8_11_b_8_11_c_8_11_d_8_11     = cast_16_to_8( regs_2_16.val[0] );
        t_u8x16 a_12_15_b_12_15_c_12_15_d_12_15 = cast_16_to_8( regs_2_16.val[1] );

        t_u8x16 e             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 f             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_3        = _mm_unpacklo_epi8(e, f);
        t_u8x16 e_0_7_f_0_7   = regs_3.val[0];
        t_u8x16 e_8_15_f_8_15 = regs_3.val[1];

        t_u8x16 g             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 h             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_4        = _mm_unpacklo_epi8(g, h);
        t_u8x16 g_0_7_h_0_7   = regs_4.val[0];
        t_u8x16 g_8_15_h_8_15 = regs_4.val[1];

        t_u16x8x2 regs_3_16                     = _mm_unpacklo_epi16(e_0_7_f_0_7, g_0_7_h_0_7);
        t_u8x16 e_0_3_f_0_3_g_0_3_h_0_3         = cast_16_to_8( regs_3_16.val[0] );
        t_u8x16 e_4_7_f_4_7_g_4_7_h_4_7         = cast_16_to_8( regs_3_16.val[1] );

        t_u16x8x2 regs_4_16                     = _mm_unpacklo_epi16(e_8_15_f_8_15, g_8_15_h_8_15);
        t_u8x16 e_8_11_f_8_11_g_8_11_h_8_11     = cast_16_to_8( regs_4_16.val[0] );
        t_u8x16 e_12_15_f_12_15_g_12_15_h_12_15 = cast_16_to_8( regs_4_16.val[1] );

        t_u8x16 i             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 j             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_5      = _mm_unpacklo_epi8(i, j);
        t_u8x16 i_0_7_j_0_7   = regs_5.val[0];
        t_u8x16 i_8_15_j_8_15 = regs_5.val[1];

        t_u8x16 k             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 l             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_6      = _mm_unpacklo_epi8(k, l);
        t_u8x16 k_0_7_l_0_7   = regs_6.val[0];
        t_u8x16 k_8_15_l_8_15 = regs_6.val[1];

        t_u16x8x2 regs_5_16                     = _mm_unpacklo_epi16(i_0_7_j_0_7, k_0_7_l_0_7);
        t_u8x16 i_0_3_j_0_3_k_0_3_l_0_3         = cast_16_to_8( regs_5_16.val[0] );
        t_u8x16 i_4_7_j_4_7_k_4_7_l_4_7         = cast_16_to_8( regs_5_16.val[1] );

        t_u16x8x2 regs_6_16                     = _mm_unpacklo_epi16(i_8_15_j_8_15, k_8_15_l_8_15);
        t_u8x16 i_8_11_j_8_11_k_8_11_l_8_11     = cast_16_to_8( regs_6_16.val[0] );
        t_u8x16 i_12_15_j_12_15_k_12_15_l_12_15 = cast_16_to_8( regs_6_16.val[1] );

        t_u8x16 m             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 n             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_7      = _mm_unpacklo_epi8(m, n);
        t_u8x16 m_0_7_n_0_7   = regs_7.val[0];
        t_u8x16 m_8_15_n_8_15 = regs_7.val[1];

        t_u8x16 o             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;
        t_u8x16 p             = LOAD_SIMD_FX(copy_p_input); copy_p_input += N;

        t_u8x16x2 regs_8      = _mm_unpacklo_epi8(o, p);
        t_u8x16 o_0_7_p_0_7   = regs_8.val[0];
        t_u8x16 o_8_15_p_8_15 = regs_8.val[1];

        t_u16x8x2 regs_7_16                     = _mm_unpacklo_epi16(m_0_7_n_0_7, o_0_7_p_0_7);
        t_u8x16 m_0_3_n_0_3_o_0_3_p_0_3         = cast_16_to_8( regs_7_16.val[0] );
        t_u8x16 m_4_7_n_4_7_o_4_7_p_4_7         = cast_16_to_8( regs_7_16.val[1] );

        t_u16x8x2 regs_8_16                     = _mm_unpacklo_epi16(m_8_15_n_8_15, o_8_15_p_8_15);
        t_u8x16 m_8_11_n_8_11_o_8_11_p_8_11     = cast_16_to_8( regs_8_16.val[0] );
        t_u8x16 m_12_15_n_12_15_o_12_15_p_12_15 = cast_16_to_8( regs_8_16.val[1] );


        // STAGE 4 DU BUTTERFLY
        t_u32x4x2 regs_1_32                                                       = _mm_unpacklo_epi32(a_0_3_b_0_3_c_0_3_d_0_3, e_0_3_f_0_3_g_0_3_h_0_3);
        t_u8x16 a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1                   = cast_32_to_8( regs_1_32.val[0] );
        t_u8x16 a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3                   = cast_32_to_8( regs_1_32.val[1] );

        t_u32x4x2 regs_2_32                                                       = _mm_unpacklo_epi32(i_0_3_j_0_3_k_0_3_l_0_3, m_0_3_n_0_3_o_0_3_p_0_3);
        t_u8x16 i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1                   = cast_32_to_8( regs_2_32.val[0] );
        t_u8x16 i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3                   = cast_32_to_8( regs_2_32.val[1] );

        t_u32x4x2 regs_3_32                                                       = _mm_unpacklo_epi32(a_4_7_b_4_7_c_4_7_d_4_7, e_4_7_f_4_7_g_4_7_h_4_7);
        t_u8x16 a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5                   = cast_32_to_8( regs_3_32.val[0] );
        t_u8x16 a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7                   = cast_32_to_8( regs_3_32.val[1] );

        t_u32x4x2 regs_4_32                                                       = _mm_unpacklo_epi32(i_4_7_j_4_7_k_4_7_l_4_7, m_4_7_n_4_7_o_4_7_p_4_7);
        t_u8x16 i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5                   = cast_32_to_8( regs_4_32.val[0] );
        t_u8x16 i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7                   = cast_32_to_8( regs_4_32.val[1] );

        t_u32x4x2 regs_5_32                                                       = _mm_unpacklo_epi32(a_8_11_b_8_11_c_8_11_d_8_11, e_8_11_f_8_11_g_8_11_h_8_11);
        DATA_TYPE a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10                = cast_32_to_8( regs_5_32.val[0] );
        DATA_TYPE a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11 = cast_32_to_8( regs_5_32.val[1] );

        t_u32x4x2 regs_6_32                                                       = _mm_unpacklo_epi32(i_8_11_j_8_11_k_8_11_l_8_11, m_8_11_n_8_11_o_8_11_p_8_11);
        DATA_TYPE i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10                = cast_32_to_8( regs_6_32.val[0] );
        DATA_TYPE i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11 = cast_32_to_8( regs_6_32.val[1] );

        t_u32x4x2 regs_7_32                                                       = _mm_unpacklo_epi32(a_12_15_b_12_15_c_12_15_d_12_15, e_12_15_f_12_15_g_12_15_h_12_15);
        DATA_TYPE a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13 = cast_32_to_8( regs_7_32.val[0] );
        DATA_TYPE a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15 = cast_32_to_8( regs_7_32.val[1] );

        t_u32x4x2 regs_8_32                                                       = _mm_unpacklo_epi32(i_12_15_j_12_15_k_12_15_l_12_15, m_12_15_n_12_15_o_12_15_p_12_15);
        DATA_TYPE i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13 = cast_32_to_8( regs_8_32.val[0] );
        DATA_TYPE i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15 = cast_32_to_8( regs_8_32.val[1] );

/*
        MY_STORE_SIMD_FX(p_output, a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1,                 i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1);
        MY_STORE_SIMD_FX(p_output, a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3,                 i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3);
        MY_STORE_SIMD_FX(p_output, a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5,                 i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5);
        MY_STORE_SIMD_FX(p_output, a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7,                 i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7);
        MY_STORE_SIMD_FX(p_output, a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,                i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10);
        MY_STORE_SIMD_FX(p_output, a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11);
        MY_STORE_SIMD_FX(p_output, a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13);
        MY_STORE_SIMD_FX(p_output, a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15);
*/
        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1, i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1, i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3, i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3, i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5, i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5, i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7, i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7, i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,         i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,         i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13));

        STORE_SIMD_FX(p_output++, COMBINE_LOW(a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15));
        STORE_SIMD_FX(p_output++, COMBINE_HIGH(a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15));

        p_input  += 1;
//        IACA_END
    }
}


#define  TYPE int8x16_t
#define uTYPE uint8x16_t
#define VECTOR_LOAD(ptr)            vld1q_s8((int8_t*)ptr) //((ptr)[0])
#define VECTOR_AND(a,b)             (vandq_s8(a,b))  //
#define VECTOR_SET1u(a)             (vdupq_n_u8(a))  //
#define VECTOR_SET1i(a)             (vdupq_n_s8(a))  //
//#define LOAD_AND_DECIDE(a) VECTOR_AND(vcleq_s8(VECTOR_LOAD(a), VECTOR_SET1i(0x00)),VECTOR_SET1i(0x01))
#define VECTOR_CMP_GE(a,b)          (vcgeq_s8(a,b))

static inline TYPE LOAD_AND_DECIDE(t_u8x16* a)
{
	TYPE value = VECTOR_LOAD(a);
	TYPE zero  = VECTOR_SET1i(0x00);
	TYPE one   = VECTOR_SET1i(0x01);
	TYPE neg   = VECTOR_CMP_GE(value, zero);
	return VECTOR_AND(neg, one);
}

void uchar_itranspose_neon(DATA_TYPE *src, DATA_TYPE *dst, int n)
{
    const int N = n / 16; // NOMBRE DE PAQUET (128 bits) PAR TRAME
    t_u8x16 *p_input  = src;
    t_u8x16 *p_output = dst;
    int loop = N;

    while( loop-- ){
//        IACA_START
//        t_u8x16 *copy_p_input  = p_input;

        // STAGE 2 DU BUTTERFLY
        t_u8x16 a             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 b             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_1      = _mm_unpacklo_epi8(a, b);
        t_u8x16 a_0_7_b_0_7   = regs_1.val[0];
        t_u8x16 a_8_15_b_8_15 = regs_1.val[1];

        t_u8x16 c             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 d             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_2      = _mm_unpacklo_epi8(c, d);
        t_u8x16 c_0_7_d_0_7   = regs_2.val[0];
        t_u8x16 c_8_15_d_8_15 = regs_2.val[1];

        t_u16x8x2 regs_1_16     = _mm_unpacklo_epi16(a_0_7_b_0_7, c_0_7_d_0_7);
        t_u8x16 a_0_3_b_0_3_c_0_3_d_0_3 = cast_16_to_8( regs_1_16.val[0] );
        t_u8x16 a_4_7_b_4_7_c_4_7_d_4_7 = cast_16_to_8( regs_1_16.val[1] );

        t_u16x8x2 regs_2_16                     = _mm_unpacklo_epi16(a_8_15_b_8_15, c_8_15_d_8_15);
        t_u8x16 a_8_11_b_8_11_c_8_11_d_8_11     = cast_16_to_8( regs_2_16.val[0] );
        t_u8x16 a_12_15_b_12_15_c_12_15_d_12_15 = cast_16_to_8( regs_2_16.val[1] );

        t_u8x16 e             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 f             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_3        = _mm_unpacklo_epi8(e, f);
        t_u8x16 e_0_7_f_0_7   = regs_3.val[0];
        t_u8x16 e_8_15_f_8_15 = regs_3.val[1];

        t_u8x16 g             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 h             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_4        = _mm_unpacklo_epi8(g, h);
        t_u8x16 g_0_7_h_0_7   = regs_4.val[0];
        t_u8x16 g_8_15_h_8_15 = regs_4.val[1];

        t_u16x8x2 regs_3_16                     = _mm_unpacklo_epi16(e_0_7_f_0_7, g_0_7_h_0_7);
        t_u8x16 e_0_3_f_0_3_g_0_3_h_0_3         = cast_16_to_8( regs_3_16.val[0] );
        t_u8x16 e_4_7_f_4_7_g_4_7_h_4_7         = cast_16_to_8( regs_3_16.val[1] );

        t_u16x8x2 regs_4_16                     = _mm_unpacklo_epi16(e_8_15_f_8_15, g_8_15_h_8_15);
        t_u8x16 e_8_11_f_8_11_g_8_11_h_8_11     = cast_16_to_8( regs_4_16.val[0] );
        t_u8x16 e_12_15_f_12_15_g_12_15_h_12_15 = cast_16_to_8( regs_4_16.val[1] );

        t_u8x16 i             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 j             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_5      = _mm_unpacklo_epi8(i, j);
        t_u8x16 i_0_7_j_0_7   = regs_5.val[0];
        t_u8x16 i_8_15_j_8_15 = regs_5.val[1];

        t_u8x16 k             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 l             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_6      = _mm_unpacklo_epi8(k, l);
        t_u8x16 k_0_7_l_0_7   = regs_6.val[0];
        t_u8x16 k_8_15_l_8_15 = regs_6.val[1];

        t_u16x8x2 regs_5_16                     = _mm_unpacklo_epi16(i_0_7_j_0_7, k_0_7_l_0_7);
        t_u8x16 i_0_3_j_0_3_k_0_3_l_0_3         = cast_16_to_8( regs_5_16.val[0] );
        t_u8x16 i_4_7_j_4_7_k_4_7_l_4_7         = cast_16_to_8( regs_5_16.val[1] );

        t_u16x8x2 regs_6_16                     = _mm_unpacklo_epi16(i_8_15_j_8_15, k_8_15_l_8_15);
        t_u8x16 i_8_11_j_8_11_k_8_11_l_8_11     = cast_16_to_8( regs_6_16.val[0] );
        t_u8x16 i_12_15_j_12_15_k_12_15_l_12_15 = cast_16_to_8( regs_6_16.val[1] );

        t_u8x16 m             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 n             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_7      = _mm_unpacklo_epi8(m, n);
        t_u8x16 m_0_7_n_0_7   = regs_7.val[0];
        t_u8x16 m_8_15_n_8_15 = regs_7.val[1];

        t_u8x16 o             = LOAD_AND_DECIDE(p_input); p_input += 1;
        t_u8x16 p             = LOAD_AND_DECIDE(p_input); p_input += 1;

        t_u8x16x2 regs_8      = _mm_unpacklo_epi8(o, p);
        t_u8x16 o_0_7_p_0_7   = regs_8.val[0];
        t_u8x16 o_8_15_p_8_15 = regs_8.val[1];

        t_u16x8x2 regs_7_16                     = _mm_unpacklo_epi16(m_0_7_n_0_7, o_0_7_p_0_7);
        t_u8x16 m_0_3_n_0_3_o_0_3_p_0_3         = cast_16_to_8( regs_7_16.val[0] );
        t_u8x16 m_4_7_n_4_7_o_4_7_p_4_7         = cast_16_to_8( regs_7_16.val[1] );

        t_u16x8x2 regs_8_16                     = _mm_unpacklo_epi16(m_8_15_n_8_15, o_8_15_p_8_15);
        t_u8x16 m_8_11_n_8_11_o_8_11_p_8_11     = cast_16_to_8( regs_8_16.val[0] );
        t_u8x16 m_12_15_n_12_15_o_12_15_p_12_15 = cast_16_to_8( regs_8_16.val[1] );


        // STAGE 4 DU BUTTERFLY
        t_u32x4x2 regs_1_32                                                       = _mm_unpacklo_epi32(a_0_3_b_0_3_c_0_3_d_0_3, e_0_3_f_0_3_g_0_3_h_0_3);
        t_u8x16 a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1                   = cast_32_to_8( regs_1_32.val[0] );
        t_u8x16 a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3                   = cast_32_to_8( regs_1_32.val[1] );

        t_u32x4x2 regs_2_32                                                       = _mm_unpacklo_epi32(i_0_3_j_0_3_k_0_3_l_0_3, m_0_3_n_0_3_o_0_3_p_0_3);
        t_u8x16 i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1                   = cast_32_to_8( regs_2_32.val[0] );
        t_u8x16 i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3                   = cast_32_to_8( regs_2_32.val[1] );

        t_u32x4x2 regs_3_32                                                       = _mm_unpacklo_epi32(a_4_7_b_4_7_c_4_7_d_4_7, e_4_7_f_4_7_g_4_7_h_4_7);
        t_u8x16 a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5                   = cast_32_to_8( regs_3_32.val[0] );
        t_u8x16 a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7                   = cast_32_to_8( regs_3_32.val[1] );

        t_u32x4x2 regs_4_32                                                       = _mm_unpacklo_epi32(i_4_7_j_4_7_k_4_7_l_4_7, m_4_7_n_4_7_o_4_7_p_4_7);
        t_u8x16 i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5                   = cast_32_to_8( regs_4_32.val[0] );
        t_u8x16 i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7                   = cast_32_to_8( regs_4_32.val[1] );

        t_u32x4x2 regs_5_32                                                       = _mm_unpacklo_epi32(a_8_11_b_8_11_c_8_11_d_8_11, e_8_11_f_8_11_g_8_11_h_8_11);
        DATA_TYPE a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10                = cast_32_to_8( regs_5_32.val[0] );
        DATA_TYPE a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11 = cast_32_to_8( regs_5_32.val[1] );

        t_u32x4x2 regs_6_32                                                       = _mm_unpacklo_epi32(i_8_11_j_8_11_k_8_11_l_8_11, m_8_11_n_8_11_o_8_11_p_8_11);
        DATA_TYPE i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10                = cast_32_to_8( regs_6_32.val[0] );
        DATA_TYPE i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11 = cast_32_to_8( regs_6_32.val[1] );

        t_u32x4x2 regs_7_32                                                       = _mm_unpacklo_epi32(a_12_15_b_12_15_c_12_15_d_12_15, e_12_15_f_12_15_g_12_15_h_12_15);
        DATA_TYPE a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13 = cast_32_to_8( regs_7_32.val[0] );
        DATA_TYPE a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15 = cast_32_to_8( regs_7_32.val[1] );

        t_u32x4x2 regs_8_32                                                       = _mm_unpacklo_epi32(i_12_15_j_12_15_k_12_15_l_12_15, m_12_15_n_12_15_o_12_15_p_12_15);
        DATA_TYPE i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13 = cast_32_to_8( regs_8_32.val[0] );
        DATA_TYPE i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15 = cast_32_to_8( regs_8_32.val[1] );

        DATA_TYPE* copy_p_output1  = p_output;
        DATA_TYPE* copy_p_output2  = p_output        + N;
        DATA_TYPE* copy_p_output3  = copy_p_output2  + N;
        DATA_TYPE* copy_p_output4  = copy_p_output3  + N;
        DATA_TYPE* copy_p_output5  = copy_p_output4  + N;
        DATA_TYPE* copy_p_output6  = copy_p_output5  + N;
        DATA_TYPE* copy_p_output7  = copy_p_output6  + N;
        DATA_TYPE* copy_p_output8  = copy_p_output7  + N;
        DATA_TYPE* copy_p_output9  = copy_p_output8  + N;
        DATA_TYPE* copy_p_output10 = copy_p_output9  + N;
        DATA_TYPE* copy_p_output11 = copy_p_output10 + N;
        DATA_TYPE* copy_p_output12 = copy_p_output11 + N;
        DATA_TYPE* copy_p_output13 = copy_p_output12 + N;
        DATA_TYPE* copy_p_output14 = copy_p_output13 + N;
        DATA_TYPE* copy_p_output15 = copy_p_output14 + N;
        DATA_TYPE* copy_p_output16 = copy_p_output15 + N;

        STORE_SIMD_FX(copy_p_output1, COMBINE_LOW(a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1, i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1));
        STORE_SIMD_FX(copy_p_output2, COMBINE_HIGH(a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1, i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1));

        STORE_SIMD_FX(copy_p_output3, COMBINE_LOW(a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3, i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3));
        STORE_SIMD_FX(copy_p_output4, COMBINE_HIGH(a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3, i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3));

        STORE_SIMD_FX(copy_p_output5, COMBINE_LOW(a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5, i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5));
        STORE_SIMD_FX(copy_p_output6, COMBINE_HIGH(a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5, i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5));

        STORE_SIMD_FX(copy_p_output7, COMBINE_LOW(a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7, i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7));
        STORE_SIMD_FX(copy_p_output8, COMBINE_HIGH(a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7, i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7));

        STORE_SIMD_FX(copy_p_output9,  COMBINE_LOW(a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,         i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10));
        STORE_SIMD_FX(copy_p_output10, COMBINE_HIGH(a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,         i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10));

        STORE_SIMD_FX(copy_p_output11, COMBINE_LOW(a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11));
        STORE_SIMD_FX(copy_p_output12, COMBINE_HIGH(a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11));

        STORE_SIMD_FX(copy_p_output13, COMBINE_LOW(a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13));
        STORE_SIMD_FX(copy_p_output14, COMBINE_HIGH(a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13));

        STORE_SIMD_FX(copy_p_output15, COMBINE_LOW(a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15));
        STORE_SIMD_FX(copy_p_output16, COMBINE_HIGH(a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15));
        p_output += 1;
    }
}

void uchar_transpose_neon(unsigned char *src, unsigned char *dst, int n)
{
    uchar_transpose_neon((DATA_TYPE*)src, (DATA_TYPE*)dst, n);
}

void uchar_itranspose_neon(unsigned char *src, unsigned char *dst, int n)
{
    uchar_itranspose_neon((DATA_TYPE*)src, (DATA_TYPE*)dst, n);
}

/*
void uchar_itranspose_neon(DATA_TYPE *src, DATA_TYPE *dst, int n)
{
    const int N = n /16; // NOMBRE DE PAQUET (128 bits) PAR TRAME
    DATA_TYPE *p_input  = src;
    DATA_TYPE *p_output = dst;
    int loop = N;

    while( loop-- ){
        // STAGE 2 DU BUTTERFLY
        DATA_TYPE a             = LOAD_SIMD_FX(p_input);
        DATA_TYPE b             = LOAD_SIMD_FX(p_input + 1);
        DATA_TYPE a_0_7_b_0_7   = _mm_unpacklo_epi8(a, b);
        DATA_TYPE a_8_15_b_8_15 = _mm_unpackhi_epi8(a, b);

        DATA_TYPE c             = LOAD_SIMD_FX(p_input + 2);
        DATA_TYPE d             = LOAD_SIMD_FX(p_input + 3);
        DATA_TYPE c_0_7_d_0_7   = _mm_unpacklo_epi8(c, d);
        DATA_TYPE c_8_15_d_8_15 = _mm_unpackhi_epi8(c, d);

        DATA_TYPE a_0_3_b_0_3_c_0_3_d_0_3         = _mm_unpacklo_epi16(a_0_7_b_0_7, c_0_7_d_0_7);
        DATA_TYPE a_4_7_b_4_7_c_4_7_d_4_7         = _mm_unpackhi_epi16(a_0_7_b_0_7, c_0_7_d_0_7);
        DATA_TYPE a_8_11_b_8_11_c_8_11_d_8_11     = _mm_unpacklo_epi16(a_8_15_b_8_15, c_8_15_d_8_15);
        DATA_TYPE a_12_15_b_12_15_c_12_15_d_12_15 = _mm_unpackhi_epi16(a_8_15_b_8_15, c_8_15_d_8_15);

        DATA_TYPE e             = LOAD_SIMD_FX(p_input + 4);
        DATA_TYPE f             = LOAD_SIMD_FX(p_input + 5);
        DATA_TYPE e_0_7_f_0_7   = _mm_unpacklo_epi8(e, f);
        DATA_TYPE e_8_15_f_8_15 = _mm_unpackhi_epi8(e, f);

        DATA_TYPE g             = LOAD_SIMD_FX(p_input + 6);
        DATA_TYPE h             = LOAD_SIMD_FX(p_input + 7);
        DATA_TYPE g_0_7_h_0_7   = _mm_unpacklo_epi8(g, h);
        DATA_TYPE g_8_15_h_8_15 = _mm_unpackhi_epi8(g, h);

        DATA_TYPE e_0_3_f_0_3_g_0_3_h_0_3         = _mm_unpacklo_epi16(e_0_7_f_0_7, g_0_7_h_0_7);
        DATA_TYPE e_4_7_f_4_7_g_4_7_h_4_7         = _mm_unpackhi_epi16(e_0_7_f_0_7, g_0_7_h_0_7);
        DATA_TYPE e_8_11_f_8_11_g_8_11_h_8_11     = _mm_unpacklo_epi16(e_8_15_f_8_15, g_8_15_h_8_15);
        DATA_TYPE e_12_15_f_12_15_g_12_15_h_12_15 = _mm_unpackhi_epi16(e_8_15_f_8_15, g_8_15_h_8_15);

        DATA_TYPE i             = LOAD_SIMD_FX(p_input + 8);
        DATA_TYPE j             = LOAD_SIMD_FX(p_input + 9);
        DATA_TYPE i_0_7_j_0_7   = _mm_unpacklo_epi8(i, j);
        DATA_TYPE i_8_15_j_8_15 = _mm_unpackhi_epi8(i, j);

        DATA_TYPE k             = LOAD_SIMD_FX(p_input + 10);
        DATA_TYPE l             = LOAD_SIMD_FX(p_input + 11);
        DATA_TYPE k_0_7_l_0_7   = _mm_unpacklo_epi8(k, l);
        DATA_TYPE k_8_15_l_8_15 = _mm_unpackhi_epi8(k, l);

        DATA_TYPE i_0_3_j_0_3_k_0_3_l_0_3         = _mm_unpacklo_epi16(i_0_7_j_0_7, k_0_7_l_0_7);
        DATA_TYPE i_4_7_j_4_7_k_4_7_l_4_7         = _mm_unpackhi_epi16(i_0_7_j_0_7, k_0_7_l_0_7);
        DATA_TYPE i_8_11_j_8_11_k_8_11_l_8_11     = _mm_unpacklo_epi16(i_8_15_j_8_15, k_8_15_l_8_15);
        DATA_TYPE i_12_15_j_12_15_k_12_15_l_12_15 = _mm_unpackhi_epi16(i_8_15_j_8_15, k_8_15_l_8_15);

        DATA_TYPE m             = LOAD_SIMD_FX(p_input + 12);
        DATA_TYPE n             = LOAD_SIMD_FX(p_input + 13);
        DATA_TYPE m_0_7_n_0_7   = _mm_unpacklo_epi8(m, n);
        DATA_TYPE m_8_15_n_8_15 = _mm_unpackhi_epi8(m, n);

        DATA_TYPE o             = LOAD_SIMD_FX(p_input + 14);
        DATA_TYPE p             = LOAD_SIMD_FX(p_input + 15);
        DATA_TYPE o_0_7_p_0_7   = _mm_unpacklo_epi8(o, p);
        DATA_TYPE o_8_15_p_8_15 = _mm_unpackhi_epi8(o, p);

        DATA_TYPE m_0_3_n_0_3_o_0_3_p_0_3         = _mm_unpacklo_epi16(m_0_7_n_0_7, o_0_7_p_0_7);
        DATA_TYPE m_4_7_n_4_7_o_4_7_p_4_7         = _mm_unpackhi_epi16(m_0_7_n_0_7, o_0_7_p_0_7);
        DATA_TYPE m_8_11_n_8_11_o_8_11_p_8_11     = _mm_unpacklo_epi16(m_8_15_n_8_15, o_8_15_p_8_15);
        DATA_TYPE m_12_15_n_12_15_o_12_15_p_12_15 = _mm_unpackhi_epi16(m_8_15_n_8_15, o_8_15_p_8_15);


        // STAGE 4 DU BUTTERFLY
        DATA_TYPE a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1                 = _mm_unpacklo_epi32(a_0_3_b_0_3_c_0_3_d_0_3, e_0_3_f_0_3_g_0_3_h_0_3);
        DATA_TYPE i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1                 = _mm_unpacklo_epi32(i_0_3_j_0_3_k_0_3_l_0_3, m_0_3_n_0_3_o_0_3_p_0_3);

        DATA_TYPE* copy_p_output1  = p_output;
        DATA_TYPE* copy_p_output2  = p_output       + N;
        DATA_TYPE* copy_p_output3  = copy_p_output2 + N;
        DATA_TYPE* copy_p_output4  = copy_p_output3 + N;
        DATA_TYPE* copy_p_output5  = copy_p_output4 + N;
        DATA_TYPE* copy_p_output6  = copy_p_output5 + N;
        DATA_TYPE* copy_p_output7  = copy_p_output6 + N;
        DATA_TYPE* copy_p_output8  = copy_p_output7 + N;
        DATA_TYPE* copy_p_output9  = copy_p_output8 + N;
        DATA_TYPE* copy_p_output10 = copy_p_output9 + N;
        DATA_TYPE* copy_p_output11 = copy_p_output10 + N;
        DATA_TYPE* copy_p_output12 = copy_p_output11 + N;
        DATA_TYPE* copy_p_output13 = copy_p_output12 + N;
        DATA_TYPE* copy_p_output14 = copy_p_output13 + N;
        DATA_TYPE* copy_p_output15 = copy_p_output14 + N;
        DATA_TYPE* copy_p_output16 = copy_p_output15 + N;
        STORE_SIMD_FX(copy_p_output1, _mm_unpacklo_epi64(a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1, i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1));
        STORE_SIMD_FX(copy_p_output2, _mm_unpackhi_epi64(a_0_1_b_0_1_c_0_1_d_0_1_e_0_1_f_0_1_g_0_1_h_0_1, i_0_1_j_0_1_k_0_1_l_0_1_m_0_1_n_0_1_o_0_1_p_0_1));

        DATA_TYPE a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3                 = _mm_unpackhi_epi32(a_0_3_b_0_3_c_0_3_d_0_3, e_0_3_f_0_3_g_0_3_h_0_3);
        DATA_TYPE i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3                 = _mm_unpackhi_epi32(i_0_3_j_0_3_k_0_3_l_0_3, m_0_3_n_0_3_o_0_3_p_0_3);

        STORE_SIMD_FX(copy_p_output3, _mm_unpacklo_epi64(a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3, i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3));
        STORE_SIMD_FX(copy_p_output4, _mm_unpackhi_epi64(a_2_3_b_2_3_c_2_3_d_2_3_e_2_3_f_2_3_g_2_3_h_2_3, i_2_3_j_2_3_k_2_3_l_2_3_m_2_3_n_2_3_o_2_3_p_2_3));

        DATA_TYPE a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5                 = _mm_unpacklo_epi32(a_4_7_b_4_7_c_4_7_d_4_7,         e_4_7_f_4_7_g_4_7_h_4_7);
        DATA_TYPE i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5                 = _mm_unpacklo_epi32(i_4_7_j_4_7_k_4_7_l_4_7, m_4_7_n_4_7_o_4_7_p_4_7);

        STORE_SIMD_FX(copy_p_output5, _mm_unpacklo_epi64(a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5, i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5));
        STORE_SIMD_FX(copy_p_output6, _mm_unpackhi_epi64(a_4_5_b_4_5_c_4_5_d_4_5_e_4_5_f_4_5_g_4_5_h_4_5, i_4_5_j_4_5_k_4_5_l_4_5_m_4_5_n_4_5_o_4_5_p_4_5));

        DATA_TYPE a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7                 = _mm_unpackhi_epi32(a_4_7_b_4_7_c_4_7_d_4_7,         e_4_7_f_4_7_g_4_7_h_4_7);
        DATA_TYPE i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7                 = _mm_unpackhi_epi32(i_4_7_j_4_7_k_4_7_l_4_7, m_4_7_n_4_7_o_4_7_p_4_7);

        STORE_SIMD_FX(copy_p_output7, _mm_unpacklo_epi64(a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7, i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7));
        STORE_SIMD_FX(copy_p_output8, _mm_unpackhi_epi64(a_6_7_b_6_7_c_6_7_d_6_7_e_6_7_f_6_7_g_6_7_h_6_7, i_6_7_j_6_7_k_6_7_l_6_7_m_6_7_n_6_7_o_6_7_p_6_7));

        DATA_TYPE a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10         = _mm_unpacklo_epi32(a_8_11_b_8_11_c_8_11_d_8_11,     e_8_11_f_8_11_g_8_11_h_8_11);
        DATA_TYPE i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10         = _mm_unpacklo_epi32(i_8_11_j_8_11_k_8_11_l_8_11,     m_8_11_n_8_11_o_8_11_p_8_11);

        STORE_SIMD_FX(copy_p_output9, _mm_unpacklo_epi64(a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,         i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10));
        STORE_SIMD_FX(copy_p_output10, _mm_unpackhi_epi64(a_8_9_b_8_9_c_8_9_d_8_9_e_8_9_f_8_9_g_8_9_h_8_10,         i_8_9_j_8_9_k_8_9_l_8_9_m_8_9_n_8_9_o_8_9_p_8_10));

        DATA_TYPE i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11 = _mm_unpackhi_epi32(i_8_11_j_8_11_k_8_11_l_8_11,     m_8_11_n_8_11_o_8_11_p_8_11);
        DATA_TYPE a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11 = _mm_unpackhi_epi32(a_8_11_b_8_11_c_8_11_d_8_11,     e_8_11_f_8_11_g_8_11_h_8_11);

        STORE_SIMD_FX(copy_p_output11, _mm_unpacklo_epi64(a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11));
        STORE_SIMD_FX(copy_p_output12, _mm_unpackhi_epi64(a_10_11_b_10_11_c_10_11_d_10_11_e_10_11_f_10_11_g_10_11_h_10_11, i_10_11_j_10_11_k_10_11_l_10_11_m_10_11_n_10_11_o_10_11_p_10_11));

        DATA_TYPE i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13 = _mm_unpacklo_epi32(i_12_15_j_12_15_k_12_15_l_12_15, m_12_15_n_12_15_o_12_15_p_12_15);
        DATA_TYPE a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13 = _mm_unpacklo_epi32(a_12_15_b_12_15_c_12_15_d_12_15, e_12_15_f_12_15_g_12_15_h_12_15);

        STORE_SIMD_FX(copy_p_output13, _mm_unpacklo_epi64(a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13));
        STORE_SIMD_FX(copy_p_output14, _mm_unpackhi_epi64(a_12_13_b_12_13_c_12_13_d_12_13_e_12_13_f_12_13_g_12_13_h_12_13, i_12_13_j_12_13_k_12_13_l_12_13_m_12_13_n_12_13_o_12_13_p_12_13));

        DATA_TYPE i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15 = _mm_unpackhi_epi32(i_12_15_j_12_15_k_12_15_l_12_15, m_12_15_n_12_15_o_12_15_p_12_15);
        DATA_TYPE a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15 = _mm_unpackhi_epi32(a_12_15_b_12_15_c_12_15_d_12_15, e_12_15_f_12_15_g_12_15_h_12_15);

        STORE_SIMD_FX(copy_p_output15, _mm_unpacklo_epi64(a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15));
        STORE_SIMD_FX(copy_p_output16, _mm_unpackhi_epi64(a_14_15_b_14_15_c_14_15_d_14_15_e_14_15_f_14_15_g_14_15_h_14_15, i_14_15_j_14_15_k_14_15_l_14_15_m_14_15_n_14_15_o_14_15_p_14_15));
        p_output += 1;
        p_input  += 16;
    }
}
*/
