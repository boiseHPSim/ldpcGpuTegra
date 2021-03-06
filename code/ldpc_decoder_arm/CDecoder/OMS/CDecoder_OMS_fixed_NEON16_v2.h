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

#ifndef CLASS_CDecoder_NEON16_OMS_V2_
#define CLASS_CDecoder_NEON16_OMS_V2_

#include "../template/CDecoder_fixed_SSE.h"
#include "cuda/decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.h"


class CDecoder_OMS_fixed_NEON16_v2 : public CDecoder_fixed_SSE{
private:
    int8x16_t** p_vn_addr;
    signed char offset;
    CGPU_Decoder_MS_SIMD_v2* gpuDecoder ;//= new CGPU_Decoder_MS_SIMD_v2( NB_THREAD_ON_GPU, _N, _K, _M );

public:
    CDecoder_OMS_fixed_NEON16_v2();
    ~CDecoder_OMS_fixed_NEON16_v2();
    void setOffset(int _offset);
    void decode(signed char var_nodes[], signed char Rprime_fix[], int nombre_iterations);

public:
    bool decode_8bits  (signed char var_nodes[], signed char Rprime_fix[], int nombre_iterations);
    bool decode_8bits_test  (signed char var_nodes[], signed char Rprime_fix[], int nombre_iterations);
};

#endif
