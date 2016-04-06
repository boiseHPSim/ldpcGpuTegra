/**
  Copyright (c) 2012-2015 "Bordeaux INP, Bertrand LE GAL"
  [http://legal.vvv.enseirb-matmeca.fr]
  This file is part of LDPC_C_Simulator.
  LDPC_C_Simulator is free software: you can redistribute it and/or modify
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
#ifndef __CDecoder_fixed_SSE__
#define __CDecoder_fixed_SSE__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>

#include "./Constantes/constantes_sse.h"
#include "./CDecoder_fixed.h"

class CDecoder_fixed_SSE : public CDecoder_fixed{
protected:
    int8x16_t* var_nodes;
    int8x16_t* var_mesgs;

public:
    CDecoder_fixed_SSE();
    virtual ~CDecoder_fixed_SSE();
    void decode(float var_nodes[], signed char Rprime_fix[], int nombre_iterations);
};

#endif
