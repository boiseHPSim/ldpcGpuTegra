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
#include "./CDecoder_fixed_x86.h"

#include "./Constantes/constantes_sse.h"

CDecoder_fixed_x86::CDecoder_fixed_x86()
{
    var_nodes   = new short[NOEUD];
    var_mesgs   = new short[MESSAGE];
}

CDecoder_fixed_x86::~CDecoder_fixed_x86()
{
    delete var_nodes;
    delete var_mesgs;
}

void CDecoder_fixed_x86::decode(float var_nodes[], signed char Rprime_fix[], int nombre_iterations)
{
	if( nombre_iterations < 0 ){
		printf("%p\n", var_nodes);
		printf("%p\n", Rprime_fix);
	}
    // ON NE FAIT RIEN !
    // CETTE METHODE ASSURE JUSTE LA COMPATIBILITE ENTRE LES CLASSES MANIPULANT
    // DES DONNEES FLOTTANTES ET CELLES MANIPULANT DES DONNEES EN VIRGULE FIXE.
}
