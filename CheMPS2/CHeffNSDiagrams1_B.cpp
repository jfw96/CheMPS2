/*
   CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum
   chemistry
   Copyright (C) 2013-2017 Sebastian Wouters

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <math.h>

#include "CHeffNS.h"
#include "Lapack.h"
#include "MPIchemps2.h"

void CheMPS2::CHeffNS::addDiagram1A( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorX * Xleft ) {

   const int NL    = denP->gNL( ikappa );
   const int TwoSL = denP->gTwoSL( ikappa );
   const int IL    = denP->gIL( ikappa );
   const int N1    = denP->gN1( ikappa );
   const int N2    = denP->gN2( ikappa );
   const int TwoJ  = denP->gTwoJ( ikappa );
   const int NR    = denP->gNR( ikappa );
   const int TwoSR = denP->gTwoSR( ikappa );
   const int IR    = denP->gIR( ikappa );

   const int theindex = denP->gIndex();

   int dimLU = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 ) {

      dcomplex * BlockX = Xleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
      dcomplex * BlockS = denS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );

      char cotrans = 'C';
      char notrans = 'N';
      dcomplex one = 1.0;

      zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &one, BlockX, &dimLU, BlockS, &dimLD, &one, memHeff, &dimLU );
   }
}

void CheMPS2::CHeffNS::addDiagram1B( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorX * Xright ) {

   const int NL    = denP->gNL( ikappa );
   const int TwoSL = denP->gTwoSL( ikappa );
   const int IL    = denP->gIL( ikappa );
   const int N1    = denP->gN1( ikappa );
   const int N2    = denP->gN2( ikappa );
   const int TwoJ  = denP->gTwoJ( ikappa );
   const int NR    = denP->gNR( ikappa );
   const int TwoSR = denP->gTwoSR( ikappa );
   const int IR    = denP->gIR( ikappa );

   const int theindex = denP->gIndex();

   int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );
   int dimRU = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 ) {
      dcomplex * BlockX = Xright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
      dcomplex * BlockS = denS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );

      char cotrans = 'C';
      char notrans = 'N';
      dcomplex one = 1.0;

      zgemm_( &notrans, &notrans, &dimLD, &dimRU, &dimRD, &one, BlockS, &dimLD, BlockX, &dimRD, &one, memHeff, &dimLD );
   }
}

void CheMPS2::CHeffNS::addDiagram1C( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, dcomplex Helem_links ) {

   const int NL    = denP->gNL( ikappa );
   const int TwoSL = denP->gTwoSL( ikappa );
   const int IL    = denP->gIL( ikappa );
   const int N1    = denP->gN1( ikappa );
   const int N2    = denP->gN2( ikappa );
   const int TwoJ  = denP->gTwoJ( ikappa );
   const int NR    = denP->gNR( ikappa );
   const int TwoSR = denP->gTwoSR( ikappa );
   const int IR    = denP->gIR( ikappa );

   const int theindex = denP->gIndex();

   const int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   const int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 && N1 == 2 ) {
      int inc          = 1;
      int dim          = dimLD * dimRD;
      dcomplex * block = denS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
      zaxpy_( &dim, &Helem_links, block, &inc, memHeff, &inc );
   }
}

void CheMPS2::CHeffNS::addDiagram1D( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, dcomplex Helem_rechts ) {

   const int NL    = denP->gNL( ikappa );
   const int TwoSL = denP->gTwoSL( ikappa );
   const int IL    = denP->gIL( ikappa );
   const int N1    = denP->gN1( ikappa );
   const int N2    = denP->gN2( ikappa );
   const int TwoJ  = denP->gTwoJ( ikappa );
   const int NR    = denP->gNR( ikappa );
   const int TwoSR = denP->gTwoSR( ikappa );
   const int IR    = denP->gIR( ikappa );

   const int theindex = denP->gIndex();

   const int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   const int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 && N2 == 2 ) {
      int inc          = 1;
      int dim          = dimLD * dimRD;
      dcomplex * block = denS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
      zaxpy_( &dim, &Helem_rechts, block, &inc, memHeff, &inc );
   }
}
