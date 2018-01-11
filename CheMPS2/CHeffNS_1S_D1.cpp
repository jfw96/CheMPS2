
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CHeffNS_1S.h"
#include "Davidson.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Wigner.h"

void CheMPS2::CHeffNS_1S::addDiagram1A( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorX * Xleft ) {
   const int index = in->gIndex();

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = bk_up->gIrrep( index );

   int dimLU = bk_up->gCurrentDim( index, NL, TwoSL, IL );
   int dimLD = bk_down->gCurrentDim( index, NL, TwoSL, IL );
   int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 ) {

      dcomplex * BlockX = Xleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
      dcomplex * BlockS = in->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

      char cotrans = 'C';
      char notrans = 'N';
      dcomplex one = 1.0;

      zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &one, BlockX, &dimLU, BlockS, &dimLD, &one, memHeff, &dimLU );
   }
}

void CheMPS2::CHeffNS_1S::addDiagram1B( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Xright ) {

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int index = out->gIndex();

   int dimLD = bk_down->gCurrentDim( index, NL, TwoSL, IL );
   int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );
   int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 ) {
      dcomplex * BlockX = Xright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
      dcomplex * BlockS = in->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

      char cotrans = 'C';
      char notrans = 'N';
      dcomplex one = 1.0;
      dcomplex fac = 1.0;

      zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &fac, BlockS, &dimLD, BlockX, &dimRU, &one, memHeff, &dimLD );
   }
}

void CheMPS2::CHeffNS_1S::addDiagram1C( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, dcomplex Helem ) {

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int theindex = out->gIndex();

   const int N = NR - NL;

   const int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   const int dimRD = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 && N == 2 ) {
      int inc          = 1;
      int dim          = dimLD * dimRD;
      dcomplex * block = in->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
      zaxpy_( &dim, &Helem, block, &inc, memHeff, &inc );
   }
}