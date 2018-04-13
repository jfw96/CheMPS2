
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CSubSpaceExpander.h"
#include "Davidson.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Wigner.h"

void CheMPS2::CSubSpaceExpander::addDiagram0ALeft( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, dcomplex off_set ) {
   const int index = in->gIndex();

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int dimLD = sseBKDown->gCurrentDim( index, NL, TwoSL, IL );
   const int dimRD = initBKDown->gCurrentDim( index + 1, NR, TwoSR, IR );
   
   int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 && memSkappa != -1 ) {
      int inc       = 1;
      int dim       = dimLD * dimRD;
      dcomplex fac  = off_set;

      dcomplex * block = in->gStorage() + in->gKappa2index( memSkappa );
      zaxpy_( &dim, &fac, block, &inc, memHeff, &inc );
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram1A( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorX * Xleft ) {
   const int index = in->gIndex();

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( index );

   if ( movingRight ) {
      int dimLU     = initBKUp->gCurrentDim( index, NL, TwoSL, IL );
      int dimLD     = initBKDown->gCurrentDim( index, NL, TwoSL, IL );
      int dimRD     = sseBKDown->gCurrentDim( index + 1, NR, TwoSR, IR );
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( dimLU > 0 && dimLD > 0 && dimRD > 0 && memSkappa != -1 ) {
         char cotrans = 'C';
         char notrans = 'N';
         dcomplex one = 1.0;

         dcomplex * BlockX = Xleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
         dcomplex * block  = in->gStorage() + in->gKappa2index( memSkappa );
         zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &one, BlockX, &dimLU, block, &dimLD, &one, memHeff, &dimLU );
      }

   } else {
      const int dimLD = sseBKDown->gCurrentDim( site, NL, TwoSL, IL );
      const int dimRD = initBKDown->gCurrentDim( site + 1, NR, TwoSR, IR );
      int memSkappa   = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( dimLD > 0 && dimRD > 0 && memSkappa != -1 ) {
         int inc = 1;
         int dim = dimLD * dimRD;
         dcomplex one     = 1.0;

         dcomplex * block = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &one, block, &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram1B( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Xright ) {
   const int index = out->gIndex();

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( index );

   if ( movingRight ) {
      const int dimLD = initBKDown->gCurrentDim( site, NL, TwoSL, IL );
      const int dimRD = sseBKDown->gCurrentDim( site + 1, NR, TwoSR, IR );
      int memSkappa   = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( dimLD > 0 && dimRD > 0 && memSkappa != -1 ) {
         int inc      = 1;
         int dim      = dimLD * dimRD;
         dcomplex one = 1.0;

         dcomplex * block = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &one, block, &inc, memHeff, &inc );
      }
   } else {
      int dimLD     = sseBKDown->gCurrentDim( index, NL, TwoSL, IL );
      int dimRD     = initBKDown->gCurrentDim( index + 1, NR, TwoSR, IR );
      int dimRU     = initBKUp->gCurrentDim( index + 1, NR, TwoSR, IR );
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( dimLD > 0 && dimRD > 0 && dimRU > 0 && memSkappa != -1 ) {
         char cotrans = 'C';
         char notrans = 'N';
         dcomplex one = 1.0;
         dcomplex fac = 1.0;

         dcomplex * BlockX = Xright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         dcomplex * Block  = in->gStorage() + in->gKappa2index( memSkappa );
         zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &fac, Block, &dimLD, BlockX, &dimRU, &one, memHeff, &dimLD );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram1C( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, dcomplex Helem ) {

   const int NL    = out->gNL( ikappa );
   const int TwoSL = out->gTwoSL( ikappa );
   const int IL    = out->gIL( ikappa );

   const int NR    = out->gNR( ikappa );
   const int TwoSR = out->gTwoSR( ikappa );
   const int IR    = out->gIR( ikappa );

   const int theindex = out->gIndex();

   const int N = NR - NL;

   const int dimLD = movingRight ? initBKDown->gCurrentDim( site, NL, TwoSL, IL ) : sseBKDown->gCurrentDim( site, NL, TwoSL, IL );
   const int dimRD = movingRight ? sseBKDown->gCurrentDim( site + 1, NR, TwoSR, IR ) : initBKDown->gCurrentDim( site + 1, NR, TwoSR, IR );
   int memSkappa   = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

   if ( dimLD > 0 && dimRD > 0 && N == 2 && memSkappa != -1 ) {
      int inc = 1;
      int dim = dimLD * dimRD;

      dcomplex * block = in->gStorage() + in->gKappa2index( memSkappa );
      zaxpy_( &dim, &Helem, block, &inc, memHeff, &inc );
   }
}