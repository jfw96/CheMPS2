
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

CheMPS2::CHeffNS_1S::CHeffNS_1S( const SyBookkeeper * bk_upIn, const SyBookkeeper * bk_downIn, const Problem * ProbIn )
    : bk_up( bk_upIn ), bk_down( bk_downIn ), Prob( ProbIn ) {}

CheMPS2::CHeffNS_1S::~CHeffNS_1S() {}

void CheMPS2::CHeffNS_1S::Apply( CTensorT * in, CTensorT * out,
                                 CTensorL *** Ltensors, CTensorLT *** LtensorsT,
                                 CTensorOperator **** Atensors, CTensorOperator **** AtensorsT,
                                 CTensorOperator **** Btensors, CTensorOperator **** BtensorsT,
                                 CTensorOperator **** Ctensors, CTensorOperator **** CtensorsT,
                                 CTensorOperator **** Dtensors, CTensorOperator **** DtensorsT,
                                 CTensorS0 **** S0tensors, CTensorS0T **** S0Ttensors,
                                 CTensorS1 **** S1tensors, CTensorS1T **** S1Ttensors,
                                 CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors,
                                 CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors,
                                 CTensorQ *** Qtensors, CTensorQT *** QtensorsT,
                                 CTensorX ** Xtensors, CTensorO ** Otensors, bool moveRight ) {
   out->Clear();

   const int index    = in->gIndex();
   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == Prob->gL() - 1 ) ? true : false;

   const int DIM_up   = std::max( bk_up->gMaxDimAtBound( index ), bk_up->gMaxDimAtBound( index + 1 ) );
   const int DIM_down = std::max( bk_down->gMaxDimAtBound( index ), bk_down->gMaxDimAtBound( index + 1 ) );

   char cotrans = 'C';
   char notrans = 'N';

   dcomplex * temp  = new dcomplex[ DIM_up * DIM_down ];
   dcomplex * temp2 = new dcomplex[ DIM_up * DIM_down ];

   for ( int ikappa = 0; ikappa < out->gNKappa(); ikappa++ ) {

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
      int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );

      int dimLD = bk_down->gCurrentDim( index, NL, TwoSL, IL );
      int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );

      dcomplex * memLUxRU = new dcomplex[ dimLU * dimRU ];
      dcomplex * memLUxRD = new dcomplex[ dimLU * dimRD ];
      dcomplex * memLDxRU = new dcomplex[ dimLD * dimRU ];
      dcomplex * memLDxRD = new dcomplex[ dimLD * dimRD ];

      for ( int cnt = 0; cnt < dimLD * dimRD; cnt++ ) {
         memLDxRD[ cnt ] = 0.0;
      }
      addDiagram1C( ikappa, memLDxRD, in, out, Prob->gMxElement( index, index, index, index ) );

      if ( !atLeft ) {
         for ( int cnt = 0; cnt < dimLU * dimRD; cnt++ ) {
            memLUxRD[ cnt ] = 0.0;
         }
         /*********************
         *  Diagrams group 1  *
         *********************/
         addDiagram1A( ikappa, memLUxRD, in, out, Xtensors[ index - 1 ] );

         /*********************
         *  Diagrams group 2  *
         *********************/
         addDiagram2b1and2b2( ikappa, memLUxRD, in, out, Atensors[ index - 1 ][ 0 ][ 0 ], AtensorsT[ index - 1 ][ 0 ][ 0 ] );
         addDiagram2b3spin0( ikappa, memLUxRD, in, out, CtensorsT[ index - 1 ][ 0 ][ 0 ] );
         addDiagram2b3spin1( ikappa, memLUxRD, in, out, DtensorsT[ index - 1 ][ 0 ][ 0 ] );

         /*********************
         *  Diagrams group 3  *
         *********************/
         addDiagram3Aand3D( ikappa, memLUxRD, in, out, Qtensors[ index - 1 ][ 0 ], QtensorsT[ index - 1 ][ 0 ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );
      }

      if ( ( !atLeft ) && ( !atRight ) ) {
         for ( int cnt = 0; cnt < dimLU * dimRU; cnt++ ) {
            memLUxRU[ cnt ] = 0.0;
         }

         /*********************
          *  Diagrams group 2  *
          *********************/
         addDiagram2a1spin0( ikappa, memLUxRU, in, out, AtensorsT, S0Ttensors, temp );
         addDiagram2a2spin0( ikappa, memLUxRU, in, out, Atensors, S0tensors, temp );
         addDiagram2a1spin1( ikappa, memLUxRU, in, out, BtensorsT, S1Ttensors, temp );
         addDiagram2a2spin1( ikappa, memLUxRU, in, out, Btensors, S1tensors, temp );
         addDiagram2a3spin0( ikappa, memLUxRU, in, out, Ctensors, CtensorsT, F0tensors, F0Ttensors, temp );
         addDiagram2a3spin1( ikappa, memLUxRU, in, out, Dtensors, DtensorsT, F1tensors, F1Ttensors, temp );

         /*********************
         *  Diagrams group 3  *
         *********************/
         addDiagram3C( ikappa, memLUxRU, in, out, Qtensors[ index - 1 ], QtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         addDiagram3J( ikappa, memLUxRU, in, out, Qtensors[ index ], QtensorsT[ index ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );

         /*********************
         *  Diagrams group 4  *
         *********************/
         addDiagram4B1and4B2spin0( ikappa, memLUxRU, in, out, Atensors[ index - 1 ], AtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         addDiagram4B1and4B2spin1( ikappa, memLUxRU, in, out, Btensors[ index - 1 ], BtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         addDiagram4B3and4B4spin0( ikappa, memLUxRU, in, out, Ctensors[ index - 1 ], CtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         addDiagram4B3and4B4spin1( ikappa, memLUxRU, in, out, Dtensors[ index - 1 ], DtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         addDiagram4E( ikappa, memLUxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp, temp2 );
         addDiagram4L1and4L2spin0( ikappa, memLUxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Atensors[ index ], AtensorsT[ index ], temp );
         addDiagram4L1and4L2spin1( ikappa, memLUxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Btensors[ index ], BtensorsT[ index ], temp );
         addDiagram4L3and4L4spin0( ikappa, memLUxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ctensors[ index ], CtensorsT[ index ], temp );
         addDiagram4L3and4L4spin1( ikappa, memLUxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Dtensors[ index ], DtensorsT[ index ], temp );
      }

      if ( !atRight && dimLD > 0 ) {
         for ( int cnt = 0; cnt < dimLD * dimRU; cnt++ ) {
            memLDxRU[ cnt ] = 0.0;
         }

         /*********************
         *  Diagrams group 1  *
         *********************/
         addDiagram1B( ikappa, memLDxRU, in, out, Xtensors[ index ] );

         /*********************
         *  Diagrams group 2  *
         *********************/
         addDiagram2e1and2e2( ikappa, memLDxRU, in, out, Atensors[ index ][ 0 ][ 0 ], AtensorsT[ index ][ 0 ][ 0 ] );
         addDiagram2e3spin0( ikappa, memLDxRU, in, out, CtensorsT[ index ][ 0 ][ 0 ] );
         addDiagram2e3spin1( ikappa, memLDxRU, in, out, DtensorsT[ index ][ 0 ][ 0 ] );

         /*********************
         *  Diagrams group 3  *
         *********************/
         addDiagram3Kand3F( ikappa, memLDxRU, in, out, Qtensors[ index ][ 0 ], QtensorsT[ index ][ 0 ], Ltensors[ index ], LtensorsT[ index ], temp );
      }
      int dimLUxRU = dimLU * dimRU;
      int dimLDxRD = dimLD * dimRD;

      dcomplex * BlockOut = out->gStorage() + out->gKappa2index( ikappa );

      dcomplex one = 1.0;
      int inc      = 1;
      zaxpy_( &dimLUxRU, &one, memLUxRU, &inc, BlockOut, &inc );

      if ( !atRight && dimRD > 0 ) {
         dcomplex * BlockOT = Otensors[ index ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &one, memLUxRD, &dimLU, BlockOT, &dimRU, &one, BlockOut, &dimLU );
      } else if ( dimRD > 0 ) {
         int dimLUxRD = dimLU * dimRD;
         zaxpy_( &dimLUxRD, &one, memLUxRD, &inc, BlockOut, &inc );
      }

      if ( !atLeft && dimLD > 0 ) {
         dcomplex * BlockO = Otensors[ index - 1 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
         zgemm_( &notrans, &notrans, &dimLU, &dimRU, &dimLD, &one, BlockO, &dimLU, memLDxRU, &dimLD, &one, BlockOut, &dimLU );

      } else if ( dimLD > 0 ) {
         int dimLDxRU = dimLD * dimRU;
         zaxpy_( &dimLDxRU, &one, memLDxRU, &inc, BlockOut, &inc );
      }

      if ( atLeft && dimRD > 0 ) {
         dcomplex * BlockOT = Otensors[ index ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

         zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &one, memLDxRD, &dimLD, BlockOT, &dimRU, &one, BlockOut, &dimLD );
      } else if ( atRight && dimLD > 0 ) {
         dcomplex * BlockO = Otensors[ index - 1 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

         zgemm_( &notrans, &notrans, &dimLU, &dimRU, &dimLD, &one, BlockO, &dimLU, memLDxRD, &dimLD, &one, BlockOut, &dimLU );
      } else if ( ( !atLeft ) && ( !atRight ) && dimLD > 0 && dimRD > 0 ) {
         dcomplex * BlockOT = Otensors[ index ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         dcomplex * BlockO  = Otensors[ index - 1 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

         dcomplex * temp3 = new dcomplex[ dimLD * dimRU ];
         dcomplex zero    = 0.0;
         zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &one, memLDxRD, &dimLD, BlockOT, &dimRU, &zero, temp3, &dimLD );

         zgemm_( &notrans, &notrans, &dimLU, &dimRU, &dimLD, &one, BlockO, &dimLU, temp3, &dimLD, &one, BlockOut, &dimLU );

         delete[] temp3;
      }
      delete[] memLUxRU;
      delete[] memLUxRD;
      delete[] memLDxRU;
      delete[] memLDxRD;
   }

   delete[] temp;
   delete[] temp2;
}