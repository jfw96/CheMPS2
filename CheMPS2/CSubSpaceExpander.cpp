
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

CheMPS2::CSubSpaceExpander::CSubSpaceExpander( const int siteIn, const bool movingRightIn, const SyBookkeeper * initBKUpIn, const SyBookkeeper * initBKDownIn, const SyBookkeeper * sseBKDownIn, const Problem * ProbIn )
    : site( siteIn ), movingRight( movingRightIn ), initBKUp( initBKUpIn ), initBKDown( initBKDownIn ), sseBKDown( sseBKDownIn ), prob( ProbIn ) {
   assert( siteIn >= 0 );
}

CheMPS2::CSubSpaceExpander::~CSubSpaceExpander() {}

void CheMPS2::CSubSpaceExpander::Expand( double * noise,
                                         CTensorT * in, CTensorT * out,
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
                                         CTensorX ** Xtensors, CTensorO ** Otensors ) {
   out->Clear();

   if ( movingRight ) {
      ApplyRight( in, out, Ltensors, LtensorsT, Atensors, AtensorsT, Btensors, BtensorsT,
                  Ctensors, CtensorsT, Dtensors, DtensorsT, S0tensors, S0Ttensors, S1tensors,
                  S1Ttensors, F0tensors, F0Ttensors, F1tensors, F1Ttensors, Qtensors,
                  QtensorsT, Xtensors, Otensors );
   } else {
      ApplyLeft( in, out, Ltensors, LtensorsT, Atensors, AtensorsT, Btensors, BtensorsT,
                 Ctensors, CtensorsT, Dtensors, DtensorsT, S0tensors, S0Ttensors, S1tensors,
                 S1Ttensors, F0tensors, F0Ttensors, F1tensors, F1Ttensors, Qtensors,
                 QtensorsT, Xtensors, Otensors );
   }

   out->scale( *noise );
}

void CheMPS2::CSubSpaceExpander::ApplyRight( CTensorT * in, CTensorT * out,
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
                                             CTensorX ** Xtensors, CTensorO ** Otensors ) {
   const int index    = site;
   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == prob->gL() - 1 ) ? true : false;

   const int DIM_up   = std::max( initBKUp->gMaxDimAtBound( index ), initBKUp->gMaxDimAtBound( index + 1 ) );
   const int DIM_down = std::max( std::max( initBKDown->gMaxDimAtBound( index ), initBKDown->gMaxDimAtBound( index + 1 ) ), std::max( sseBKDown->gMaxDimAtBound( index ), sseBKDown->gMaxDimAtBound( index + 1 ) ) );

   char cotrans = 'C';
   char notrans = 'N';

   // // PARALLEL
   // #pragma omp parallel
   {
      dcomplex * temp  = new dcomplex[ DIM_up * DIM_down ];
      dcomplex * temp2 = new dcomplex[ DIM_up * DIM_down ];

      // #pragma omp for schedule( dynamic )
      for ( int ikappa = 0; ikappa < out->gNKappa(); ikappa++ ) {

         const int NL    = out->gNL( ikappa );
         const int TwoSL = out->gTwoSL( ikappa );
         const int IL    = out->gIL( ikappa );

         const int NR    = out->gNR( ikappa );
         const int TwoSR = out->gTwoSR( ikappa );
         const int IR    = out->gIR( ikappa );

         const int N    = NR - NL;
         const int TwoS = ( N == 1 ) ? 1 : 0;
         const int I    = initBKUp->gIrrep( index );

         int dimLU = initBKUp->gCurrentDim( index, NL, TwoSL, IL );
         int dimLD = initBKDown->gCurrentDim( index, NL, TwoSL, IL );
         int dimRD = sseBKDown->gCurrentDim( index + 1, NR, TwoSR, IR );

         dcomplex * memLUxRD = new dcomplex[ dimLU * dimRD ];
         dcomplex * memLDxRD = new dcomplex[ dimLD * dimRD ];
         for ( int cnt = 0; cnt < dimLD * dimRD; cnt++ ) {
            memLDxRD[ cnt ] = 0.0;
         }
         for ( int cnt = 0; cnt < dimLU * dimRD; cnt++ ) {
            memLUxRD[ cnt ] = 0.0;
         }

         // addDiagram0ALeft( ikappa, memLDxRD, in, out, prob->gEconst() );
         // addDiagram1C( ikappa, memLDxRD, in, out, prob->gMxElement( index, index, index, index ) );

         // if ( !atLeft ) {
         //    /*********************
         //    *  Diagrams group 1  *
         //    *********************/
         //    addDiagram1A( ikappa, memLDxRD, in, out, Xtensors[ index - 1 ] );

         //    /*********************
         //    *  Diagrams group 2  *
         //    *********************/
         //    addDiagram2b1and2b2Left( ikappa, memLDxRD, in, out, Atensors[ index - 1 ][ 0 ][ 0 ], AtensorsT[ index - 1 ][ 0 ][ 0 ] );
         //    addDiagram2b3spin0Left( ikappa, memLDxRD, in, out, CtensorsT[ index - 1 ][ 0 ][ 0 ] );
         //    addDiagram2b3spin1Left( ikappa, memLDxRD, in, out, DtensorsT[ index - 1 ][ 0 ][ 0 ] );

         //    /*********************
         //    *  Diagrams group 3  *
         //    *********************/
         //    addDiagram3Aand3DLeft( ikappa, memLDxRD, in, out, Qtensors[ index - 1 ][ 0 ], QtensorsT[ index - 1 ][ 0 ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );
         // }

         // if ( ( !atLeft ) && ( !atRight ) ) {

         //    /*********************
         //    *  Diagrams group 2  *
         //    *********************/
         //    addDiagram2a1spin0Left( ikappa, memLDxRU, in, out, AtensorsT, S0Ttensors, temp );
         //    addDiagram2a2spin0Left( ikappa, memLDxRU, in, out, Atensors, S0tensors, temp );
         //    addDiagram2a1spin1Left( ikappa, memLDxRU, in, out, BtensorsT, S1Ttensors, temp );
         //    addDiagram2a2spin1Left( ikappa, memLDxRU, in, out, Btensors, S1tensors, temp );
         //    addDiagram2a3spin0Left( ikappa, memLDxRU, in, out, Ctensors, CtensorsT, F0tensors, F0Ttensors, temp );
         //    addDiagram2a3spin1Left( ikappa, memLDxRU, in, out, Dtensors, DtensorsT, F1tensors, F1Ttensors, temp );

         //    /*********************
         //    *  Diagrams group 3  *
         //    *********************/
         //    addDiagram3CLeft( ikappa, memLDxRU, in, out, Qtensors[ index - 1 ], QtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         //    addDiagram3JLeft( ikappa, memLDxRU, in, out, Qtensors[ index ], QtensorsT[ index ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );

         //    /*********************
         //    *  Diagrams group 4  *
         //    *********************/
         //    addDiagram4B1and4B2spin0Left( ikappa, memLDxRU, in, out, Atensors[ index - 1 ], AtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         //    addDiagram4B1and4B2spin1Left( ikappa, memLDxRU, in, out, Btensors[ index - 1 ], BtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         //    addDiagram4B3and4B4spin0Left( ikappa, memLDxRU, in, out, Ctensors[ index - 1 ], CtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         //    addDiagram4B3and4B4spin1Left( ikappa, memLDxRU, in, out, Dtensors[ index - 1 ], DtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
         //    addDiagram4ELeft( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp, temp2 );
         //    addDiagram4L1and4L2spin0Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Atensors[ index ], AtensorsT[ index ], temp );
         //    addDiagram4L3and4L4spin0Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ctensors[ index ], CtensorsT[ index ], temp );
         //    addDiagram4L1and4L2spin1Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Btensors[ index ], BtensorsT[ index ], temp );
         //    addDiagram4L3and4L4spin1Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Dtensors[ index ], DtensorsT[ index ], temp );
         // }

         // if ( !atRight && dimLD > 0 ) {

         //    /*********************
         //    *  Diagrams group 1  *
         //    *********************/
         //    addDiagram1B( ikappa, memLDxRU, in, out, Xtensors[ index ] );

         //    /*********************
         //    *  Diagrams group 2  *
         //    *********************/
         //    addDiagram2e1and2e2Left( ikappa, memLDxRU, in, out, Atensors[ index ][ 0 ][ 0 ], AtensorsT[ index ][ 0 ][ 0 ] );
         //    addDiagram2e3spin0Left( ikappa, memLDxRU, in, out, CtensorsT[ index ][ 0 ][ 0 ] );
         //    addDiagram2e3spin1Left( ikappa, memLDxRU, in, out, DtensorsT[ index ][ 0 ][ 0 ] );

         //    /*********************
         //    *  Diagrams group 3  *
         //    *********************/
         //    addDiagram3Kand3FLeft( ikappa, memLDxRU, in, out, Qtensors[ index ][ 0 ], QtensorsT[ index ][ 0 ], Ltensors[ index ], LtensorsT[ index ], temp );
         // }

         int dimLUxRD = dimLU * dimRD;
         int dimLDxRD = dimLD * dimRD;

         dcomplex * BlockOut = out->gStorage() + out->gKappa2index( ikappa );

         dcomplex one = 1.0;
         int inc      = 1;
         zaxpy_( &dimLUxRD, &one, memLUxRD, &inc, BlockOut, &inc );

         if ( atLeft ) {
            int dimLDxRD = dimLD * dimRD;
            zaxpy_( &dimLDxRD, &one, memLDxRD, &inc, BlockOut, &inc );
         } else if (dimLU > 0 && dimRD > 0) {
            dcomplex * BlockOT = Otensors[ index ]->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &one, memLDxRD, &dimLU, BlockOT, &dimLD, &one, BlockOut, &dimLU );
         }

         delete[] memLUxRD;
         delete[] memLDxRD;
      }

      delete[] temp;
      delete[] temp2;
   }

   std::cout << "hi right" << std::endl;
   abort();
}

void CheMPS2::CSubSpaceExpander::ApplyLeft( CTensorT * in, CTensorT * out,
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
                                            CTensorX ** Xtensors, CTensorO ** Otensors ) {

   const int index    = site;
   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == prob->gL() - 1 ) ? true : false;

   const int DIM_up   = std::max( initBKUp->gMaxDimAtBound( index ), initBKUp->gMaxDimAtBound( index + 1 ) );
   const int DIM_down = std::max( std::max( initBKDown->gMaxDimAtBound( index ), initBKDown->gMaxDimAtBound( index + 1 ) ), std::max( sseBKDown->gMaxDimAtBound( index ), sseBKDown->gMaxDimAtBound( index + 1 ) ) );

   char cotrans = 'C';
   char notrans = 'N';

   // // PARALLEL
   // #pragma omp parallel
   {
      dcomplex * temp  = new dcomplex[ DIM_up * DIM_down ];
      dcomplex * temp2 = new dcomplex[ DIM_up * DIM_down ];

      // #pragma omp for schedule( dynamic )
      for ( int ikappa = 0; ikappa < out->gNKappa(); ikappa++ ) {

         const int NL    = out->gNL( ikappa );
         const int TwoSL = out->gTwoSL( ikappa );
         const int IL    = out->gIL( ikappa );

         const int NR    = out->gNR( ikappa );
         const int TwoSR = out->gTwoSR( ikappa );
         const int IR    = out->gIR( ikappa );

         const int N    = NR - NL;
         const int TwoS = ( N == 1 ) ? 1 : 0;
         const int I    = initBKUp->gIrrep( index );

         int dimRU = initBKUp->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimRD = initBKDown->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimLD = sseBKDown->gCurrentDim( index, NL, TwoSL, IL );

         dcomplex * memLDxRU = new dcomplex[ dimLD * dimRU ];
         dcomplex * memLDxRD = new dcomplex[ dimLD * dimRD ];
         for ( int cnt = 0; cnt < dimLD * dimRD; cnt++ ) {
            memLDxRD[ cnt ] = 0.0;
         }
         for ( int cnt = 0; cnt < dimLD * dimRU; cnt++ ) {
            memLDxRU[ cnt ] = 0.0;
         }

         addDiagram0ALeft( ikappa, memLDxRD, in, out, prob->gEconst() );
         addDiagram1C( ikappa, memLDxRD, in, out, prob->gMxElement( index, index, index, index ) );

         if ( !atLeft ) {
            /*********************
            *  Diagrams group 1  *
            *********************/
            addDiagram1A( ikappa, memLDxRD, in, out, Xtensors[ index - 1 ] );

            /*********************
            *  Diagrams group 2  *
            *********************/
            addDiagram2b1and2b2Left( ikappa, memLDxRD, in, out, Atensors[ index - 1 ][ 0 ][ 0 ], AtensorsT[ index - 1 ][ 0 ][ 0 ] );
            addDiagram2b3spin0Left( ikappa, memLDxRD, in, out, CtensorsT[ index - 1 ][ 0 ][ 0 ] );
            addDiagram2b3spin1Left( ikappa, memLDxRD, in, out, DtensorsT[ index - 1 ][ 0 ][ 0 ] );

            /*********************
            *  Diagrams group 3  *
            *********************/
            addDiagram3Aand3DLeft( ikappa, memLDxRD, in, out, Qtensors[ index - 1 ][ 0 ], QtensorsT[ index - 1 ][ 0 ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );
         }

         if ( ( !atLeft ) && ( !atRight ) ) {

            /*********************
            *  Diagrams group 2  *
            *********************/
            addDiagram2a1spin0Left( ikappa, memLDxRU, in, out, AtensorsT, S0Ttensors, temp );
            addDiagram2a2spin0Left( ikappa, memLDxRU, in, out, Atensors, S0tensors, temp );
            addDiagram2a1spin1Left( ikappa, memLDxRU, in, out, BtensorsT, S1Ttensors, temp );
            addDiagram2a2spin1Left( ikappa, memLDxRU, in, out, Btensors, S1tensors, temp );
            addDiagram2a3spin0Left( ikappa, memLDxRU, in, out, Ctensors, CtensorsT, F0tensors, F0Ttensors, temp );
            addDiagram2a3spin1Left( ikappa, memLDxRU, in, out, Dtensors, DtensorsT, F1tensors, F1Ttensors, temp );

            /*********************
            *  Diagrams group 3  *
            *********************/
            addDiagram3CLeft( ikappa, memLDxRU, in, out, Qtensors[ index - 1 ], QtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram3JLeft( ikappa, memLDxRU, in, out, Qtensors[ index ], QtensorsT[ index ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );

            /*********************
            *  Diagrams group 4  *
            *********************/
            addDiagram4B1and4B2spin0Left( ikappa, memLDxRU, in, out, Atensors[ index - 1 ], AtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4B1and4B2spin1Left( ikappa, memLDxRU, in, out, Btensors[ index - 1 ], BtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4B3and4B4spin0Left( ikappa, memLDxRU, in, out, Ctensors[ index - 1 ], CtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4B3and4B4spin1Left( ikappa, memLDxRU, in, out, Dtensors[ index - 1 ], DtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4ELeft( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp, temp2 );
            addDiagram4L1and4L2spin0Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Atensors[ index ], AtensorsT[ index ], temp );
            addDiagram4L3and4L4spin0Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ctensors[ index ], CtensorsT[ index ], temp );
            addDiagram4L1and4L2spin1Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Btensors[ index ], BtensorsT[ index ], temp );
            addDiagram4L3and4L4spin1Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Dtensors[ index ], DtensorsT[ index ], temp );
         }

         if ( !atRight && dimLD > 0 ) {

            /*********************
            *  Diagrams group 1  *
            *********************/
            addDiagram1B( ikappa, memLDxRU, in, out, Xtensors[ index ] );

            /*********************
            *  Diagrams group 2  *
            *********************/
            addDiagram2e1and2e2Left( ikappa, memLDxRU, in, out, Atensors[ index ][ 0 ][ 0 ], AtensorsT[ index ][ 0 ][ 0 ] );
            addDiagram2e3spin0Left( ikappa, memLDxRU, in, out, CtensorsT[ index ][ 0 ][ 0 ] );
            addDiagram2e3spin1Left( ikappa, memLDxRU, in, out, DtensorsT[ index ][ 0 ][ 0 ] );

            /*********************
            *  Diagrams group 3  *
            *********************/
            addDiagram3Kand3FLeft( ikappa, memLDxRU, in, out, Qtensors[ index ][ 0 ], QtensorsT[ index ][ 0 ], Ltensors[ index ], LtensorsT[ index ], temp );
         }

         int dimLDxRU = dimLD * dimRU;
         int dimLDxRD = dimLD * dimRD;

         dcomplex * BlockOut = out->gStorage() + out->gKappa2index( ikappa );

         dcomplex one = 1.0;
         int inc      = 1;
         zaxpy_( &dimLDxRU, &one, memLDxRU, &inc, BlockOut, &inc );

         if ( atRight ) {
            int dimLDxRD = dimLD * dimRD;
            zaxpy_( &dimLDxRD, &one, memLDxRD, &inc, BlockOut, &inc );
         } else if (dimLD > 0 && dimRU > 0) {
            dcomplex * BlockOT = Otensors[ index ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
            zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &one, memLDxRD, &dimLD, BlockOT, &dimRU, &one, BlockOut, &dimLD );
         }

         delete[] memLDxRU;
         delete[] memLDxRD;
      }

      delete[] temp;
      delete[] temp2;
   }
}

void CheMPS2::CSubSpaceExpander::AddNonExpandedToExpanded( CTensorT * expanded, CTensorT * nonExpanded ) {
   assert( expanded->gIndex() == nonExpanded->gIndex() );

   const int index = expanded->gIndex();

   for ( int ikappa = 0; ikappa < expanded->gNKappa(); ikappa++ ) {

      const int NL    = expanded->gNL( ikappa );
      const int TwoSL = expanded->gTwoSL( ikappa );
      const int IL    = expanded->gIL( ikappa );

      const int NR    = expanded->gNR( ikappa );
      const int TwoSR = expanded->gTwoSR( ikappa );
      const int IR    = expanded->gIR( ikappa );

      const int N    = NR - NL;
      const int TwoS = ( N == 1 ) ? 1 : 0;
      const int I    = initBKUp->gIrrep( index );

      int memNonExpKappa = nonExpanded->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );
      if ( memNonExpKappa != -1 ) {

         int dimLEXP = expanded->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimREXP = expanded->gBK()->gCurrentDim( index + 1, NR, TwoSR, IR );

         int dimLNON = nonExpanded->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimRNON = nonExpanded->gBK()->gCurrentDim( index + 1, NR, TwoSR, IR );

         // If nonExpanded has the block it has to have the same size
         assert( dimLEXP == dimLNON );
         assert( dimREXP == dimRNON );

         int dim      = dimLEXP * dimREXP;
         int inc      = 1;
         dcomplex one = 1.0;

         dcomplex * BlockExp = expanded->gStorage() + expanded->gKappa2index( ikappa );
         dcomplex * BlockNon = nonExpanded->gStorage() + nonExpanded->gKappa2index( memNonExpKappa );

         zaxpy_( &dim, &one, BlockNon, &inc, BlockExp, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::decomposeMovingLeft( bool change, int virtualdimensionD, double cut_off,
                                                      CTensorT * expandedLeft, SyBookkeeper * expandedLeftBK,
                                                      CTensorT * expandedRight, SyBookkeeper * expandedRightBK,
                                                      CTensorT * newLeft, SyBookkeeper * newLeftBK,
                                                      CTensorT * newRight, SyBookkeeper * newRightBK ) {
   assert( expandedLeftBK == expandedRightBK );
   assert( newLeftBK == newRightBK );
   int index = site;

   int nMiddleSectors = 0;
   for ( int NM = expandedRightBK->gNmin( index ); NM <= expandedRightBK->gNmax( index ); NM++ ) {
      for ( int TwoSM = expandedRightBK->gTwoSmin( index, NM ); TwoSM <= expandedRightBK->gTwoSmax( index, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < expandedRightBK->getNumberOfIrreps(); IM++ ) {
            int dimM = expandedRightBK->gCurrentDim( index, NM, TwoSM, IM );
            if ( dimM > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NM == expandedRight->gNL( ikappa ) ) && ( TwoSM == expandedRight->gTwoSL( ikappa ) ) && ( IM == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  nMiddleSectors++;
               }
            }
         }
      }
   }

   int * SplitSectNM    = new int[ nMiddleSectors ];
   int * SplitSectTwoJM = new int[ nMiddleSectors ];
   int * SplitSectIM    = new int[ nMiddleSectors ];
   int * DimLs          = new int[ nMiddleSectors ];
   int * DimMs          = new int[ nMiddleSectors ];
   int * DimRs          = new int[ nMiddleSectors ];

   nMiddleSectors = 0;
   for ( int NM = expandedRightBK->gNmin( index ); NM <= expandedRightBK->gNmax( index ); NM++ ) {
      for ( int TwoSM = expandedRightBK->gTwoSmin( index, NM ); TwoSM <= expandedRightBK->gTwoSmax( index, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < expandedRightBK->getNumberOfIrreps(); IM++ ) {
            int dimM = expandedRightBK->gCurrentDim( index, NM, TwoSM, IM );
            if ( dimM > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NM == expandedRight->gNL( ikappa ) ) && ( TwoSM == expandedRight->gTwoSL( ikappa ) ) && ( IM == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  SplitSectNM[ nMiddleSectors ]    = NM;
                  SplitSectTwoJM[ nMiddleSectors ] = TwoSM;
                  SplitSectIM[ nMiddleSectors ]    = IM;
                  DimLs[ nMiddleSectors ]          = dimM;
                  DimMs[ nMiddleSectors ]          = std::min( dimM, dimRtotal );
                  DimRs[ nMiddleSectors ]          = dimRtotal;
                  nMiddleSectors++;
               }
            }
         }
      }
   }

   double ** Lambdas = NULL;
   dcomplex ** Us    = NULL;
   dcomplex ** VTs   = NULL;

   Lambdas = new double *[ nMiddleSectors ];
   Us      = new dcomplex *[ nMiddleSectors ];
   VTs     = new dcomplex *[ nMiddleSectors ];

   for ( int iMiddleSector = 0; iMiddleSector < nMiddleSectors; iMiddleSector++ ) {

      Lambdas[ iMiddleSector ] = new double[ DimMs[ iMiddleSector ] ];
      Us[ iMiddleSector ]      = new dcomplex[ DimLs[ iMiddleSector ] * DimMs[ iMiddleSector ] ];
      VTs[ iMiddleSector ]     = new dcomplex[ DimMs[ iMiddleSector ] * DimRs[ iMiddleSector ] ];

      // Copy the relevant parts from storage to mem & multiply with factor !!
      dcomplex * mem = new dcomplex[ DimRs[ iMiddleSector ] * DimLs[ iMiddleSector ] ];
      int dimRtotal2 = 0;
      for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
         if ( ( SplitSectNM[ iMiddleSector ] == expandedRight->gNL( ikappa ) ) && ( SplitSectTwoJM[ iMiddleSector ] == expandedRight->gTwoSL( ikappa ) ) && ( SplitSectIM[ iMiddleSector ] == expandedRight->gIL( ikappa ) ) ) {
            int dimR = expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
            if ( dimR > 0 ) {
               double factor = sqrt( ( expandedRight->gTwoSR( ikappa ) + 1.0 ) / ( SplitSectTwoJM[ iMiddleSector ] + 1.0 ) );
               for ( int l = 0; l < DimLs[ iMiddleSector ]; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     dcomplex * storage                                     = expandedRight->gStorage() + expandedRight->gKappa2index( ikappa );
                     mem[ l + DimLs[ iMiddleSector ] * ( dimRtotal2 + r ) ] = factor * storage[ l + DimLs[ iMiddleSector ] * r ];
                  }
               }
               dimRtotal2 += dimR;
            }
         }
      }

      // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
      // central symmetry
      char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
      int lwork       = DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + 3 * DimMs[ iMiddleSector ];
      int lrwork      = std::max( 5 * DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + 5 * DimMs[ iMiddleSector ], 2 * std::max( DimLs[ iMiddleSector ], DimRs[ iMiddleSector ] ) * DimMs[ iMiddleSector ] + 2 * DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + DimMs[ iMiddleSector ] );
      dcomplex * work = new dcomplex[ lwork ];
      double * rwork  = new double[ lrwork ];
      int * iwork     = new int[ 8 * DimMs[ iMiddleSector ] ];
      int info;

      // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
      zgesdd_( &jobz, DimLs + iMiddleSector, DimRs + iMiddleSector, mem, DimLs + iMiddleSector,
               Lambdas[ iMiddleSector ], Us[ iMiddleSector ], DimLs + iMiddleSector, VTs[ iMiddleSector ],
               DimMs + iMiddleSector, work, &lwork, rwork, iwork, &info );

      delete[] work;
      delete[] rwork;
      delete[] iwork;
      delete[] mem;
   }

   double discardedWeight = 0.0; // Only if change==true; will the discardedWeight be meaningful and different from zero.
   int updateSectors      = 0;
   int * NewDims          = NULL;

   // If change: determine new virtual dimensions.
   if ( change ) {
      NewDims = new int[ nMiddleSectors ];

      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         NewDims[ iSector ] = DimMs[ iSector ];
         totalDimSVD += NewDims[ iSector ];
      }

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         if ( NewDims[ iSector ] > 0 ) {
            dcopy_( NewDims + iSector, Lambdas[ iSector ], &inc, values + totalDimSVD, &inc );
            totalDimSVD += NewDims[ iSector ];
         }
      }

      // Sort them in decreasing order
      char ID = 'D';
      int info;
      dlasrt_( &ID, &totalDimSVD, values, &info ); // Quicksort

      int maxD = 0;
      while ( maxD < totalDimSVD && maxD < virtualdimensionD && cut_off < values[ maxD ] ) {
         maxD++;
      }

      // int maxD = virtualdimensionD;
      // If larger then the required virtualdimensionD, new virtual dimensions
      // will be set in NewDims.
      if ( totalDimSVD > maxD ) {

         // The D+1'th value becomes the lower bound Schmidt value. Every value
         // smaller than or equal to the D+1'th value is thrown out (hence Dactual // <= Ddesired).
         const double lowerBound = values[ maxD ];
         for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
            for ( int cnt = 0; cnt < NewDims[ iSector ]; cnt++ ) {
               if ( Lambdas[ iSector ][ cnt ] <= lowerBound ) {
                  NewDims[ iSector ] = cnt;
               }
            }
         }

         // Discarded weight
         double totalSum     = 0.0;
         double discardedSum = 0.0;
         for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
            for ( int iLocal = 0; iLocal < DimMs[ iSector ]; iLocal++ ) {
               double temp = ( expandedRight->gTwoSL( iSector ) + 1 ) * Lambdas[ iSector ][ iLocal ] * Lambdas[ iSector ][ iLocal ];
               totalSum += temp;
               if ( Lambdas[ iSector ][ iLocal ] <= lowerBound ) {
                  discardedSum += temp;
               }
            }
         }
         discardedWeight = discardedSum / totalSum;
      }
      // Clean-up
      delete[] values;

      // Check if there is a sector which differs
      updateSectors = 0;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         const int MPSdim = expandedRightBK->gCurrentDim( index, SplitSectNM[ iSector ], SplitSectTwoJM[ iSector ], SplitSectIM[ iSector ] );
         if ( NewDims[ iSector ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }

   if ( updateSectors == 1 ) {
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         newLeftBK->SetDim( index, SplitSectNM[ iSector ], SplitSectTwoJM[ iSector ], SplitSectIM[ iSector ], NewDims[ iSector ] );
      }
      newRight->Reset();
      newLeft->Reset();
   }

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

   newRight->Clear();
   newLeft->Clear();

   // Copy first dimM per central symmetry sector to the relevant parts
   for ( int iCenter = 0; iCenter < nMiddleSectors; iCenter++ ) {

      int dimLtotal2 = 0;
      for ( int ikappa = 0; ikappa < newLeft->gNKappa(); ikappa++ ) {

         const int NL    = newLeft->gNL( ikappa );
         const int TwoSL = newLeft->gTwoSL( ikappa );
         const int IL    = newLeft->gIL( ikappa );

         const int NR    = newLeft->gNR( ikappa );
         const int TwoSR = newLeft->gTwoSR( ikappa );
         const int IR    = newLeft->gIR( ikappa );

         if ( ( SplitSectNM[ iCenter ] == NR ) && ( SplitSectTwoJM[ iCenter ] == TwoSR ) && ( SplitSectIM[ iCenter ] == IR ) ) {
            const int dimL = newLeftBK->gCurrentDim( index - 1, NL, TwoSL, IL );
            const int dimM = expandedLeftBK->gCurrentDim( index, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
            if ( dimL > 0){
               dcomplex * TleftOld = expandedLeft->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
               dcomplex * TleftNew = newLeft->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               const int dimension_limit_right = std::min( dimM, DimMs[ iCenter ] );
               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = 0; r < dimension_limit_right; r++ ) {
                     TleftNew[ l + dimL * r ] = 0.0;
                     for ( int m = 0; m < dimM; m++ ){
                        TleftNew[ l + dimL * r ] += TleftOld[ l + dimL * m ] * Us[ iCenter ][ m + DimMs[ iCenter ] * r ] * Lambdas[ iCenter ][ r ];
                     }
                  }
               }
               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = dimension_limit_right; r < dimM; r++ ) {
                     TleftNew[ l + dimL * r ] = 0.0;
                  }
               }
               dimLtotal2 += dimL;
            }
         }
      }

      // Copy from mem to storage & multiply with factor !!
      int dimRtotal2 = 0;
      for ( int ikappa = 0; ikappa < newRight->gNKappa(); ikappa++ ) {

         const int NL    = newRight->gNL( ikappa );
         const int TwoSL = newRight->gTwoSL( ikappa );
         const int IL    = newRight->gIL( ikappa );

         const int NR    = newRight->gNR( ikappa );
         const int TwoSR = newRight->gTwoSR( ikappa );
         const int IR    = newRight->gIR( ikappa );

         if ( ( SplitSectNM[ iCenter ] == NL ) && ( SplitSectTwoJM[ iCenter ] == TwoSL ) && ( SplitSectIM[ iCenter ] == IL ) ) {
            int dimR = newRightBK->gCurrentDim( index + 1, NR, TwoSR, IR );
            if ( dimR > 0 ) {
               double factor      = sqrt( ( SplitSectTwoJM[ iCenter ] + 1.0 ) / ( TwoSR + 1.0 ) );
               dcomplex * storage = newRight->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               for ( int l = 0; l < DimMs[ iCenter ]; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     storage[ l + DimLs[ iCenter ] * r ] = factor * VTs[ iCenter ][ l + DimMs[ iCenter ] * ( r + dimRtotal2 ) ];
                  }
               }
               dimRtotal2 += dimR;
            }
         }
      }
   }
}