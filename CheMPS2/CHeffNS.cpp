
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CHeffNS.h"
#include "Davidson.h"
#include "Lapack.h"
#include "MPIchemps2.h"

CheMPS2::CHeffNS::CHeffNS( const SyBookkeeper * bk_upIn, const SyBookkeeper * bk_downIn, const Problem * ProbIn )
    : bk_up( bk_upIn ), bk_down( bk_downIn ), Prob( ProbIn ) {}

CheMPS2::CHeffNS::~CHeffNS() {}

void CheMPS2::CHeffNS::makeHeff( dcomplex * memS, dcomplex * memP, CSobject * denS, CSobject * denP,
                                 CTensorL *** Ltensors, CTensorLT *** LtensorsT, CTensorOperator **** Atensors,
                                 CTensorOperator **** AtensorsT, CTensorOperator **** Btensors,
                                 CTensorOperator **** BtensorsT, CTensorOperator **** Ctensors,
                                 CTensorOperator **** CtensorsT, CTensorOperator **** Dtensors,
                                 CTensorOperator **** DtensorsT, CTensorS0 **** S0tensors,
                                 CTensorS0T **** S0tensorsT, CTensorS1 **** S1tensors, CTensorS1T **** S1tensorsT,
                                 CTensorF0 **** F0tensors, CTensorF0T **** F0tensorsT, CTensorF1 **** F1tensors,
                                 CTensorF1T **** F1tensorsT, CTensorQ *** Qtensors, CTensorQT *** QtensorsT,
                                 CTensorX ** Xtensors, CTensorO * OtensorsL, CTensorO * OtensorsR ) {

   const int indexS   = denP->gIndex();
   const bool atLeft  = ( indexS == 0 ) ? true : false;
   const bool atRight = ( indexS == Prob->gL() - 2 ) ? true : false;

   const int DIM_up   = std::max( bk_up->gMaxDimAtBound( indexS ), bk_up->gMaxDimAtBound( indexS + 2 ) );
   const int DIM_down = std::max( bk_down->gMaxDimAtBound( indexS ), bk_down->gMaxDimAtBound( indexS + 2 ) );

   char cotrans = 'C';
   char notrans = 'N';

// PARALLEL
#pragma omp parallel
   {
      dcomplex * temp  = new dcomplex[ DIM_up * DIM_down ];
      dcomplex * temp2 = new dcomplex[ DIM_up * DIM_down ];

#pragma omp for schedule( dynamic )
      for ( int ikappaBIS = 0; ikappaBIS < denP->gNKappa(); ikappaBIS++ ) {
         const int ikappa = denP->gReorder( ikappaBIS );

         const int NL    = denP->gNL( ikappa );
         const int TwoSL = denP->gTwoSL( ikappa );
         const int IL    = denP->gIL( ikappa );

         const int NR    = denP->gNR( ikappa );
         const int TwoSR = denP->gTwoSR( ikappa );
         const int IR    = denP->gIR( ikappa );

         const int N1   = denP->gN1( ikappa );
         const int N2   = denP->gN2( ikappa );
         const int TwoJ = denP->gTwoJ( ikappa );

         const int theindex = denP->gIndex();

         int dimLU = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimRU = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

         int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

         if ( dimLD > 0 && dimRD > 0 ) {

            dcomplex * memLUxRU = new dcomplex[ dimLU * dimRU ];
            dcomplex * memLUxRD = new dcomplex[ dimLU * dimRD ];
            dcomplex * memLDxRU = new dcomplex[ dimLD * dimRU ];
            dcomplex * memLDxRD = new dcomplex[ dimLD * dimRD ];

            for ( int cnt = 0; cnt < dimLU * dimRU; cnt++ ) {
               memLUxRU[ cnt ] = 0.0;
            }
            for ( int cnt = 0; cnt < dimLU * dimRD; cnt++ ) {
               memLUxRD[ cnt ] = 0.0;
            }
            for ( int cnt = 0; cnt < dimLD * dimRU; cnt++ ) {
               memLDxRU[ cnt ] = 0.0;
            }
            for ( int cnt = 0; cnt < dimLD * dimRD; cnt++ ) {
               memLDxRD[ cnt ] = 0.0;
            }

            addDiagram1C( ikappa, memLDxRD, denS, denP, Prob->gMxElement( indexS, indexS, indexS, indexS ) );
            addDiagram1D( ikappa, memLDxRD, denS, denP, Prob->gMxElement( indexS + 1, indexS + 1, indexS + 1, indexS + 1 ) );

            addDiagram2dall( ikappa, memS, memLDxRD, denS );
            addDiagram3Eand3H( ikappa, memS, memLDxRD, denS );

            if ( !atLeft ) {
               /*********************
                *  Diagrams group 1  *
                *********************/
               addDiagram1A( ikappa, memLUxRD, denS, denP, Xtensors[ indexS - 1 ] );

               /*********************
                   *  Diagrams group 2  *
                   *********************/
               addDiagram2b1and2b2( ikappa, memS, memLUxRD, denS, Atensors[ indexS - 1 ][ 0 ][ 0 ], AtensorsT[ indexS - 1 ][ 0 ][ 0 ] );
               addDiagram2c1and2c2( ikappa, memS, memLUxRD, denS, Atensors[ indexS - 1 ][ 0 ][ 1 ], AtensorsT[ indexS - 1 ][ 0 ][ 1 ] );
               addDiagram2b3spin0( ikappa, memS, memLUxRD, denS, CtensorsT[ indexS - 1 ][ 0 ][ 0 ] );
               addDiagram2b3spin1( ikappa, memS, memLUxRD, denS, DtensorsT[ indexS - 1 ][ 0 ][ 0 ] );
               addDiagram2c3spin0( ikappa, memS, memLUxRD, denS, CtensorsT[ indexS - 1 ][ 0 ][ 1 ] );
               addDiagram2c3spin1( ikappa, memS, memLUxRD, denS, DtensorsT[ indexS - 1 ][ 0 ][ 1 ] );

               /*********************
                   *  Diagrams group 3  *
                   *********************/
               addDiagram3Aand3D( ikappa, memS, memLUxRD, denS, Qtensors[ indexS - 1 ][ 0 ], QtensorsT[ indexS - 1 ][ 0 ], Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], temp );
               addDiagram3Band3I( ikappa, memS, memLUxRD, denS, Qtensors[ indexS - 1 ][ 1 ], QtensorsT[ indexS - 1 ][ 1 ], Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], temp );

               /*********************
                   *  Diagrams group 4  *
                   *********************/
               addDiagram4A1and4A2spin0( ikappa, memS, memLUxRD, denS, Atensors[ indexS - 1 ][ 1 ][ 0 ], AtensorsT[ indexS - 1 ][ 1 ][ 0 ] );
               addDiagram4A1and4A2spin1( ikappa, memS, memLUxRD, denS, Btensors[ indexS - 1 ][ 1 ][ 0 ], BtensorsT[ indexS - 1 ][ 1 ][ 0 ] );
               addDiagram4A3and4A4spin0( ikappa, memS, memLUxRD, denS, Ctensors[ indexS - 1 ][ 1 ][ 0 ], CtensorsT[ indexS - 1 ][ 1 ][ 0 ] );
               addDiagram4A3and4A4spin1( ikappa, memS, memLUxRD, denS, Dtensors[ indexS - 1 ][ 1 ][ 0 ], DtensorsT[ indexS - 1 ][ 1 ][ 0 ] );
               addDiagram4D( ikappa, memS, memLUxRD, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], temp );
               addDiagram4I( ikappa, memS, memLUxRD, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], temp );
            }

            if ( !atRight ) {
               /*********************
                *  Diagrams group 1  *
                *********************/
               addDiagram1B( ikappa, memLDxRU, denS, denP, Xtensors[ indexS + 1 ] );

               /*********************
                   *  Diagrams group 2  *
                   *********************/
               addDiagram2e1and2e2( ikappa, memS, memLDxRU, denS, Atensors[ indexS + 1 ][ 0 ][ 1 ], AtensorsT[ indexS + 1 ][ 0 ][ 1 ] );
               addDiagram2f1and2f2( ikappa, memS, memLDxRU, denS, Atensors[ indexS + 1 ][ 0 ][ 0 ], AtensorsT[ indexS + 1 ][ 0 ][ 0 ] );
               addDiagram2e3spin0( ikappa, memS, memLDxRU, denS, CtensorsT[ indexS + 1 ][ 0 ][ 1 ] );
               addDiagram2e3spin1( ikappa, memS, memLDxRU, denS, DtensorsT[ indexS + 1 ][ 0 ][ 1 ] );
               addDiagram2f3spin0( ikappa, memS, memLDxRU, denS, CtensorsT[ indexS + 1 ][ 0 ][ 0 ] );
               addDiagram2f3spin1( ikappa, memS, memLDxRU, denS, DtensorsT[ indexS + 1 ][ 0 ][ 0 ] );

               // /*********************
               // *  Diagrams group 3  *
               // *********************/
               addDiagram3Kand3F( ikappa, memS, memLDxRU, denS, Qtensors[ indexS + 1 ][ 1 ], QtensorsT[ indexS + 1 ][ 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram3Land3G( ikappa, memS, memLDxRU, denS, Qtensors[ indexS + 1 ][ 0 ], QtensorsT[ indexS + 1 ][ 0 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );

               /*********************
                   *  Diagrams group 4  *
                   *********************/
               addDiagram4J1and4J2spin0( ikappa, memS, memLDxRU, denS, Atensors[ indexS + 1 ][ 1 ][ 0 ], AtensorsT[ indexS + 1 ][ 1 ][ 0 ] );
               addDiagram4J1and4J2spin1( ikappa, memS, memLDxRU, denS, Btensors[ indexS + 1 ][ 1 ][ 0 ], BtensorsT[ indexS + 1 ][ 1 ][ 0 ] );
               addDiagram4J3and4J4spin0( ikappa, memS, memLDxRU, denS, Ctensors[ indexS + 1 ][ 1 ][ 0 ], CtensorsT[ indexS + 1 ][ 1 ][ 0 ] );
               addDiagram4J3and4J4spin1( ikappa, memS, memLDxRU, denS, Dtensors[ indexS + 1 ][ 1 ][ 0 ], DtensorsT[ indexS + 1 ][ 1 ][ 0 ] );
               addDiagram4F( ikappa, memS, memLDxRU, denS, Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4G( ikappa, memS, memLDxRU, denS, Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
            }
            if ( ( !atLeft ) && ( !atRight ) ) {
               addDiagram2a1spin0( ikappa, memS, memLUxRU, denS, AtensorsT, S0tensorsT, temp );
               addDiagram2a2spin0( ikappa, memS, memLUxRU, denS, Atensors, S0tensors, temp );
               addDiagram2a1spin1( ikappa, memS, memLUxRU, denS, BtensorsT, S1tensorsT, temp );
               addDiagram2a2spin1( ikappa, memS, memLUxRU, denS, Btensors, S1tensors, temp );
               addDiagram2a3spin0( ikappa, memS, memLUxRU, denS, Ctensors, CtensorsT, F0tensors, F0tensorsT, temp );
               addDiagram2a3spin1( ikappa, memS, memLUxRU, denS, Dtensors, DtensorsT, F1tensors, F1tensorsT, temp );

               addDiagram3C( ikappa, memS, memLUxRU, denS, Qtensors[ indexS - 1 ], QtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram3J( ikappa, memS, memLUxRU, denS, Qtensors[ indexS + 1 ], QtensorsT[ indexS + 1 ], Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], temp );

               addDiagram4B1and4B2spin0( ikappa, memS, memLUxRU, denS, Atensors[ indexS - 1 ], AtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4B1and4B2spin1( ikappa, memS, memLUxRU, denS, Btensors[ indexS - 1 ], BtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4B3and4B4spin0( ikappa, memS, memLUxRU, denS, Ctensors[ indexS - 1 ], CtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4B3and4B4spin1( ikappa, memS, memLUxRU, denS, Dtensors[ indexS - 1 ], DtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4C1and4C2spin0( ikappa, memS, memLUxRU, denS, Atensors[ indexS - 1 ], AtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4C1and4C2spin1( ikappa, memS, memLUxRU, denS, Btensors[ indexS - 1 ], BtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4C3and4C4spin0( ikappa, memS, memLUxRU, denS, Ctensors[ indexS - 1 ], CtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4C3and4C4spin1( ikappa, memS, memLUxRU, denS, Dtensors[ indexS - 1 ], DtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp );
               addDiagram4E( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram4H( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram4K1and4K2spin0( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Atensors[ indexS + 1 ], AtensorsT[ indexS + 1 ], temp );
               addDiagram4L1and4L2spin0( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Atensors[ indexS + 1 ], AtensorsT[ indexS + 1 ], temp );
               addDiagram4K1and4K2spin1( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Btensors[ indexS + 1 ], BtensorsT[ indexS + 1 ], temp );
               addDiagram4L1and4L2spin1( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Btensors[ indexS + 1 ], BtensorsT[ indexS + 1 ], temp );
               addDiagram4K3and4K4spin0( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ctensors[ indexS + 1 ], CtensorsT[ indexS + 1 ], temp );
               addDiagram4L3and4L4spin0( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ctensors[ indexS + 1 ], CtensorsT[ indexS + 1 ], temp );
               addDiagram4K3and4K4spin1( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Dtensors[ indexS + 1 ], DtensorsT[ indexS + 1 ], temp );
               addDiagram4L3and4L4spin1( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Dtensors[ indexS + 1 ], DtensorsT[ indexS + 1 ], temp );

               addDiagram5A( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram5B( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram5C( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram5D( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram5E( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
               addDiagram5F( ikappa, memS, memLUxRU, denS, Ltensors[ indexS - 1 ], LtensorsT[ indexS - 1 ], Ltensors[ indexS + 1 ], LtensorsT[ indexS + 1 ], temp, temp2 );
            }

            int dimLUxRU = dimLU * dimRU;
            int dimLDxRD = dimLD * dimRD;

            dcomplex * BlockP = memP + denP->gKappa2index( ikappa );

            dcomplex one = 1.0;
            int inc      = 1;
            zaxpy_( &dimLUxRU, &one, memLUxRU, &inc, BlockP, &inc );

            if ( !atRight ) {
               dcomplex * BlockOT = OtensorsR->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

               zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &one,
                       memLUxRD, &dimLU, BlockOT, &dimRU, &one, BlockP, &dimLU );
            } else {
               int dimLUxRD = dimLU * dimRD;
               zaxpy_( &dimLUxRD, &one, memLUxRD, &inc, BlockP, &inc );
            }

            if ( !atLeft ) {
               dcomplex * BlockO = OtensorsL->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

               zgemm_( &notrans, &notrans, &dimLU, &dimRU, &dimLD, &one, BlockO, &dimLU,
                       memLDxRU, &dimLD, &one, BlockP, &dimLU );
            } else {
               int dimLDxRU = dimLD * dimRU;
               zaxpy_( &dimLDxRU, &one, memLDxRU, &inc, BlockP, &inc );
            }

            if ( atLeft ) {
               dcomplex * BlockOT = OtensorsR->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

               zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &one, memLDxRD, &dimLD,
                       BlockOT, &dimRU, &one, BlockP, &dimLD );
            }

            if ( atRight ) {
               dcomplex * BlockO = OtensorsL->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

               zgemm_( &notrans, &notrans, &dimLU, &dimRU, &dimLD, &one, BlockO, &dimLU, memLDxRD, &dimLD, &one, BlockP, &dimLU );
            }

            if ( ( !atLeft ) && ( !atRight ) ) {
               dcomplex * BlockOT = OtensorsR->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
               dcomplex * BlockO  = OtensorsL->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

               dcomplex * temp3 = new dcomplex[ dimLD * dimRU ];
               dcomplex zero    = 0.0;
               zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &one, memLDxRD, &dimLD, BlockOT, &dimRU, &zero, temp3, &dimLD );

               zgemm_( &notrans, &notrans, &dimLU, &dimRU, &dimLD, &one, BlockO, &dimLU, temp3, &dimLD, &one, BlockP, &dimLU );

               delete[] temp3;
            }

            delete[] memLUxRU;
            delete[] memLUxRD;
            delete[] memLDxRU;
            delete[] memLDxRD;
         }
      }

      delete[] temp;
      delete[] temp2;
   }
}

void CheMPS2::CHeffNS::Apply( CSobject * denS, CSobject * denP, CTensorL *** Ltensors, CTensorLT *** LtensorsT,
                              CTensorOperator **** Atensors, CTensorOperator **** AtensorsT,
                              CTensorOperator **** Btensors, CTensorOperator **** BtensorsT,
                              CTensorOperator **** Ctensors, CTensorOperator **** CtensorsT,
                              CTensorOperator **** Dtensors, CTensorOperator **** DtensorsT,
                              CTensorS0 **** S0tensors, CTensorS0T **** S0tensorsT, CTensorS1 **** S1tensors,
                              CTensorS1T **** S1tensorsT, CTensorF0 **** F0tensors, CTensorF0T **** F0tensorsT,
                              CTensorF1 **** F1tensors, CTensorF1T **** F1tensorsT, CTensorQ *** Qtensors,
                              CTensorQT *** QtensorsT, CTensorX ** Xtensors, CTensorO * OtensorsL,
                              CTensorO * OtensorsR ) {

   int inc1 = 1;

   // Copy content of Sobject
   denS->prog2symm(); // Convert mem of Sobject to symmetric conventions
   int oldlength        = denS->gKappa2index( denS->gNKappa() );
   dcomplex * old_state = new dcomplex[ oldlength ];
   zcopy_( &oldlength, denS->gStorage(), &inc1, old_state, &inc1 );

   // Prepare space for result
   denP->prog2symm(); // Convert mem of Pobject to symmetric conventions
   int newlength = denP->gKappa2index( denP->gNKappa() );

   dcomplex * new_state = new dcomplex[ newlength ];
   for ( int cnt = 0; cnt < newlength; cnt++ ) {
      new_state[ cnt ] = 0.0;
   }

   makeHeff( old_state, new_state, denS, denP, Ltensors, LtensorsT, Atensors,
             AtensorsT, Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors,
             DtensorsT, S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
             F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT, Xtensors,
             OtensorsL, OtensorsR );

   zcopy_( &newlength, new_state, &inc1, denP->gStorage(), &inc1 );

   delete[] old_state;
   delete[] new_state;

   denP->symm2prog();
   denS->symm2prog();
}