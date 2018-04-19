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

void CheMPS2::CSubSpaceExpander::addDiagram2a1spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** ATtensors, CTensorS0T **** S0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int theindex = out->gIndex();
   int dimL           = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR           = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {
            int ILdown    = Irreps::directProd( IL, S0Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockS0 = S0Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockS0, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }

   } else {

      for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < prob->gL(); l_delta++ ) {
            int ILdown    = Irreps::directProd( IL, ATtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockA = ATtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockA, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a1spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** BTtensors, CTensorS1T **** S1Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   const int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( theindex );

   int dimL = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) && ( abs( TwoSLdown - TwoSR ) <= TwoS ) ) {
            int fase                 = phase( TwoSR + TwoSL + TwoS + 2 );
            const dcomplex thefactor = fase * sqrt( ( TwoSR + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
               for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {
                  int ILdown    = Irreps::directProd( IL, S1Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = thefactor;
                     dcomplex beta  = 1.0;

                     dcomplex * BlockS1 = S1Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                     dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockS1, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }

   } else {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) && ( abs( TwoSLdown - TwoSR ) <= TwoS ) ) {
            int fase                 = phase( TwoSR + TwoSL + TwoS + 2 );
            const dcomplex thefactor = fase * sqrt( ( TwoSR + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
               for ( int l_delta = l_gamma + 1; l_delta < prob->gL(); l_delta++ ) {
                  int ILdown    = Irreps::directProd( IL, BTtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = thefactor;
                     dcomplex beta  = 1.0;

                     dcomplex * BlockB = BTtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockB, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a2spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out,
                                                          CTensorOperator **** Atensors, CTensorS0 **** S0tensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( theindex );

   int dimL = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {
            int ILdown    = Irreps::directProd( IL, S0tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockS0 = S0tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockS0, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }

   } else {

      for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < prob->gL(); l_delta++ ) {
            int ILdown    = Irreps::directProd( IL, Atensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockA = Atensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockA, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a2spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Btensors, CTensorS1 **** S1tensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( theindex );

   int dimL = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) && ( abs( TwoSLdown - TwoSR ) <= TwoS ) ) {
            int fase                 = phase( TwoSLdown + TwoSR + TwoS + 2 );
            const dcomplex thefactor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1 ) ) * ( TwoSR + 1 ) *
                                       Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
               for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {
                  int ILdown    = Irreps::directProd( IL, S1tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = thefactor;
                     dcomplex beta  = 1.0;

                     dcomplex * BlockS1 = S1tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                     dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockS1, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }

   } else {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) && ( abs( TwoSLdown - TwoSR ) <= TwoS ) ) {
            int fase                 = phase( TwoSLdown + TwoSR + TwoS + 2 );
            const dcomplex thefactor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1 ) ) * ( TwoSR + 1 ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
               for ( int l_delta = l_gamma + 1; l_delta < prob->gL(); l_delta++ ) {
                  int ILdown    = Irreps::directProd( IL, Btensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = thefactor;
                     dcomplex beta  = 1.0;

                     dcomplex * BlockB = Btensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockB, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a3spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Ctensors, CTensorOperator **** CTtensors, CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( theindex );

   int dimL = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
         for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {
            int ILdown    = Irreps::directProd( IL, F0tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               //no transpose
               dcomplex * BlockF0 = F0tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockF0, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {
            int ILdown    = Irreps::directProd( IL, F0Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               //transpose
               dcomplex * BlockF0 = F0Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, BlockF0, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }

   } else {

      for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
         for ( int l_beta = l_delta + 1; l_beta < prob->gL(); l_beta++ ) {

            int ILdown    = Irreps::directProd( IL, Ctensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
            int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * ptr    = Ctensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, ptr, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }

      for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
         for ( int l_delta = l_beta; l_delta < prob->gL(); l_delta++ ) {

            int ILdown    = Irreps::directProd( IL, CTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * ptr    = CTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );

               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &alpha, ptr, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a3spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Dtensors, CTensorOperator **** DTtensors, CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = initBKUp->gIrrep( theindex );

   int dimL = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {
      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) && ( abs( TwoSLdown - TwoSR ) <= TwoS ) ) {
            int fase           = phase( TwoSLdown + TwoSR + TwoS + 2 );
            dcomplex prefactor = fase * sqrt( ( TwoSLdown + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
               for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {
                  int ILdown    = Irreps::directProd( IL, F1tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 1.0;

                     dcomplex * BlockF1 = F1tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                     dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &prefactor, BlockF1, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }

               fase      = phase( TwoSL + TwoSR + TwoS + 2 );
               prefactor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {
                     int ILdown    = Irreps::directProd( IL, F1Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->get_irrep() );
                     int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                     int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

                     if ( memSkappa != -1 ) {
                        dcomplex beta = 1.0;

                        dcomplex * BlockF1 = F1Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                        dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &prefactor, BlockF1, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }

   } else {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) && ( abs( TwoSLdown - TwoSR ) <= TwoS ) ) {
            int fase           = phase( TwoSLdown + TwoSR + TwoS + 2 );
            dcomplex prefactor = fase * sqrt( ( TwoSLdown + 1.0 ) * ( TwoSR + 1.0 ) ) *
                                 Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
               for ( int l_beta = l_delta + 1; l_beta < prob->gL(); l_beta++ ) {

                  int ILdown    = Irreps::directProd( IL, Dtensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 1.0;

                     dcomplex * ptr    = Dtensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &prefactor, ptr, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }
            }

            // 1 of 2
            fase      = phase( TwoSL + TwoSR + TwoS + 2 );
            prefactor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1 ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
               for ( int l_delta = l_beta; l_delta < prob->gL(); l_delta++ ) {

                  int ILdown    = Irreps::directProd( IL, DTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 1.0;

                     dcomplex * ptr    = DTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );

                     zgemm_( &notrans, &notrans, &dimL, &dimR, &dimLdown, &prefactor, ptr, &dimL, BlockS, &dimLdown, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2b1and2b2Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N == 0 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimLdown = initBKDown->gCurrentDim( theindex, NL - 2, TwoSL, IL );

      if ( dimLdown > 0 ) {
         int dimLup = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

         int memSkappa = in->gKappa( NL - 2, TwoSL, IL, NR, TwoSR, IR );
         if ( memSkappa != -1 ) {
            dcomplex alpha = sqrt( 2.0 );
            dcomplex beta  = 1.0;
            int inc        = 1;
            int dim        = dimLdown * dimR;

            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
            zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
         }
      }
   }

   if ( N == 2 ) {

      int theindex = out->gIndex();
      int NL       = out->gNL( ikappa );
      int TwoSL    = out->gTwoSL( ikappa );
      int IL       = out->gIL( ikappa );

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimLdown = initBKDown->gCurrentDim( theindex, NL + 2, TwoSL, IL );
      if ( dimLdown > 0 ) {
         int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

         int memSkappa = in->gKappa( NL + 2, TwoSL, IL, NR, TwoSR, IR );
         if ( memSkappa != -1 ) {
            dcomplex alpha = sqrt( 2.0 );
            dcomplex beta  = 1.0;
            int inc        = 1;
            int dim        = dimLdown * dimR;

            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
            zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2b3spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N != 0 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );

      int dimLU = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimLD = initBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR  = sseBKDown->gCurrentDim( theindex + 1, out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      int memSkappa = in->gKappa( out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ), out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );
      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
         dcomplex alpha    = ( ( N == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &notrans, &dimLU, &dimR, &dimLD, &alpha, Cblock, &dimLU, in->gStorage() + in->gKappa2index( memSkappa ), &dimLD, &beta, memHeff, &dimLU );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2b3spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N          = out->gNR( ikappa ) - out->gNL( ikappa );
   const int TwoS = ( N == 1 ) ? 1 : 0;
   if ( N == 1 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );
      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimLup = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR   = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int dimLdown = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, IL );
         if ( dimLdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NL, TwoSL, IL, NL, TwoSLdown, IL );

            int TwoS2     = 0;
            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;

            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = in->gKappa( NL, TwoSLdown, IL, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {

                     int fase       = phase( TwoSLdown + TwoSR + TwoS + TwoS2 + TwoJdown - 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoS + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoS, 2, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta  = 1.0;
                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Dblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2e1and2e2Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N1 == 2 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimLdown = initBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 );
         dcomplex beta  = 1.0;
         int inc        = 1;
         int dim        = dimLdown * dimRdown;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
      }
   }

   if ( N1 == 0 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimL     = initBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 );
         dcomplex beta  = 1.0;
         int inc        = 1;
         int dim        = dimL * dimRdown;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2e3spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N1 != 0 ) {

      int theindex = out->gIndex();

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimRD = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int dimL  = initBKDown->gCurrentDim( theindex, out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ) );

      int memSkappa = in->gKappa( out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ), out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      if ( memSkappa != -1 ) {

         dcomplex alpha = ( ( N1 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta  = 1.0;
         int inc        = 1;
         int dim        = dimL * dimRD;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2e3spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );
   if ( N1 == 1 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int TwoJ = ( N1 == 1 ) ? 1 : 0;
      int N2   = 0;

      int dimL   = initBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      if ( dimRdown > 0 ) {

         dcomplex * Dblock = DTtensor->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

         int TwoS2     = ( N2 == 1 ) ? 1 : 0;
         int TwoJstart = ( ( TwoSR != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
            if ( abs( TwoSL - TwoSR ) <= TwoJdown ) {

               int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

               if ( memSkappa != -1 ) {
                  int fase       = phase( TwoSR + TwoSL + 2 * TwoJ + TwoS2 + 1 );
                  dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSR, TwoSR, TwoSL );
                  int inc        = 1;
                  int dim        = dimL * dimRdown;

                  dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
               }
            }
         }
      }
   }
}
