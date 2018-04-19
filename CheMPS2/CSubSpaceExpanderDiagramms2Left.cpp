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

void CheMPS2::CSubSpaceExpander::addDiagram2a1spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** ATtensors, CTensorS0T **** S0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   const int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( leftSum ) {
      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {
            int IRdown    = Irreps::directProd( IR, ATtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
            int dimRdown  = initBKUp->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockA = ATtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockA, &dimR, &beta, memHeff, &dimL );
            }
         }
      }

   } else {
      for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < prob->gL(); l_delta++ ) {
            int IRdown    = Irreps::directProd( IR, S0Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
            int dimRdown  = initBKUp->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockS0 = S0Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockS0, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a1spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** BTtensors, CTensorS1T **** S1Ttensors, dcomplex * workspace ) {

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

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( leftSum ) {

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSL - TwoSRdown ) <= TwoS ) ) {
            int fase                 = phase( TwoSRdown + TwoSL + TwoS + 2 );
            const dcomplex thefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
               for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {
                  int IRdown    = Irreps::directProd( IR, BTtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = thefactor;
                     dcomplex beta  = 1.0;

                     dcomplex * BlockB = BTtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockB, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }

   } else {

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSL - TwoSRdown ) <= TwoS ) ) {
            int fase                 = phase( TwoSRdown + TwoSL + TwoS + 2 );
            const dcomplex thefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
               for ( int l_delta = l_gamma + 1; l_delta < prob->gL(); l_delta++ ) {
                  int IRdown    = Irreps::directProd( IR, S1Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = thefactor;
                     dcomplex beta  = 1.0;

                     dcomplex * BlockS1 = S1Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                     dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockS1, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a2spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out,
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

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {
            int IRdown    = Irreps::directProd( IR, Atensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockA = Atensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockA, &dimR, &beta, memHeff, &dimL );
            }
         }
      }

   } else {

      for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < prob->gL(); l_delta++ ) {
            int IRdown    = Irreps::directProd( IR, S0tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockS0 = S0tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockS0, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a2spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Btensors, CTensorS1 **** S1tensors, dcomplex * workspace ) {

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

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) ) {
               int fase                 = phase( TwoSLdown + TwoSR + TwoS + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1 ) ) * ( TwoSRdown + 1 ) *
                                          Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {
                     int IRdown    = Irreps::directProd( IR, Btensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
                     int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );
                     int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {
                        dcomplex alpha = thefactor;
                        dcomplex beta  = 1.0;

                        dcomplex * BlockB = Btensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                        dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockB, &dimR, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }

   } else {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) ) {
               int fase                 = phase( TwoSLdown + TwoSR + TwoS + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1 ) ) * ( TwoSRdown + 1 ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_gamma = theindex + 1; l_gamma < prob->gL(); l_gamma++ ) {
                  for ( int l_delta = l_gamma + 1; l_delta < prob->gL(); l_delta++ ) {
                     int IRdown    = Irreps::directProd( IR, S1tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
                     int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );
                     int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {
                        dcomplex alpha = thefactor;
                        dcomplex beta  = 1.0;

                        dcomplex * BlockS1 = S1tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                        dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockS1, &dimR, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a3spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Ctensors, CTensorOperator **** CTtensors, CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors, dcomplex * workspace ) {

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

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( leftSum ) {

      for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
         for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {
            int IRdown    = Irreps::directProd( IR, Ctensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->get_irrep() );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockC = Ctensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockC, &dimR, &beta, memHeff, &dimL );
            }
         }
      }

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {
            int IRdown    = Irreps::directProd( IR, CTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->get_irrep() );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockC = CTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockC, &dimR, &beta, memHeff, &dimL );
            }
         }
      }

   } else {

      for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
         for ( int l_beta = l_delta + 1; l_beta < prob->gL(); l_beta++ ) {
            int IRdown    = Irreps::directProd( IR, F0tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->get_irrep() );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockF0 = F0tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockF0, &dimR, &beta, memHeff, &dimL );
            }
         }
      }

      for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
         for ( int l_delta = l_beta; l_delta < prob->gL(); l_delta++ ) {
            int IRdown    = Irreps::directProd( IR, F0Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->get_irrep() );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex alpha = 1.0;
               dcomplex beta  = 1.0;

               dcomplex * BlockF0 = F0Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, BlockF0, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2a3spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Dtensors, CTensorOperator **** DTtensors, CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors, dcomplex * workspace ) {

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

   const bool leftSum = ( theindex < prob->gL() * 0.5 ) ? true : false;

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( leftSum ) {

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSL - TwoSRdown ) <= TwoS ) ) {
            int fase           = phase( TwoSL + TwoSRdown + TwoS + 2 );
            dcomplex prefactor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
               for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {
                  int IRdown    = Irreps::directProd( IR, Dtensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->get_irrep() );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 0.0;

                     dcomplex * ptr    = Dtensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, workspace, &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }

            fase      = phase( TwoSL + TwoSR + TwoS + 2 );
            prefactor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
               for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {
                  int IRdown    = Irreps::directProd( IR, DTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->get_irrep() );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 0.0;

                     dcomplex * BlockD = DTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                     dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, BlockS, &dimL, BlockD, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }

   } else {

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSL - TwoSRdown ) <= TwoS ) ) {
            int fase           = phase( TwoSL + TwoSRdown + TwoS + 2 );
            dcomplex prefactor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) *
                                 Wigner::wigner6j( TwoSL, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
               for ( int l_beta = l_delta + 1; l_beta < prob->gL(); l_beta++ ) {
                  int IRdown    = Irreps::directProd( IR, F1tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->get_irrep() );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 0.0;

                     dcomplex * BlockF1 = F1tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                     dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, BlockS, &dimL, BlockF1, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }

            fase      = phase( TwoSL + TwoSR + TwoS + 2 );
            prefactor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1 ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

            for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
               for ( int l_delta = l_beta; l_delta < prob->gL(); l_delta++ ) {
                  int IRdown    = Irreps::directProd( IR, F1Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->get_irrep() );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex beta = 1.0;

                     dcomplex * BlockF1 = F1Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                     dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, BlockS, &dimL, BlockF1, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2b1and2b2Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N = out->gNR( ikappa ) - out->gNL( ikappa );

   int theindex = out->gIndex();

   if ( N == 0 ) {
      int dimL      = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR      = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         int dim        = dimL * dimR;
         int inc        = 1;
         dcomplex alpha = sqrt( 2.0 );

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
      }
   }

   if ( N == 2 ) {
      int dimL      = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR      = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int memSkappa = in->gKappa( NL + 2, TwoSL, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 );
         int dim        = dimL * dimR;
         int inc        = 1;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2b3spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N = out->gNR( ikappa ) - out->gNL( ikappa );

   int theindex = out->gIndex();

   if ( N != 0 ) {
      int dimL      = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR      = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int memSkappa = in->gKappa( out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ), out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex alpha = ( ( N == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         int dim        = dimL * dimR;
         int inc        = 1;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2b3spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int theindex = out->gIndex();

   int N          = out->gNR( ikappa ) - out->gNL( ikappa );
   const int TwoS = ( N == 1 ) ? 1 : 0;

   if ( N == 1 ) {

      int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int TwoS2     = 0;
      int TwoJstart = ( ( TwoSR != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;

      for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
         if ( abs( TwoSL - TwoSR ) <= TwoJdown ) {

            int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase       = phase( TwoSL + TwoSR + TwoS + TwoS2 + TwoJdown - 1 );
               dcomplex alpha = fase * sqrt( 3.0 * ( TwoS + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoS, 2, TwoSL, TwoSL, TwoSR );
               int dim        = dimL * dimR;
               int inc        = 1;

               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zaxpy_( &dim, &alpha, BlockS, &inc, memHeff, &inc );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2e1and2e2Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL       = out->gNL( ikappa );
   int TwoSL    = out->gTwoSL( ikappa );
   int IL       = out->gIL( ikappa );
   int NR       = out->gNR( ikappa );
   int TwoSR    = out->gTwoSR( ikappa );
   int IR       = out->gIR( ikappa );

   int theindex = out->gIndex();

   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSR, IR );
   int dimRup   = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N1 == 2 ) {
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IR );

      if ( memSkappa != -1 ) {
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;

         dcomplex * BlockA = ATtensor->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IR );
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }

   if ( N1 == 0 ) {
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IR );

      if ( memSkappa != -1 ) {
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;

         dcomplex * BlockA = Atensor->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IR );
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2e3spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int theindex = out->gIndex();

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N1 != 0 ) {
      int dimRU = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int dimRD = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int dimL  = sseBKDown->gCurrentDim( theindex, out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ) );
      int memSkappa = in->gKappa( out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ), out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex beta     = 1.0;
         dcomplex alpha    = ( ( N1 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );

         dcomplex * Cblock = CTtensor->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         zgemm_( &notrans, &cotrans, &dimL, &dimRU, &dimRD, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Cblock, &dimRU, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram2e3spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor ) {

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
      int TwoJ  = ( N1 == 1 ) ? 1 : 0;
      int N2    = 0;

      int dimL   = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRup = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IR );

         if ( dimRdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IR );

            int TwoS2     = ( N2 == 1 ) ? 1 : 0;
            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IR );

                  if ( memSkappa != -1 ) {
                     int fase       = phase( TwoSRdown + TwoSL + 2 * TwoJ + TwoS2 + 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta  = 1.0;

                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Dblock, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}
