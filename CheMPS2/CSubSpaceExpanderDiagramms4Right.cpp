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

void CheMPS2::CSubSpaceExpander::addDiagram4B1and4B2spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Aleft, CTensorOperator *** ATleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B1A.spin0
   if ( N1 == 0 ) {

      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
               int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 && dimLup > 0 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  dcomplex * Ablock = ATleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Ablock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   //4B1B.spin0
   if ( N1 == 1 ) {

      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSR >= 0 ) ) {
         int fase              = phase( TwoSR + TwoSL + 2 + TwoJ + 2 * TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSR, TwoSR, TwoSL );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

            int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 && dimLup > 0 ) {
               dcomplex alpha = factor;
               dcomplex beta  = 1.0;

               dcomplex * Ablock = ATleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
               dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Ablock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   //4B2A.spin0
   if ( N1 == 1 ) {

      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSR >= 0 ) ) {
         int fase              = phase( TwoSR + TwoSL + 1 + TwoJ + 2 * TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSR, TwoSR, TwoSL );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

            int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 && dimLup > 0 ) {
               dcomplex alpha = factor;
               dcomplex beta  = 1.0;

               dcomplex * Ablock = Aleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
               dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Ablock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   //4B2B.spin0
   if ( N1 == 2 ) {

      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSR >= 0 ) ) {

            int fase              = phase( TwoSR + TwoSL + 2 + TwoJdown + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

               int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
               int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 && dimLup > 0 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Ablock = Aleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Ablock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4B1and4B2spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Bleft, CTensorOperator *** BTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B1A.spin1
   if ( N1 == 0 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = ( TwoS2 == 0 ) ? 1 : -1;
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1 ) * ( TwoSL + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSR, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 && dimLup > 0 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Bblock = BTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Bblock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4B1B.spin1
   if ( N1 == 1 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSR - TwoSR + TwoSL + 3 - TwoSLdown + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1 ) * ( TwoSL + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSR, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

               int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
               int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 && dimLup > 0 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Bblock = BTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Bblock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   //4B2A.spin1
   if ( N1 == 1 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = ( TwoS2 == 0 ) ? 1 : -1;
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSR, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
               int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 && dimLup > 0 ) {
                  dcomplex alpha    = factor;
                  dcomplex beta     = 1.0;
                  dcomplex * Bblock = Bleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Bblock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   //4B2B.spin1
   if ( N1 == 2 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = phase( TwoSLdown + 3 - TwoSL + TwoSR - TwoSR + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSR, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

                  int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 && dimLup > 0 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Bblock = Bleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Bblock, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4B3and4B4spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Cleft, CTensorOperator *** CTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B3A.spin0
   if ( N1 == 1 ) {

      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSR >= 0 ) ) {
         int fase              = phase( TwoSR + TwoSL + TwoJ + 2 * TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSR, TwoSR, TwoSL );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
            int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 && dimLdown > 0 ) {
               dcomplex alpha = factor;
               dcomplex beta  = 1.0;

               dcomplex * ptr    = CTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
               dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   //4B3B.spin0
   if ( N1 == 2 ) {

      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
               int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 && dimLdown > 0 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * ptr    = CTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   //4B4A.spin0
   if ( N1 == 0 ) {

      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSR + TwoSL + TwoJdown + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

               int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
               int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 && dimLdown > 0 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * ptr    = Cleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   //4B4B.spin0
   if ( N1 == 1 ) {

      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSR >= 0 ) ) {
         int fase              = phase( TwoSR + TwoSL + 1 + TwoJ + 2 * TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSR, TwoSR, TwoSL );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

            int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSL, ILdown );
            int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 && dimLdown > 0 ) {
               dcomplex alpha = factor;
               dcomplex beta  = 1.0;

               dcomplex * ptr    = Cleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
               dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4B3and4B4spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Dleft, CTensorOperator *** DTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B3A.spin1
   if ( N1 == 1 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSL - TwoSLdown + TwoSR - TwoSR + 3 + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1.0 ) * ( TwoJ + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSR, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
               int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * ptr    = DTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );

                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   //4B3B.spin1
   if ( N1 == 2 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = ( TwoS2 == 0 ) ? -1 : 1;
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSR, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * ptr    = DTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4B4A.spin1
   if ( N1 == 0 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = phase( TwoSR - TwoSR + TwoSLdown - TwoSL + 3 + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSR, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                  int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * ptr    = Dleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4B4B.spin1
   if ( N1 == 1 ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = ( TwoS2 == 0 ) ? -1 : 1;
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSR, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
               int dimLdown  = initBKDown->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
               int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSR, IR );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * ptr    = Dleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                  dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, ptr, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4ERight( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   if ( N1 == 0 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR - TwoS2 );
            const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoSR, TwoSLdown, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {
               int ILdown   = Irreps::directProd( IL, Irrep );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                  bool isPossibleLeft = false;
                  for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_alpha ) ) { isPossibleLeft = true; }
                  }
                  bool isPossibleRight = false;
                  for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_beta ) ) { isPossibleRight = true; }
                  }
                  if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                     for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_alpha ) ) {

                           int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha        = factor;
                              dcomplex beta         = 1.0;
                              dcomplex * LblockLeft = LTleft[ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                              dcomplex * blockS     = in->gStorage() + in->gKappa2index( memSkappa );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LblockLeft, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4E2
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {

            int fase              = phase( TwoSLdown + TwoSR - TwoS2 );
            const dcomplex factor = fase * sqrt( ( TwoSLdown + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS2, TwoSR, TwoSL, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

               int ILdown   = Irreps::directProd( IL, Irrep );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                  bool isPossibleLeft = false;
                  for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_gamma ) ) { isPossibleLeft = true; }
                  }
                  bool isPossibleRight = false;
                  for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_delta ) ) { isPossibleRight = true; }
                  }
                  if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                     for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_gamma ) ) {

                           int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha = factor;
                              dcomplex beta  = 1.0;

                              dcomplex * LblockLeft = Lleft[ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                              dcomplex * blockS     = in->gStorage() + in->gKappa2index( memSkappa );
                              zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LblockLeft, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4E3A
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {

               int fase               = phase( TwoSL + TwoSR + TwoJ + TwoSLdown + TwoSR + 1 - TwoS2 );
               const dcomplex factor1 = fase * sqrt( ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoJdown, 1, TwoSLdown ) * Wigner::wigner6j( TwoJ, 1, TwoS2, TwoSR, TwoSL, TwoSR );

               dcomplex factor2 = 0.0;
               if ( TwoJ == TwoJdown ) {
                  fase    = phase( TwoSL + TwoSR + TwoJ + 3 + 2 * TwoS2 );
                  factor2 = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoJ, TwoSR, TwoSL, 1 );
               }

               for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

                  int ILdown   = Irreps::directProd( IL, Irrep );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                     bool isPossibleLeft = false;
                     for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_alpha ) ) { isPossibleLeft = true; }
                     }
                     bool isPossibleRight = false;
                     for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_delta ) ) { isPossibleRight = true; }
                     }
                     if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                        for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                           if ( Irrep == initBKUp->gIrrep( l_alpha ) ) {
                              dcomplex theFactor = 0;
                              for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                                 if ( Irrep == initBKUp->gIrrep( l_delta ) ) {
                                    dcomplex prefact = factor1 * prob->gMxElement( l_alpha, theindex, theindex, l_delta );
                                    if ( TwoJ == TwoJdown ) { prefact += factor2 * prob->gMxElement( l_alpha, theindex, l_delta, theindex ); }
                                    theFactor += prefact;
                                 }
                              }

                              int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha        = theFactor;
                                 dcomplex beta         = 1.0;
                                 dcomplex * LblockLeft = LTleft[ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                                 dcomplex * blockS     = in->gStorage() + in->gKappa2index( memSkappa );

                                 zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LblockLeft, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {

            int fase              = phase( TwoSL + TwoSR - TwoS2 + 3 );
            const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS2, TwoSR, TwoSL, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

               int ILdown   = Irreps::directProd( IL, Irrep );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                  bool isPossibleLeft = false;
                  for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_alpha ) ) { isPossibleLeft = true; }
                  }
                  bool isPossibleRight = false;
                  for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_delta ) ) { isPossibleRight = true; }
                  }
                  if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                     for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_alpha ) ) {
                           dcomplex theFactor = 0.0;

                           for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                              if ( Irrep == initBKUp->gIrrep( l_delta ) ) {
                                 dcomplex prefact = prob->gMxElement( l_alpha, theindex, theindex, l_delta ) - 2 * prob->gMxElement( l_alpha, theindex, l_delta, theindex );
                                 theFactor += prefact;
                              }
                           }

                           int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha        = factor * theFactor;
                              dcomplex beta         = 1.0;
                              dcomplex * LblockLeft = LTleft[ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                              dcomplex * blockS     = in->gStorage() + in->gKappa2index( memSkappa );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LblockLeft, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4E4A
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {

               int fase               = phase( TwoSL + TwoSR + TwoJdown + TwoSLdown + TwoSR + 1 - TwoS2 );
               const dcomplex factor1 = fase * sqrt( ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS2, TwoJ, 1, TwoSL ) * Wigner::wigner6j( TwoJdown, 1, TwoS2, TwoSR, TwoSLdown, TwoSR );

               dcomplex factor2 = 0.0;
               if ( TwoJ == TwoJdown ) {
                  fase    = phase( TwoSLdown + TwoSR + TwoJ + 3 + 2 * TwoS2 );
                  factor2 = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSR, TwoSLdown, 1 );
               }

               for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

                  int ILdown   = Irreps::directProd( IL, Irrep );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                     bool isPossibleLeft = false;
                     for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_gamma ) ) { isPossibleLeft = true; }
                     }
                     bool isPossibleRight = false;
                     for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_beta ) ) { isPossibleRight = true; }
                     }
                     if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                        for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                           if ( Irrep == initBKUp->gIrrep( l_gamma ) ) {
                              dcomplex theFactor = 0.0;

                              for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                                 if ( Irrep == initBKUp->gIrrep( l_beta ) ) {
                                    dcomplex prefact = factor1 * prob->gMxElement( l_gamma, theindex, theindex, l_beta );
                                    if ( TwoJ == TwoJdown ) { prefact += factor2 * prob->gMxElement( l_gamma, theindex, l_beta, theindex ); }
                                    theFactor += prefact;
                                 }
                              }

                              int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = theFactor;
                                 dcomplex beta  = 1.0;

                                 dcomplex * LblockLeft = Lleft[ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                                 dcomplex * blockS     = in->gStorage() + in->gKappa2index( memSkappa );
                                 zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LblockLeft, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4E4B
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {

            int fase              = phase( TwoSLdown + TwoSR - TwoS2 + 3 );
            const dcomplex factor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoSR, TwoSLdown, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

               int ILdown   = Irreps::directProd( IL, Irrep );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                  bool isPossibleLeft = false;
                  for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_gamma ) ) { isPossibleLeft = true; }
                  }
                  bool isPossibleRight = false;
                  for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                     if ( Irrep == initBKUp->gIrrep( l_beta ) ) { isPossibleRight = true; }
                  }
                  if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                     for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                        if ( Irrep == initBKUp->gIrrep( l_gamma ) ) {
                           dcomplex theFactor = 0.0;
                           for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                              if ( Irrep == initBKUp->gIrrep( l_beta ) ) {
                                 dcomplex prefact = prob->gMxElement( l_gamma, theindex, theindex, l_beta ) - 2 * prob->gMxElement( l_gamma, theindex, l_beta, theindex );
                                 theFactor += prefact;
                              }
                           }

                           int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha = theFactor * factor;
                              dcomplex beta  = 1.0;

                              dcomplex * LblockLeft = Lleft[ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                              dcomplex * blockS     = in->gStorage() + in->gKappa2index( memSkappa );
                              zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LblockLeft, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L1and4L2spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Aright, CTensorOperator *** ATright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L1A.spin0
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {

            int fase              = phase( TwoSLdown + TwoSR + 2 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSL, TwoSLdown, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4L1B.spin0
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) ) {
               int fase              = phase( TwoSLdown + TwoSR + 3 + TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L2A.spin0
   if ( N1 == 0 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) ) {
               int fase              = phase( TwoSL + TwoSR + 2 + TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSLdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L2B.spin0
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {

            int fase              = phase( TwoSL + TwoSR + 3 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSLdown, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L1and4L2spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Bright, CTensorOperator *** BTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L1A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {

            int fase              = phase( 1 + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );

               int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4L1B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = phase( TwoSR - TwoSR + TwoSLdown - TwoSL + TwoS2 - TwoJdown ); //bug fixed
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L2A.spin1
   if ( N1 == 0 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = phase( 1 + TwoS2 - TwoJdown );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L2B.spin1
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSR - TwoSR + TwoSL - TwoSLdown + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L3and4L4spin0Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Cright, CTensorOperator *** CTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L3A.spin0
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR + 1 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSLdown, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4L3B.spin0
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) ) {
               int fase              = phase( TwoSL + TwoSR + 2 + TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSLdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex beta  = 1.0;
                        dcomplex alpha = factor;

                        dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L4A.spin0
   if ( N1 == 0 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) ) {
               int fase              = phase( TwoSLdown + TwoSR + 1 + TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex beta  = 1.0;
                        dcomplex alpha = factor;

                        dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L4B.spin0
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {
            int fase              = phase( TwoSLdown + TwoSR + 2 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSLdown, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L3and4L4spin1Right( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Dright, CTensorOperator *** DTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRdown = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L3A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSL - TwoSLdown + 2 + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1.0 ) * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4L3B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = phase( TwoSR - TwoSR + 1 + TwoS2 - TwoJdown );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1.0 ) * ( TwoSLdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L4A.spin1
   if ( N1 == 0 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int TwoJstart = ( ( TwoSLdown != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
               int fase              = phase( TwoSLdown + 2 - TwoSL + TwoS2 - TwoJdown );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
                  int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                        dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                        zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4L4B.spin1
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
            int fase              = phase( TwoSR - TwoSR + 1 + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSR, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
               int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

               if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * blockS = in->gStorage() + in->gKappa2index( memSkappa );
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, blockL, &dimLup, blockS, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}