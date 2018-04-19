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

void CheMPS2::CSubSpaceExpander::addDiagram4B1and4B2spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Aleft, CTensorOperator *** ATleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B1A.spin0
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4B1B.spin0
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSR + TwoSL + 2 + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4B2A.spin0
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4B2B.spin0
   if ( N1 == 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSRdown + TwoSL + 2 + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4B1and4B2spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Bleft, CTensorOperator *** BTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B1A.spin1
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = ( TwoS2 == 0 ) ? 1 : -1;
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoSL + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4B1B.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSR - TwoSRdown + TwoSL + 3 - TwoSL + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoSL + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4B2A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = ( TwoS2 == 0 ) ? 1 : -1;
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4B2B.spin1
   if ( N1 == 2 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSL + 3 - TwoSL + TwoSRdown - TwoSR + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4B3and4B4spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Cleft, CTensorOperator *** CTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B3A.spin0
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSR + TwoSL + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4B3B.spin0
   if ( N1 == 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4B4A.spin0
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSRdown + TwoSL + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4B4B.spin0
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4B3and4B4spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Dleft, CTensorOperator *** DTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B3A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSL - TwoSL + TwoSR - TwoSRdown + 3 + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4B3B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = ( TwoS2 == 0 ) ? -1 : 1;
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4B4A.spin1
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSRdown - TwoSR + TwoSL - TwoSL + 3 + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

                  int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
                  int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4B4B.spin1
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = ( TwoS2 == 0 ) ? -1 : 1;
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSL, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

            for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {

               int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
               int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
               int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4ELeft( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4E1
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoJ ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR - TwoS2 );
            const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoSRdown, TwoSL, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {
               int IRdown   = Irreps::directProd( IR, Irrep );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {
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

                           int size = dimR * dimRdown;
                           for ( int cnt = 0; cnt < size; cnt++ ) {
                              temp[ cnt ] = 0.0;
                           }
                           for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                              if ( Irrep == initBKUp->gIrrep( l_beta ) ) {
                                 dcomplex * LblockRight = Lright[ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                                 dcomplex prefact       = prob->gMxElement( l_alpha, l_beta, theindex, theindex );
                                 zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                              }
                           }

                           int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

                           if ( memSkappa != -1 ) {
                              dcomplex alpha = factor;
                              dcomplex beta  = 1.0;

                              zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimR, &beta, memHeff, &dimL );
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
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoJ ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSRdown - TwoS2 );
            const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS2, TwoSR, TwoSL, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {
               int IRdown   = Irreps::directProd( IR, Irrep );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {
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

                           int size = dimR * dimRdown;
                           for ( int cnt = 0; cnt < size; cnt++ ) {
                              temp[ cnt ] = 0.0;
                           }
                           for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                              if ( Irrep == initBKUp->gIrrep( l_delta ) ) {
                                 dcomplex * LblockRight = LTright[ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                                 dcomplex prefact       = prob->gMxElement( l_gamma, l_delta, theindex, theindex );
                                 zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                              }
                           }

                           int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha = factor;
                              dcomplex beta  = 1.0;

                              zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimR, &beta, memHeff, &dimL );
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
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase               = phase( TwoSL + TwoSR + TwoJ + TwoSL + TwoSRdown + 1 - TwoS2 );
               const dcomplex factor1 = fase * sqrt( ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS2, TwoJdown, 1, TwoSL ) * Wigner::wigner6j( TwoJ, 1, TwoS2, TwoSRdown, TwoSL, TwoSR );

               dcomplex factor2 = 0.0;
               if ( TwoJ == TwoJdown ) {
                  fase    = phase( TwoSL + TwoSRdown + TwoJ + 3 + 2 * TwoS2 );
                  factor2 = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoJ, TwoSR, TwoSL, 1 );
               }

               for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

                  int IRdown   = Irreps::directProd( IR, Irrep );
                  int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                  if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {
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

                              int size = dimR * dimRdown;
                              for ( int cnt = 0; cnt < size; cnt++ ) {
                                 temp[ cnt ] = 0.0;
                              }
                              for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                                 if ( Irrep == initBKUp->gIrrep( l_delta ) ) {
                                    dcomplex * LblockRight = LTright[ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                                    dcomplex prefact       = factor1 * prob->gMxElement( l_alpha, theindex, theindex, l_delta );
                                    if ( TwoJ == TwoJdown ) { prefact += factor2 * prob->gMxElement( l_alpha, theindex, l_delta, theindex ); }
                                    zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                 }
                              }

                              int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = 1.0;
                                 dcomplex beta  = 1.0;

                                 zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimR, &beta, memHeff, &dimL );
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

   //4E3B
   if ( N1 == 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoJ ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase              = phase( TwoSL + TwoSRdown - TwoS2 + 3 );
            const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS2, TwoSR, TwoSL, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

               int IRdown   = Irreps::directProd( IR, Irrep );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {
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

                           int size = dimR * dimRdown;
                           for ( int cnt = 0; cnt < size; cnt++ ) {
                              temp[ cnt ] = 0.0;
                           }
                           for ( int l_delta = theindex + 1; l_delta < prob->gL(); l_delta++ ) {
                              if ( Irrep == initBKUp->gIrrep( l_delta ) ) {
                                 dcomplex * LblockRight = LTright[ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                                 dcomplex prefact       = prob->gMxElement( l_alpha, theindex, theindex, l_delta ) - 2 * prob->gMxElement( l_alpha, theindex, l_delta, theindex );
                                 zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                              }
                           }

                           int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha = factor;
                              dcomplex beta  = 1.0;

                              zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimR, &beta, memHeff, &dimL );
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
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( ( TwoJdown == 1 ) && abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase               = phase( TwoSL + TwoSR + TwoJdown + TwoSL + TwoSRdown + 1 - TwoS2 );
               const dcomplex factor1 = fase * sqrt( ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoJ, 1, TwoSL ) * Wigner::wigner6j( TwoJdown, 1, TwoS2, TwoSR, TwoSL, TwoSRdown );

               dcomplex factor2 = 0.0;
               if ( TwoJ == TwoJdown ) {
                  fase    = phase( TwoSL + TwoSR + TwoJ + 3 + 2 * TwoS2 );
                  factor2 = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSL, 1 );
               }

               for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

                  int IRdown   = Irreps::directProd( IR, Irrep );
                  int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                  if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {
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

                              int size = dimR * dimRdown;
                              for ( int cnt = 0; cnt < size; cnt++ ) {
                                 memHeff[ cnt ] = 0.0;
                              }
                              for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                                 if ( Irrep == initBKUp->gIrrep( l_beta ) ) {
                                    dcomplex * LblockRight = Lright[ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                                    dcomplex prefact       = factor1 * prob->gMxElement( l_gamma, theindex, theindex, l_beta );
                                    if ( TwoJ == TwoJdown ) { prefact += factor2 * prob->gMxElement( l_gamma, theindex, l_beta, theindex ); }
                                    zaxpy_( &size, &prefact, LblockRight, &inc, memHeff, &inc );
                                 }
                              }

                              int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = 1.0;
                                 dcomplex beta  = 1.0;

                                 zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, memHeff, &dimR, &beta, memHeff, &dimL );
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
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoJ ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR - TwoS2 + 3 );
            const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoSRdown, TwoSL, 1 );

            for ( int Irrep = 0; Irrep < ( initBKUp->getNumberOfIrreps() ); Irrep++ ) {

               int IRdown   = Irreps::directProd( IR, Irrep );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {
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

                           int size = dimR * dimRdown;
                           for ( int cnt = 0; cnt < size; cnt++ ) {
                              temp[ cnt ] = 0.0;
                           }
                           for ( int l_beta = theindex + 1; l_beta < prob->gL(); l_beta++ ) {
                              if ( Irrep == initBKUp->gIrrep( l_beta ) ) {
                                 dcomplex * LblockRight = Lright[ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                                 dcomplex prefact       = prob->gMxElement( l_gamma, theindex, theindex, l_beta ) - 2 * prob->gMxElement( l_gamma, theindex, l_beta, theindex );
                                 zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                              }
                           }

                           int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex alpha = factor;
                              dcomplex beta  = 1.0;
                              zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, memHeff, &dimR, &beta, memHeff, &dimL );
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

void CheMPS2::CSubSpaceExpander::addDiagram4L1and4L2spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Aright, CTensorOperator *** ATright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L1A.spin0
   if ( N1 == 1 ) {
      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSL >= 0 ) ) {

         int fase              = phase( TwoSL + TwoSR + 2 + TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSL, TwoSL, TwoSR );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {

            int IRdown   = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );
            int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );

            if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

               int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IRdown );
               if ( memSkappa != -1 ) {
                  dcomplex * blockA = ATright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
                  dcomplex beta     = 1.0;
                  dcomplex alpha    = factor;

                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockA, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4L1B.spin0
   if ( N1 == 2 ) {
      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSL >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR + 3 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int IRdown   = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockA = ATright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockA, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4L2A.spin0
   if ( N1 == 0 ) {
      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSL >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR + 2 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

               int IRdown   = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex beta  = 1.0;
                     dcomplex alpha = factor;

                     dcomplex * blockA = Aright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockA, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4L2B.spin0
   if ( N1 == 1 ) {
      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSL >= 0 ) ) {
         int fase              = phase( TwoSL + TwoSR + 3 + TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSL, TwoSL, TwoSR );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {
            int IRdown   = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );
            int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );

            if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

               int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IRdown );
               if ( memSkappa != -1 ) {
                  dcomplex alpha = factor;
                  dcomplex beta  = 1.0;

                  dcomplex * blockA = Aright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockA, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L1and4L2spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Bright, CTensorOperator *** BTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L1A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( 1 + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSL, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int IRdown   = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex alpha = factor;
                     dcomplex beta  = 1.0;

                     dcomplex * blockB = BTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockB, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4L1B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSR - TwoSRdown + TwoSL - TwoSL + TwoS2 - TwoJdown ); //bug fixed
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSL, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

                  int IRdown   = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );
                  int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );

                  if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSRdown, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockB = BTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                        zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockB, &dimR, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }
   }

   //4L2A.spin1
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( 1 + TwoS2 - TwoJdown );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSL, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

                  int IRdown   = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );
                  int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );

                  if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSRdown, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex alpha = factor;
                        dcomplex beta  = 1.0;

                        dcomplex * blockB = Bright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                        zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockB, &dimR, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }
   }

   //4L2B.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSRdown - TwoSR + TwoSL - TwoSL + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSL, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

               int IRdown   = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex beta  = 1.0;
                     dcomplex alpha = factor;

                     dcomplex * blockB = Bright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, blockB, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L3and4L4spin0Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Cright, CTensorOperator *** CTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L3A.spin0
   if ( N1 == 1 ) {
      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSL >= 0 ) ) {
         int fase              = phase( TwoSL + TwoSR + 1 + TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSL, TwoSL, TwoSR );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {

            int IRdown   = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );
            int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

            if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

               int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );
               if ( memSkappa != -1 ) {
                  dcomplex beta  = 1.0;
                  dcomplex alpha = factor;
                  dcomplex * ptr = Cright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );

                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }

   //4L3B.spin0
   if ( N1 == 2 ) {
      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSL >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR + 2 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

               int IRdown   = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex beta  = 1.0;
                     dcomplex alpha = factor;

                     dcomplex * ptr = Cright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4L4A.spin0
   if ( N1 == 0 ) {
      int TwoJstart = ( ( TwoSL != TwoSR ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

      for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
         if ( ( abs( TwoSL - TwoSR ) <= TwoJdown ) && ( TwoSL >= 0 ) ) {
            int fase              = phase( TwoSL + TwoSR + 1 + TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSL, TwoSR );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

               int IRdown   = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex beta  = 1.0;
                     dcomplex alpha = factor;

                     dcomplex * ptr = CTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4L4B.spin0
   if ( N1 == 1 ) {
      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSL >= 0 ) ) {
         int fase              = phase( TwoSL + TwoSR + 2 + TwoS2 );
         const dcomplex factor = fase * sqrt( 0.5 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSL, TwoSL, TwoSR );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {

            int IRdown   = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );
            int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

            if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

               int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IRdown );
               if ( memSkappa != -1 ) {
                  dcomplex beta  = 1.0;
                  dcomplex alpha = factor;

                  dcomplex * ptr = CTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram4L3and4L4spin1Left( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Dright, CTensorOperator *** DTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimL     = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L3A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSL - TwoSL + 2 + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSL, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int IRdown   = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex beta  = 1.0;
                     dcomplex alpha = factor;

                     dcomplex * ptr = Dright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   //4L3B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSR - TwoSRdown + 1 + TwoS2 - TwoJdown );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSL, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

                  int IRdown   = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );
                  int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                  if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex beta  = 1.0;
                        dcomplex alpha = factor;

                        dcomplex * ptr = Dright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                        zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }
   }

   //4L4A.spin1
   if ( N1 == 0 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;

         for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
            if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
               int fase              = phase( TwoSL + 2 - TwoSL + TwoS2 - TwoJdown );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSL, TwoSL, 1, TwoJdown, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

                  int IRdown   = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );
                  int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                  if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex beta  = 1.0;
                        dcomplex alpha = factor;

                        dcomplex * ptr = DTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                        zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                     }
                  }
               }
            }
         }
      }
   }

   //4L4B.spin1
   if ( N1 == 1 ) {
      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
            int fase              = phase( TwoSR - TwoSRdown + 1 + TwoS2 - TwoJ );
            const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSL, 1, TwoJ, TwoS2 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {
               int IRdown   = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );
               int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

               if ( ( dimL > 0 ) && ( dimRdown > 0 ) ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     dcomplex beta  = 1.0;
                     dcomplex alpha = factor;

                     dcomplex * ptr = DTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}