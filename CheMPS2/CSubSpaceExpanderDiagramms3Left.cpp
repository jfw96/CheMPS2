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

void CheMPS2::CSubSpaceExpander::addDiagram3Aand3DLeft( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qleft, CTensorQT * QTleft, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = out->gIndex();

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( N1 == 2 ) { //3A1A and 3D1
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         int fase        = phase( 2 * TwoSL + TwoSL + TwoSR + 3 );
         dcomplex factor = sqrt( ( TwoSL + 1.0 ) / ( TwoSL + 1.0 ) ) * fase;
         int dim         = dimL * dimR;
         int inc         = 1;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
      }
   }

   if ( N1 == 1 ) { //3A1B
      int memSkappa = in->gKappa( NL, TwoSR, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         int fase        = phase( 2 * TwoSL + 2 * TwoSR + 2 );
         int inc         = 1;
         int dim         = dimL * dimR;
         dcomplex factor = fase;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
      }
   }

   if ( N1 == 0 ) { //3A2A
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         int fase        = phase( 2 * TwoSL + TwoSR + TwoSL + 2 );
         dcomplex factor = fase;
         int inc         = 1;
         int dim         = dimL * dimR;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
      }
   }

   if ( N1 == 1 ) { //3A2B ans 3D2
      int memSkappa = in->gKappa( NL, TwoSR, IL, NR, TwoSR, IR );

      if ( memSkappa != -1 ) {
         int fase        = phase( 3 * TwoSR + TwoSL + 3 );
         dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) );
         dcomplex beta   = 1.0;
         int inc         = 1;
         int dim         = dimL * dimR;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram3CLeft( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qleft, CTensorQT ** QTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = out->gIndex();

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   //First do 3C1
   for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
      if ( ( abs( TwoSL - TwoSRdown ) <= TwoS ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
         int fase              = phase( TwoSL + TwoSR + 3 * TwoS + 1 );
         const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSL, 1 );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
            int IRdown    = Irreps::directProd( IR, initBKDown->gIrrep( l_index ) );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0;
               dcomplex alpha = factor;

               dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, BlockS, &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }

   //Then do 3C2
   for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
      if ( ( abs( TwoSL - TwoSRdown ) <= TwoS ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
         int fase            = phase( TwoSL + TwoSRdown + 3 * TwoS + 1 );
         const double factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSL, 1 );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
            int IRdown    = Irreps::directProd( IR, initBKDown->gIrrep( l_index ) );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0;
               dcomplex alpha = factor;

               dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, temp, &dimL, Lblock, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram3Kand3FLeft( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qright, CTensorQT * QTright, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );
   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int theindex = out->gIndex();
   int IRdown   = Irreps::directProd( IR, initBKDown->gIrrep( theindex ) );

   int ILdown = Irreps::directProd( IL, initBKDown->gIrrep( theindex ) );
   int TwoS2  = 0;

   int dimL   = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   if ( N1 == 1 ) { //3K1A
      int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSL, IRdown );
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSL, IRdown );

      if ( memSkappa != -1 ) {
         int fase          = phase( 2 * TwoSL + 2 * TwoSR + 2 );
         dcomplex factor   = fase;
         dcomplex beta     = 1.0; //add

         dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSL, IRdown );
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
      }
   }

   if ( N1 == 2 ) { //3K1B and 3F1
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase        = phase( 3 * TwoSR + TwoSRdown + 3 );
               dcomplex factor = sqrt( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
               dcomplex beta   = 1.0; //add

               dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
               int inc           = 1;
               int size          = dimRup * dimRdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  if ( initBKUp->gIrrep( l_index ) == initBKUp->gIrrep( theindex ) ) {
                     dcomplex alpha    = prob->gMxElement( theindex, theindex, theindex, l_index );
                     dcomplex * BlockL = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }

   if ( N1 == 0 ) { //3K2A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase          = phase( 2 * TwoSR + 2 * TwoSRdown + 3 );
               dcomplex factor   = ( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
               dcomplex beta     = 1.0; //add
               dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }

   if ( N1 == 1 ) { //3K2B and 3F2
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
            
            if ( memSkappa != -1 ) {
               int fase        = phase( 3 * TwoSRdown + TwoSR + 3 );
               dcomplex factor = sqrt( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
               dcomplex beta   = 1.0; //add

               dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               int inc           = 1;
               int size          = dimRup * dimRdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
                  if ( initBKUp->gIrrep( l_index ) == initBKUp->gIrrep( theindex ) ) {
                     dcomplex alpha    = prob->gMxElement( theindex, theindex, theindex, l_index );
                     dcomplex * BlockL = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram3JLeft( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qright, CTensorQT ** QTright, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = out->gIndex();

   int dimL = sseBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = initBKUp->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   //First do 3J2
   for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
      if ( ( abs( TwoSL - TwoSRdown ) <= TwoS ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
         int fase            = phase( TwoSL + TwoSR + 3 * TwoS + 1 );
         const double factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSL, 1 );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {
            int IRdown    = Irreps::directProd( IR, initBKDown->gIrrep( l_index ) );
            int dimRdown  = initBKDown->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0; //set
               dcomplex alpha = factor;

               dcomplex * Qblock = Qright[ theindex - l_index ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, temp, &dimL, Qblock, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }

   //Then do 3J1
   for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
      if ( ( abs( TwoSL - TwoSRdown ) <= TwoS ) && ( TwoSL >= 0 ) && ( TwoSRdown >= 0 ) ) {
         int fase            = phase( TwoSL + TwoSRdown + 3 * TwoS + 1 );
         const double factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSL, 1 );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {
            int IRdown    = Irreps::directProd( IR, initBKUp->gIrrep( l_index ) );
            int dimRdown = initBKDown->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );

            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0; //set
               dcomplex alpha = factor;

               dcomplex * Qblock = QTright[ theindex - l_index ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, temp, &dimL, Qblock, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }
}