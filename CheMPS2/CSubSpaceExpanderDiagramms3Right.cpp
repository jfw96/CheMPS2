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

void CheMPS2::CSubSpaceExpander::addDiagram3Aand3DRight( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qleft, CTensorQT * QTleft, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

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
   int ILdown   = Irreps::directProd( IL, initBKUp->gIrrep( theindex ) );

   int dimLup = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR   = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   if ( N1 == 2 ) { //3A1A and 3D1
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         int dimLdown = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

         if ( dimLdown > 0 && dimLup > 0) {
            int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               int fase        = phase( 2 * TwoSL + TwoSLdown + TwoSR + 3 );
               dcomplex factor = sqrt( ( TwoSLdown + 1.0 ) / ( TwoSL + 1.0 ) ) * fase;
               dcomplex beta   = 1.0; //add
               char notr       = 'N';

               dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
               int inc           = 1;
               int size          = dimLup * dimLdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  if ( initBKUp->gIrrep( l_index ) == initBKUp->gIrrep( theindex ) ) {
                     dcomplex alpha    = prob->gMxElement( l_index, theindex, theindex, theindex );
                     dcomplex * BlockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   if ( N1 == 1 ) { //3A1B
      int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 1, TwoSR, ILdown );
      int memSkappa = in->gKappa( NL + 1, TwoSR, ILdown, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         int fase          = phase( 2 * TwoSL + 2 * TwoSR + 2 );
         dcomplex factor   = fase;
         dcomplex beta     = 1.0;
         dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSR, ILdown );
         zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
      }
   }

   if ( N1 == 0 ) { //3A2A
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 && dimLup > 0) {

            int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase          = phase( 2 * TwoSLdown + TwoSR + TwoSL + 2 );
               dcomplex factor   = fase;
               dcomplex beta     = 1.0;
               dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   if ( N1 == 1 ) { //3A2B ans 3D2
      int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 1, TwoSR, ILdown );
      int memSkappa = in->gKappa( NL - 1, TwoSR, ILdown, NR, TwoSR, IR );
      if ( memSkappa != -1 && dimLdown > 0 && dimLup > 0) {
         int fase        = phase( 3 * TwoSR + TwoSL + 3 );
         dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) );
         dcomplex beta   = 1.0;

         dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSR, ILdown );
         int inc           = 1;
         int size          = dimLup * dimLdown;
         zcopy_( &size, BlockQ, &inc, temp, &inc );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {
            if ( initBKUp->gIrrep( l_index ) == initBKUp->gIrrep( theindex ) ) {
               dcomplex alpha    = prob->gMxElement( l_index, theindex, theindex, theindex );
               dcomplex * BlockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSR, ILdown );
               zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
            }
         }

         zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram3CRight( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qleft, CTensorQT ** QTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

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

   int dimLup = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR   = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
   //First do 3C1
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      if ( ( abs( TwoSLdown - TwoSR ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
         int fase              = phase( TwoSLdown + TwoSR + 3 * TwoS + 1 );
         const dcomplex factor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSR, TwoSLdown, 1 );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
            int ILdown    = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
            int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0; //set
               dcomplex alpha = factor;

               dcomplex * Qblock = Qleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Qblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   //Then do 3C2
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      if ( ( abs( TwoSLdown - TwoSR ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
         int fase            = phase( TwoSL + TwoSR + 3 * TwoS + 1 );
         const double factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSR, TwoSLdown, 1 );

         for ( int l_index = theindex + 1; l_index < prob->gL(); l_index++ ) {
            int ILdown    = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
            int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );

            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0; //set
               dcomplex alpha = factor;

               dcomplex * Qblock = QTleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Qblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram3Kand3FRight( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qright, CTensorQT * QTright, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int theindex = out->gIndex();

   int ILdown = Irreps::directProd( IL, initBKUp->gIrrep( theindex ) );
   int TwoS2  = 0;

   int dimL = initBKDown->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   if ( N1 == 1 ) { //3K1A
      int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSL, IR );
      if ( memSkappa != -1 ) {
         int fase        = phase( 2 * TwoSL + 2 * TwoSR + 2 );
         dcomplex factor = fase;
         dcomplex beta   = 1.0; //add
         int inc         = 1;
         int dim         = dimL * dimR;

         dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
         zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
      }
   }

   if ( N1 == 2 ) { //3K1B and 3F1

      if ( dimR > 0 ) {
         int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );

         if ( memSkappa != -1 ) {
            int fase        = phase( 3 * TwoSR + TwoSR + 3 );
            dcomplex factor = sqrt( ( TwoSR + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
            dcomplex beta   = 1.0; //add
            int inc         = 1;
            int dim         = dimL * dimR;

            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
            zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
         }
      }
   }

   if ( N1 == 0 ) { //3K2A

      if ( dimR > 0 ) {

         int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );
         if ( memSkappa != -1 ) {
            int fase        = phase( 2 * TwoSR + 2 * TwoSR + 3 );
            dcomplex factor = ( ( TwoSR + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
            int inc         = 1;
            int dim         = dimL * dimR;

            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
            zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
         }
      }
   }

   if ( N1 == 1 ) { //3K2B and 3F2
      if ( ( abs( TwoSL - TwoSR ) <= TwoS2 ) && ( TwoSR >= 0 ) ) {

         int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );
         if ( memSkappa != -1 ) {
            int fase        = phase( 3 * TwoSR + TwoSR + 3 );
            dcomplex factor = sqrt( ( TwoSR + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
            int inc         = 1;
            int dim         = dimL * dimR;

            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
            zaxpy_( &dim, &factor, BlockS, &inc, memHeff, &inc );
         }
      }
   }
}

void CheMPS2::CSubSpaceExpander::addDiagram3JRight( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qright, CTensorQT ** QTright, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

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

   int dimR   = sseBKDown->gCurrentDim( theindex + 1, NR, TwoSR, IR );
   int dimLup = initBKUp->gCurrentDim( theindex, NL, TwoSL, IL );

   //First do 3J2
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      if ( ( abs( TwoSLdown - TwoSR ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
         int fase            = phase( TwoSLdown + TwoSR + 3 * TwoS + 1 );
         const double factor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSR + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSR, TwoSLdown, 1 );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {

            int ILdown    = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
            int dimLdown  = initBKDown->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
            int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0; //set
               dcomplex alpha = factor;

               dcomplex * Lblock = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Lblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   //Then do 3J1
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      if ( ( abs( TwoSLdown - TwoSR ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSR >= 0 ) ) {
         int fase            = phase( TwoSL + TwoSR + 3 * TwoS + 1 );
         const double factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSR, TwoSLdown, 1 );

         for ( int l_index = 0; l_index < theindex; l_index++ ) {

            int ILdown    = Irreps::directProd( IL, initBKUp->gIrrep( l_index ) );
            int dimLdown = initBKDown->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
            int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               dcomplex beta  = 1.0; //set
               dcomplex alpha = factor;

               dcomplex * Lblock = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Lblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }
}