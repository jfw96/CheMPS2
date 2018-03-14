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

void CheMPS2::CHeffNS_1S::addDiagram3Aand3D( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qleft, CTensorQT * QTleft, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );
   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1    = NR - NL;
   int TwoS  = ( N1 == 1 ) ? 1 : 0;
   int TwoS2 = 0;
   int TwoJ  = TwoS;

   int theindex = out->gIndex();
   int ILdown   = Irreps::directProd( IL, bk_up->gIrrep( theindex ) );

   int dimR   = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   // if ( N1 == 2 ) { //3A1A and 3D1
   //    for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

   //       int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
   //       if ( dimLdown > 0 ) {

   //          int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
   //          if ( memSkappa != -1 ) {
   //             // int fase        = phase( 2 * TwoSL + TwoSLdown + TwoSR + 3 );
   //             // dcomplex factor = sqrt( ( TwoSLdown + 1.0 ) / ( TwoSL + 1.0 ) ) * fase;
   //             int fase      = phase( TwoSL + TwoSR + 2 + TwoS2 );
   //             dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSLdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );
   //             dcomplex beta = 1.0; //add
   //             char notr     = 'N';

   //             dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
   //             int inc           = 1;
   //             int size          = dimLup * dimLdown;
   //             zcopy_( &size, BlockQ, &inc, temp, &inc );

   //             for ( int l_index = 0; l_index < theindex; l_index++ ) {
   //                if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
   //                   dcomplex alpha    = Prob->gMxElement( l_index, theindex, theindex, theindex );
   //                   dcomplex * BlockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
   //                   zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
   //                }
   //             }

   //             zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
   //          }
   //       }
   //    }
   // }

   if ( N1 == 2 ) { //3A1A and 3D1
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 ) {

            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     int fase        = phase( TwoSL + TwoSR + 2 + TwoS2 );
                     dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSLdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta   = 1.0; //add

                     dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     int inc           = 1;
                     int size          = dimLup * dimLdown;
                     zcopy_( &size, BlockQ, &inc, temp, &inc );

                     for ( int l_index = 0; l_index < theindex; l_index++ ) {
                        if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
                           dcomplex alpha    = Prob->gMxElement( l_index, theindex, theindex, theindex );
                           dcomplex * BlockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                           zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                        }
                     }

                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   // if ( N1 == 1 ) { //3A1B
   //    int dimLdown  = bk_down->gCurrentDim( theindex, NL + 1, TwoSR, ILdown );
   //    int memSkappa = in->gKappa( NL + 1, TwoSR, ILdown, NR, TwoSR, IR );
   //    if ( memSkappa != -1 ) {
   //       // int fase          = phase( 2 * TwoSL + 2 * TwoSR + 2 );
   //       // dcomplex factor   = fase;
   //       int fase = phase(TwoSL+TwoSR+1+0);
   //       double factor = sqrt((TwoSLdown+1)*(TwoS+1.0))*fase*Wigner::wigner6j(0,TwoS,1,TwoSL,TwoSLdown,TwoSR);
   //       dcomplex beta     = 1.0;
   //       dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSR, ILdown );
   //       zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
   //    }
   // }

   // if ( N1 == 1 ) { //3A1B
   //    for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
   //       if ( ( abs( TwoSLdown - TwoSR ) <= 0 ) && ( TwoSLdown >= 0 ) ) {
   //          int dimLdown  = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
   //          int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
   //          if ( memSkappa != -1 ) {
   //             int fase          = phase( TwoSL + TwoSR + 1 + 0 );
   //             dcomplex factor   = sqrt( ( TwoSLdown + 1 ) * ( TwoS + 1.0 ) ) * fase * Wigner::wigner6j( 0, TwoS, 1, TwoSL, TwoSLdown, TwoSR );
   //             dcomplex beta     = 1.0;
   //             char notr         = 'N';
   //             dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
   //             zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
   //          }
   //       }
   //    }
   // }

   if ( N1 == 1 ) { //3A1B
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {
            int dimLdown  = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
            int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSL + TwoSR + 1 + TwoS2 );
               dcomplex factor   = sqrt( ( TwoSLdown + 1 ) * ( TwoJ + 1.0 ) ) * fase * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSL, TwoSLdown, TwoSR );
               dcomplex beta     = 1.0;
               dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   // if ( N1 == 0 ) { //3A2A
   //    for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

   //       int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
   //       if ( dimLdown > 0 ) {

   //          int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
   //          if ( memSkappa != -1 ) {
   //             int fase          = phase( 2 * TwoSLdown + TwoSR + TwoSL + 2 );
   //             dcomplex factor   = fase;
   //             dcomplex beta     = 1.0;
   //             dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
   //             zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
   //          }
   //       }
   //    }
   // }

   if ( N1 == 0 ) { //3A2A
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 ) {

            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     int fase          = phase( TwoSLdown + TwoSR + 1 + TwoS2 );
                     dcomplex factor   = fase * sqrt( ( TwoSL + 1 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta     = 1.0;
                     dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   // if ( N1 == 1 ) { //3A2B ans 3D2
   //    int dimLdown  = bk_down->gCurrentDim( theindex, NL - 1, TwoSR, ILdown );
   //    int memSkappa = in->gKappa( NL - 1, TwoSR, ILdown, NR, TwoSR, IR );
   //    if ( memSkappa != -1 ) {
   //       int fase        = phase( 3 * TwoSR + TwoSL + 3 );
   //       dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) );
   //       dcomplex beta   = 1.0;

   //       dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSR, ILdown );
   //       int inc           = 1;
   //       int size          = dimLup * dimLdown;
   //       zcopy_( &size, BlockQ, &inc, temp, &inc );

   //       for ( int l_index = 0; l_index < theindex; l_index++ ) {
   //          if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
   //             dcomplex alpha    = Prob->gMxElement( l_index, theindex, theindex, theindex );
   //             dcomplex * BlockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSR, ILdown );
   //             zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
   //          }
   //       }

   //       zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
   //    }
   // }

   if ( N1 == 1 ) { //3A2B ans 3D2
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {
            int dimLdown  = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
            int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase        = phase( TwoSLdown + TwoSR + 2 + TwoS2 );
               dcomplex factor = fase * sqrt( ( TwoSL + 1 ) * ( TwoS + 1.0 ) ) * Wigner::wigner6j( TwoS2, TwoS, 1, TwoSL, TwoSLdown, TwoSR );
               dcomplex beta   = 1.0;

               dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
               int inc           = 1;
               int size          = dimLup * dimLdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
                     dcomplex alpha    = Prob->gMxElement( l_index, theindex, theindex, theindex );
                     dcomplex * BlockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram3C( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qleft, CTensorQT ** QTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

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

   int N2    = 0;
   int TwoJ  = TwoS;
   int TwoS2 = 0;

   int theindex = out->gIndex();

   int dimRup = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   // //First do 3C1
   // for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
   //    for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //       if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //          int fase              = phase( TwoSLdown + TwoSR + 3 * TwoS + 1 );
   //          const dcomplex factor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSLdown, 1 );

   //          for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   //             int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
   //             int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //             int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );
   //             if ( memSkappa != -1 ) {
   //                int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
   //                int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                dcomplex * Qblock = Qleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
   //                dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );

   //                dcomplex beta  = 0.0; //set
   //                dcomplex alpha = factor;
   //                zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Qblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

   //                beta  = 1.0; //add
   //                alpha = 1.0;
   //                zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Lblock, &dimRup, &beta, memHeff, &dimLup );
   //             }
   //          }
   //       }
   //    }
   // }

   //First do 3C1
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase              = phase( TwoSLdown + TwoSR + TwoJ + 1 + ( ( N1 == 1 ) ? 2 : 0 ) + ( ( N2 == 1 ) ? 2 : 0 ) );
            const dcomplex factor = fase * sqrt( ( TwoSLdown + 1 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

               int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
               int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
               int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );
               if ( memSkappa != -1 ) {
                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                  dcomplex * Qblock = Qleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                  dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );

                  dcomplex beta  = 0.0; //set
                  dcomplex alpha = factor;
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Qblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

                  beta  = 1.0; //add
                  alpha = 1.0;
                  zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Lblock, &dimRup, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }

   // //Then do 3C2
   // for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
   //    for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //       if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //          int fase            = phase( TwoSL + TwoSRdown + 3 * TwoS + 1 );
   //          const double factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSLdown, 1 );

   //          for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   //             int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
   //             int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //             int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );
   //             if ( memSkappa != -1 ) {
   //                int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
   //                int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

   //                dcomplex * Qblock = QTleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
   //                dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );

   //                dcomplex beta  = 0.0; //set
   //                dcomplex alpha = factor;
   //                zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Qblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

   //                beta  = 1.0; //add
   //                alpha = 1.0;
   //                zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Lblock, &dimRup, &beta, memHeff, &dimLup );
   //             }
   //          }
   //       }
   //    }
   // }

   //Then do 3C2
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase            = phase( TwoSL + TwoSRdown + TwoJ + 1 + ( ( N1 == 1 ) ? 2 : 0 ) + ( ( N2 == 1 ) ? 2 : 0 ) );
            const double factor = fase * sqrt( ( TwoSL + 1 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

               int ILdown    = Irreps::directProd( IL, bk_down->gIrrep( l_index ) );
               int IRdown    = Irreps::directProd( IR, bk_down->gIrrep( l_index ) );
               int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );
               if ( memSkappa != -1 ) {
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                  dcomplex * Qblock = QTleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                  dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );

                  dcomplex beta  = 0.0; //set
                  dcomplex alpha = factor;
                  zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Qblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

                  beta  = 1.0; //add
                  alpha = 1.0;
                  zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Lblock, &dimRup, &beta, memHeff, &dimLup );
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram3Kand3F( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qright, CTensorQT * QTright, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );
   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int theindex = out->gIndex();
   int IRdown   = Irreps::directProd( IR, bk_up->gIrrep( theindex ) );

   int ILdown = Irreps::directProd( IL, bk_up->gIrrep( theindex ) );
   int TwoS2  = 0;
   int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;
   int TwoJ = TwoS;
   int N2   = 0;

   // if ( N1 == 1 ) { //3K1A
   //    int dimRdown  = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSL, IRdown );
   //    int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSL, IRdown );
   //    if ( memSkappa != -1 ) {
   //       int fase          = phase( 2 * TwoSL + 2 * TwoSR + 2 );
   //       dcomplex factor   = fase;
   //       dcomplex beta     = 1.0; //add
   //       dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSL, IRdown );
   //       zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
   //    }
   // }

   if ( N1 == 1 ) { //3K1A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSL + TwoSR + TwoJ + 2 * TwoS2 );
               dcomplex factor   = sqrt( ( TwoJ + 1 ) * ( TwoSR + 1.0 ) ) * fase * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSR, TwoSRdown, TwoSL );
               dcomplex beta     = 1.0; //add
               dcomplex * BlockQ = Qright->gStorage( NR - 1, TwoSRdown, IRdown, NR, TwoSR, IR );
               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }

   // if ( N1 == 2 ) { //3K1B and 3F1
   //    for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

   //       int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
   //       if ( dimRdown > 0 ) {

   //          int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );
   //          if ( memSkappa != -1 ) {
   //             int fase        = phase( 3 * TwoSR + TwoSRdown + 3 );
   //             dcomplex factor = sqrt( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
   //             dcomplex beta   = 1.0; //add

   //             dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //             int inc           = 1;
   //             int size          = dimRup * dimRdown;
   //             zcopy_( &size, BlockQ, &inc, temp, &inc );

   //             for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {
   //                if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
   //                   dcomplex alpha    = Prob->gMxElement( theindex, theindex, theindex, l_index );
   //                   dcomplex * BlockL = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                   zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
   //                }
   //             }

   //             zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
   //          }
   //       }
   //    }
   // }

   if ( N1 == 2 ) { //3K1B and 3F1
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int fase        = phase( TwoSL + TwoSR + TwoJdown + 1 + 2 * TwoS2 );
                     dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSR + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta   = 1.0; //add

                     dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     int inc           = 1;
                     int size          = dimRup * dimRdown;
                     zcopy_( &size, BlockQ, &inc, temp, &inc );

                     for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {
                        if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
                           dcomplex alpha    = Prob->gMxElement( theindex, theindex, theindex, l_index );
                           dcomplex * BlockL = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                           zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                        }
                     }

                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   // if ( N1 == 0 ) { //3K2A
   //    for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

   //       int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
   //       if ( dimRdown > 0 ) {

   //          int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
   //          if ( memSkappa != -1 ) {
   //             int fase          = phase( 2 * TwoSR + 2 * TwoSRdown + 3 );
   //             dcomplex factor   = ( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
   //             dcomplex beta     = 1.0; //add
   //             dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //             zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
   //          }
   //       }
   //    }
   // }

   if ( N1 == 0 ) { //3K2A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int fase          = phase( TwoSL + TwoSRdown + TwoJdown + 2 * TwoS2 );
                     dcomplex factor   = sqrt( ( TwoJdown + 1 ) * ( TwoSRdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta     = 1.0; //add
                     dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   // if ( N1 == 1 ) { //3K2B and 3F2
   //    for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //       if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
   //          int dimRdown  = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
   //          int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
   //          if ( memSkappa != -1 ) {
   //             int fase        = phase( 3 * TwoSRdown + TwoSR + 3 );
   //             dcomplex factor = sqrt( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) ) * fase;
   //             dcomplex beta   = 1.0; //add

   //             dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //             int inc           = 1;
   //             int size          = dimRup * dimRdown;
   //             zcopy_( &size, BlockQ, &inc, temp, &inc );

   //             for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {
   //                if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
   //                   dcomplex alpha    = Prob->gMxElement( theindex, theindex, theindex, l_index );
   //                   dcomplex * BlockL = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                   zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
   //                }
   //             }

   //             zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
   //          }
   //       }
   //    }
   // }

   if ( N1 == 1 ) { //3K2B and 3F2
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
            int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase        = phase( TwoSL + TwoSRdown + TwoJ + 1 + 2 * TwoS2 );
               dcomplex factor = sqrt( ( TwoJ + 1 ) * ( TwoSRdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSR, TwoSRdown, TwoSL );
               dcomplex beta   = 1.0; //add

               dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               int inc           = 1;
               int size          = dimRup * dimRdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {
                  if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
                     dcomplex alpha    = Prob->gMxElement( theindex, theindex, theindex, l_index );
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

void CheMPS2::CHeffNS_1S::addDiagram3J( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qright, CTensorQT ** QTright, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

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

   int dimRup = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   //First do 3J2
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase            = phase( TwoSLdown + TwoSR + 3 * TwoS + 1 );
            const double factor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_q( Prob->gL(), l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {

                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                     dcomplex * Lblock = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * Qblock = Qright[ theindex - l_index ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );

                     dcomplex beta  = 0.0; //set
                     dcomplex alpha = factor;
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

                     beta  = 1.0; //add
                     alpha = 1.0;
                     zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Qblock, &dimRup, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //Then do 3J1
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase            = phase( TwoSL + TwoSRdown + 3 * TwoS + 1 );
            const double factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_q( Prob->gL(), l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                     dcomplex * Lblock = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * Qblock = QTright[ theindex - l_index ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );

                     dcomplex beta  = 0.0; //set
                     dcomplex alpha = factor;
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLup, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

                     beta  = 1.0; //add
                     alpha = 1.0;
                     zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Qblock, &dimRup, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}