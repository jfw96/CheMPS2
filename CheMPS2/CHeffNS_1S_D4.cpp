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

void CheMPS2::CHeffNS_1S::addDiagram4B1and4B2spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Aleft, CTensorOperator *** ATleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

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

               // //original
               // int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
               // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               //notes
               int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               // // simplified
               // int fase              = phase( 3 + TwoSL + 2 * TwoSR + TwoSRdown );
               // const dcomplex factor = fase * sqrt( 0.5 ) * sqrt( ( TwoSRdown + 1.0 ) / ( TwoSR + 1.0 ) );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * Ablock = ATleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
                        alpha             = factor;
                        beta              = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
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

            // // original
            // int fase              = phase( TwoSR + TwoSL + 2 + TwoJ + 2 * TwoS2 );
            // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            // original
            int fase              = phase( TwoSR + TwoSL + 2 + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            // // simplified
            // int fase              = phase( 4 + 2 * TwoSL + 2 * TwoSR );
            // const dcomplex factor = fase * sqrt( 0.5 );

            for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {

                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                     dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     dcomplex alpha    = 1.0;
                     dcomplex beta     = 0.0; //set
                     zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                     dcomplex * Ablock = ATleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
                     alpha             = factor;
                     beta              = 1.0; //add
                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //4B2A.spin0
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {

            // // original
            // int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
            // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            // notes
            int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {

                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                     dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     dcomplex alpha    = 1.0;
                     dcomplex beta     = 0.0; //set
                     zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                     dcomplex * Ablock = Aleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                     alpha             = factor;
                     beta              = 1.0; //add
                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                  }
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

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * Ablock = Aleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                        alpha             = factor;
                        beta              = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //    //4B1A.spin0
   //    if ( N1 == 0 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

   //          int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //          for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //             if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
   //                const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * Ablock = ATleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
   //                         alpha             = factor;
   //                         beta              = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B1B.spin0
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {

   //             int fase              = phase( TwoSR + TwoSL + 2 + TwoJ + 2 * TwoS2 );
   //             const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

   //             for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                {
   //                   int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                   int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                   int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

   //                   if ( memSkappa != -1 ) {

   //                      int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
   //                      int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                      dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                      dcomplex alpha    = 1.0;
   //                      dcomplex beta     = 0.0; //set
   //                      zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                      dcomplex * Ablock = ATleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
   //                      alpha             = factor;
   //                      beta              = 1.0; //add
   //                      zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B2A.spin0
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {

   //             int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
   //             const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

   //             for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                {
   //                   int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                   int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                   int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

   //                   if ( memSkappa != -1 ) {

   //                      int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
   //                      int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                      dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                      dcomplex alpha    = 1.0;
   //                      dcomplex beta     = 0.0; //set
   //                      zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                      dcomplex * Ablock = Aleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
   //                      alpha             = factor;
   //                      beta              = 1.0; //add
   //                      zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B2B.spin0
   //    if ( N1 == 2 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

   //          int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //          for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //             if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = phase( TwoSRdown + TwoSL + 2 + TwoJdown + 2 * TwoS2 );
   //                const dcomplex factor = fase * sqrt( 0.5 * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Aleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, out->gStorage() + out->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * Ablock = Aleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
   //                         alpha             = factor;
   //                         beta              = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Ablock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }
}

void CheMPS2::CHeffNS_1S::addDiagram4B1and4B2spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Bleft, CTensorOperator *** BTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B1A.spin1
   if ( N1 == 0 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  // // original
                  // int fase              = ( TwoS2 == 0 ) ? 1 : -1;
                  // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1 ) * ( TwoSL + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  // notes
                  int fase              = ( TwoS2 == 0 ) ? 1 : -1;
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoSL + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                        int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                           dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                           dcomplex alpha    = 1.0;
                           dcomplex beta     = 0.0; //set
                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                           dcomplex * Bblock = BTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                           alpha             = factor;
                           beta              = 1.0; //add
                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4B1B.spin1
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               //  // original
               // int fase              = phase( TwoSR - TwoSRdown + TwoSL + 3 - TwoSLdown + 2 * TwoS2 );
               // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1 ) * ( TwoSL + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               // original
               int fase              = phase( TwoSR - TwoSRdown + TwoSL + 3 - TwoSLdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoSL + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * Bblock = BTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                        alpha             = factor;
                        beta              = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4B2A.spin1
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               // // original
               // int fase              = ( TwoS2 == 0 ) ? 1 : -1;
               // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoSLdown + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               // notes
               int fase              = ( TwoS2 == 0 ) ? 1 : -1;
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * Bblock = Bleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                        alpha             = factor;
                        beta              = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4B2B.spin1
   if ( N1 == 2 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  // // original
                  // int fase              = phase( TwoSLdown + 3 - TwoSL + TwoSRdown - TwoSR + 2 * TwoS2 );
                  // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoJdown + 1 ) * ( TwoSLdown + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  // notes
                  int fase              = phase( TwoSLdown + 3 - TwoSL + TwoSRdown - TwoSR + 2 * TwoS2 );
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                        int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                           dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                           dcomplex alpha    = 1.0;
                           dcomplex beta     = 0.0; //set
                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                           dcomplex * Bblock = Bleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                           alpha             = factor;
                           beta              = 1.0; //add
                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //    //4B1A.spin1
   //    if ( N1 == 0 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

   //             int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //             for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //                if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                   int fase              = ( TwoS2 == 0 ) ? 1 : -1;
   //                   const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

   //                   for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                      if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                      {
   //                         int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                         int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                         int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

   //                         if ( memSkappa != -1 ) {

   //                            int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
   //                            int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                            dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                            dcomplex alpha    = 1.0;
   //                            dcomplex beta     = 0.0; //set
   //                            zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                            dcomplex * Bblock = BTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
   //                            alpha             = factor;
   //                            beta              = 1.0; //add
   //                            zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                         }
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B1B.spin1
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
   //             if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = phase( TwoSR - TwoSRdown + TwoSL + 3 - TwoSLdown + 2 * TwoS2 );
   //                const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * Bblock = BTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
   //                         alpha             = factor;
   //                         beta              = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B2A.spin1
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
   //             if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = ( TwoS2 == 0 ) ? 1 : -1;
   //                const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * Bblock = Bleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
   //                         alpha             = factor;
   //                         beta              = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B2B.spin1
   //    if ( N1 == 2 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

   //             int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //             for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //                if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                   int fase              = phase( TwoSLdown + 3 - TwoSL + TwoSRdown - TwoSR + 2 * TwoS2 );
   //                   const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

   //                   for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                      if ( MPIchemps2::owner_absigma( theindex, l_index ) == MPIRANK )
   // #endif
   //                      {
   //                         int ILdown    = Irreps::directProd( IL, Bleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                         int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                         int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

   //                         if ( memSkappa != -1 ) {

   //                            int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
   //                            int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                            dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                            dcomplex alpha    = 1.0;
   //                            dcomplex beta     = 0.0; //set
   //                            zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                            dcomplex * Bblock = Bleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
   //                            alpha             = factor;
   //                            beta              = 1.0; //add
   //                            zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, Bblock, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                         }
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }
}

void CheMPS2::CHeffNS_1S::addDiagram4B3and4B4spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Cleft, CTensorOperator *** CTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B3A.spin0
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {

            // // original
            // int fase              = phase( TwoSR + TwoSL + TwoJ + 2 * TwoS2 );
            // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            // notes
            int fase              = phase( TwoSR + TwoSL + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {

                     int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                     dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     dcomplex alpha    = 1.0;
                     dcomplex beta     = 0.0; //set
                     zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                     dcomplex * ptr = CTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

                     alpha = factor;
                     beta  = 1.0; //add
                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                  }
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

               // // original
               // int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
               // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSR + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               // notes
               int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * ptr = CTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

                        alpha = factor;
                        beta  = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
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

               // // original
               // int fase              = phase( TwoSRdown + TwoSL + TwoJdown + 2 * TwoS2 );
               // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1 ) * ( TwoJdown + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               // notes
               int fase              = phase( TwoSRdown + TwoSL + TwoJdown + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 0.5 * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * ptr = Cleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

                        alpha = factor;
                        beta  = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
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

            // // original
            // int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
            // const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1 ) * ( TwoJ + 1 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            // notes
            int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
            const dcomplex factor = fase * sqrt( 0.5 * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

            for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

                  if ( memSkappa != -1 ) {

                     int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                     dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     dcomplex alpha    = 1.0;
                     dcomplex beta     = 0.0; //set
                     zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                     dcomplex * ptr = Cleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

                     alpha = factor;
                     beta  = 1.0; //add
                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //    //4B3A.spin0
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {

   //             int fase              = phase( TwoSR + TwoSL + TwoJ + 2 * TwoS2 );
   //             const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

   //             for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                {
   //                   int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                   int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                   int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

   //                   if ( memSkappa != -1 ) {

   //                      int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
   //                      int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                      dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                      dcomplex alpha    = 1.0;
   //                      dcomplex beta     = 0.0; //set
   //                      zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                      dcomplex * ptr = CTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

   //                      alpha = factor;
   //                      beta  = 1.0; //add
   //                      zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B3B.spin0
   //    if ( N1 == 2 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

   //          int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //          for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //             if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = phase( TwoSR + TwoSL + 1 + TwoJdown + 2 * TwoS2 );
   //                const dcomplex factor = fase * sqrt( 0.5 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR - 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * ptr = CTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

   //                         alpha = factor;
   //                         beta  = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B4A.spin0
   //    if ( N1 == 0 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

   //          int TwoJstart = ( ( TwoSL != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //          for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //             if ( ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = phase( TwoSRdown + TwoSL + TwoJdown + 2 * TwoS2 );
   //                const dcomplex factor = fase * sqrt( 0.5 ) * sqrt( ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * ptr = Cleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

   //                         alpha = factor;
   //                         beta  = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B4B.spin0
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {

   //             int fase              = phase( TwoSRdown + TwoSL + 1 + TwoJ + 2 * TwoS2 );
   //             const dcomplex factor = fase * sqrt( 0.5 ) * sqrt( ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJ, TwoS2, 1, TwoSRdown, TwoSR, TwoSL );

   //             for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                {
   //                   int ILdown    = Irreps::directProd( IL, Cleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                   int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                   int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR + 1, TwoSRdown, IRdown );

   //                   if ( memSkappa != -1 ) {

   //                      int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
   //                      int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                      dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                      dcomplex alpha    = 1.0;
   //                      dcomplex beta     = 0.0; //set
   //                      zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                      dcomplex * ptr = Cleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );

   //                      alpha = factor;
   //                      beta  = 1.0; //add
   //                      zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }
}

void CheMPS2::CHeffNS_1S::addDiagram4B3and4B4spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Dleft, CTensorOperator *** DTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4B3A.spin1
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               // // original
               // int fase              = phase( TwoSL - TwoSLdown + TwoSR - TwoSRdown + 3 + 2 * TwoS2 );
               // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1 ) * ( TwoJ + 1 ) * ( TwoSL + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               // notes
               int fase              = phase( TwoSL - TwoSLdown + TwoSR - TwoSRdown + 3 + 2 * TwoS2 );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * ptr = DTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

                        alpha = factor;
                        beta  = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //4B3B.spin1
   if ( N1 == 2 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  // // original
                  // int fase              = ( TwoS2 == 0 ) ? -1 : 1;
                  // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSR + 1 ) * ( TwoJdown + 1 ) * ( TwoSL + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  // notes
                  int fase              = ( TwoS2 == 0 ) ? -1 : 1;
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                        int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                           dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                           dcomplex alpha    = 1.0;
                           dcomplex beta     = 0.0; //set
                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                           dcomplex * ptr = DTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

                           alpha = factor;
                           beta  = 1.0; //add
                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4B4A.spin1
   if ( N1 == 0 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  // // original
                  // int fase              = phase( TwoSRdown - TwoSR + TwoSLdown - TwoSL + 3 + 2 * TwoS2 );
                  // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoJdown + 1 ) * ( TwoSLdown + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  //  notes
                  int fase              = phase( TwoSRdown - TwoSR + TwoSLdown - TwoSL + 3 + 2 * TwoS2 );
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

                  for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                        int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                           dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                           dcomplex alpha    = 1.0;
                           dcomplex beta     = 0.0; //set
                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                           dcomplex * ptr = Dleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

                           alpha = factor;
                           beta  = 1.0; //add
                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4B4B.spin1
   if ( N1 == 1 ) {

      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               // // original
               // int fase              = ( TwoS2 == 0 ) ? -1 : 1;
               // const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1 ) * ( TwoJ + 1 ) * ( TwoSLdown + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               // notes
               int fase              = ( TwoS2 == 0 ) ? -1 : 1;
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

               for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
#endif
                  {
                     int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
                     int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                     int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

                     if ( memSkappa != -1 ) {

                        int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                        dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                        dcomplex alpha    = 1.0;
                        dcomplex beta     = 0.0; //set
                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

                        dcomplex * ptr = Dleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

                        alpha = factor;
                        beta  = 1.0; //add
                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }

   //    //4B3A.spin1
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
   //             if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = phase( TwoSL - TwoSLdown + TwoSR - TwoSRdown + 3 + 2 * TwoS2 );
   //                const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoJ + 1.0 ) * ( TwoSL + 1 ) ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * ptr = DTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

   //                         alpha = factor;
   //                         beta  = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B3B.spin1
   //    if ( N1 == 2 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

   //             int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //             for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //                if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                   int fase              = ( TwoS2 == 0 ) ? -1 : 1;
   //                   const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1 ) ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

   //                   for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                      if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                      {
   //                         int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                         int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                         int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );

   //                         if ( memSkappa != -1 ) {

   //                            int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
   //                            int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

   //                            dcomplex * Lblock = LTright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
   //                            dcomplex alpha    = 1.0;
   //                            dcomplex beta     = 0.0; //set
   //                            zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                            dcomplex * ptr = DTleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

   //                            alpha = factor;
   //                            beta  = 1.0; //add
   //                            zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                         }
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B4A.spin1
   //    if ( N1 == 0 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

   //             int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
   //             for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
   //                if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                   int fase              = phase( TwoSRdown - TwoSR + TwoSLdown - TwoSL + 3 + 2 * TwoS2 );
   //                   const dcomplex factor = fase * sqrt( 3.0 * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSLdown, TwoSL, 1, TwoSRdown, TwoSR, 1, TwoJdown, TwoS2 );

   //                   for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                      if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                      {
   //                         int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                         int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                         int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

   //                         if ( memSkappa != -1 ) {

   //                            int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
   //                            int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                            dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                            dcomplex alpha    = 1.0;
   //                            dcomplex beta     = 0.0; //set
   //                            zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                            dcomplex * ptr = Dleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

   //                            alpha = factor;
   //                            beta  = 1.0; //add
   //                            zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                         }
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }

   //    //4B4B.spin1
   //    if ( N1 == 1 ) {

   //       for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
   //          for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
   //             if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

   //                int fase              = ( TwoS2 == 0 ) ? -1 : 1;
   //                const dcomplex factor = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSL, TwoSLdown, 1, TwoSR, TwoSRdown, 1, TwoJ, TwoS2 );

   //                for ( int l_index = theindex + 1; l_index < Prob->gL(); l_index++ ) {

   // #ifdef CHEMPS2_MPI_COMPILATION
   //                   if ( MPIchemps2::owner_cdf( Prob->gL(), theindex, l_index ) == MPIRANK )
   // #endif
   //                   {
   //                      int ILdown    = Irreps::directProd( IL, Dleft[ l_index - theindex ][ 0 ]->get_irrep() );
   //                      int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
   //                      int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );

   //                      if ( memSkappa != -1 ) {

   //                         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
   //                         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

   //                         dcomplex * Lblock = Lright[ l_index - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
   //                         dcomplex alpha    = 1.0;
   //                         dcomplex beta     = 0.0; //set
   //                         zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, Lblock, &dimRup, &beta, temp, &dimLdown );

   //                         dcomplex * ptr = Dleft[ l_index - theindex ][ 0 ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );

   //                         alpha = factor;
   //                         beta  = 1.0; //add
   //                         zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, ptr, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
   //                      }
   //                   }
   //                }
   //             }
   //          }
   //       }
   //    }
}

void CheMPS2::CHeffNS_1S::addDiagram4E( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

//4E1
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::owner_specific_diagram( Prob->gL(), MPI_CHEMPS2_4E1 ) == MPIRANK ) && ( N1 == 0 ) ) {
#else
   if ( N1 == 0 ) {
#endif

      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSL + TwoSR - TwoS2 );
               const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoSRdown, TwoSLdown, 1 );

               for ( int Irrep = 0; Irrep < ( bk_up->getNumberOfIrreps() ); Irrep++ ) {

                  int ILdown   = Irreps::directProd( IL, Irrep );
                  int IRdown   = Irreps::directProd( IR, Irrep );
                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                     bool isPossibleLeft = false;
                     for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                        if ( Irrep == bk_up->gIrrep( l_alpha ) ) { isPossibleLeft = true; }
                     }
                     bool isPossibleRight = false;
                     for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                        if ( Irrep == bk_up->gIrrep( l_beta ) ) { isPossibleRight = true; }
                     }
                     if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                        for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                           if ( Irrep == bk_up->gIrrep( l_alpha ) ) {

                              int size = dimRup * dimRdown;
                              for ( int cnt = 0; cnt < size; cnt++ ) {
                                 temp[ cnt ] = 0.0;
                              }
                              for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                                 if ( Irrep == bk_up->gIrrep( l_beta ) ) {
                                    dcomplex * LblockRight = Lright[ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                                    dcomplex prefact       = Prob->gMxElement( l_alpha, l_beta, theindex, theindex );
                                    zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                 }
                              }

                              int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = factor;
                                 dcomplex beta  = 0.0; //set
                                 zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, temp, &dimRup, &beta, temp2, &dimLdown );

                                 alpha                 = 1.0;
                                 beta                  = 1.0; //add
                                 dcomplex * LblockLeft = LTleft[ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                                 zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, LblockLeft, &dimLup, temp2, &dimLdown, &beta, memHeff, &dimLup );
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

//4E2
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::owner_specific_diagram( Prob->gL(), MPI_CHEMPS2_4E2 ) == MPIRANK ) && ( N1 == 2 ) ) {
#else
   if ( N1 == 2 ) {
#endif

      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSLdown + TwoSRdown - TwoS2 );
               const dcomplex factor = fase * sqrt( ( TwoSLdown + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS2, TwoSR, TwoSL, 1 );

               for ( int Irrep = 0; Irrep < ( bk_up->getNumberOfIrreps() ); Irrep++ ) {

                  int ILdown   = Irreps::directProd( IL, Irrep );
                  int IRdown   = Irreps::directProd( IR, Irrep );
                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                     bool isPossibleLeft = false;
                     for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                        if ( Irrep == bk_up->gIrrep( l_gamma ) ) { isPossibleLeft = true; }
                     }
                     bool isPossibleRight = false;
                     for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                        if ( Irrep == bk_up->gIrrep( l_delta ) ) { isPossibleRight = true; }
                     }
                     if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                        for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                           if ( Irrep == bk_up->gIrrep( l_gamma ) ) {

                              int size = dimRup * dimRdown;
                              for ( int cnt = 0; cnt < size; cnt++ ) {
                                 temp[ cnt ] = 0.0;
                              }
                              for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                                 if ( Irrep == bk_up->gIrrep( l_delta ) ) {
                                    dcomplex * LblockRight = LTright[ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                                    dcomplex prefact       = Prob->gMxElement( l_gamma, l_delta, theindex, theindex );
                                    zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                 }
                              }

                              int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = factor;
                                 dcomplex beta  = 0.0; //set
                                 zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, temp, &dimRup, &beta, temp2, &dimLdown );

                                 alpha                 = 1.0;
                                 beta                  = 1.0; //add
                                 dcomplex * LblockLeft = Lleft[ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                                 zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, LblockLeft, &dimLup, temp2, &dimLdown, &beta, memHeff, &dimLup );
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

//4E3A
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::owner_specific_diagram( Prob->gL(), MPI_CHEMPS2_4E3A ) == MPIRANK ) && ( N1 == 1 ) ) {
#else
   if ( N1 == 1 ) {
#endif

      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  int fase               = phase( TwoSL + TwoSR + TwoJ + TwoSLdown + TwoSRdown + 1 - TwoS2 );
                  const dcomplex factor1 = fase * sqrt( ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSRdown, TwoS2, TwoJdown, 1, TwoSLdown ) * Wigner::wigner6j( TwoJ, 1, TwoS2, TwoSRdown, TwoSL, TwoSR );

                  dcomplex factor2 = 0.0;
                  if ( TwoJ == TwoJdown ) {
                     fase    = phase( TwoSL + TwoSRdown + TwoJ + 3 + 2 * TwoS2 );
                     factor2 = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 1 );
                  }

                  for ( int Irrep = 0; Irrep < ( bk_up->getNumberOfIrreps() ); Irrep++ ) {

                     int ILdown   = Irreps::directProd( IL, Irrep );
                     int IRdown   = Irreps::directProd( IR, Irrep );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                        bool isPossibleLeft = false;
                        for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                           if ( Irrep == bk_up->gIrrep( l_alpha ) ) { isPossibleLeft = true; }
                        }
                        bool isPossibleRight = false;
                        for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                           if ( Irrep == bk_up->gIrrep( l_delta ) ) { isPossibleRight = true; }
                        }
                        if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                           for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                              if ( Irrep == bk_up->gIrrep( l_alpha ) ) {

                                 int size = dimRup * dimRdown;
                                 for ( int cnt = 0; cnt < size; cnt++ ) {
                                    temp[ cnt ] = 0.0;
                                 }
                                 for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                                    if ( Irrep == bk_up->gIrrep( l_delta ) ) {
                                       dcomplex * LblockRight = LTright[ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                                       dcomplex prefact       = factor1 * Prob->gMxElement( l_alpha, theindex, theindex, l_delta );
                                       if ( TwoJ == TwoJdown ) { prefact += factor2 * Prob->gMxElement( l_alpha, theindex, l_delta, theindex ); }
                                       zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                    }
                                 }

                                 int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );
                                 if ( memSkappa != -1 ) {
                                    dcomplex alpha = 1.0;
                                    dcomplex beta  = 0.0; //set
                                    zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, temp, &dimRup, &beta, temp2, &dimLdown );

                                    beta                  = 1.0; //add
                                    dcomplex * LblockLeft = LTleft[ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                                    zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, LblockLeft, &dimLup, temp2, &dimLdown, &beta, memHeff, &dimLup );
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
   }

//4E3B
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::owner_specific_diagram( Prob->gL(), MPI_CHEMPS2_4E3B ) == MPIRANK ) && ( N1 == 2 ) ) {
#else
   if ( N1 == 2 ) {
#endif

      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSL + TwoSRdown - TwoS2 + 3 );
               const dcomplex factor = fase * sqrt( ( TwoSL + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS2, TwoSR, TwoSL, 1 );

               for ( int Irrep = 0; Irrep < ( bk_up->getNumberOfIrreps() ); Irrep++ ) {

                  int ILdown   = Irreps::directProd( IL, Irrep );
                  int IRdown   = Irreps::directProd( IR, Irrep );
                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 1, TwoSRdown, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                     bool isPossibleLeft = false;
                     for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                        if ( Irrep == bk_up->gIrrep( l_alpha ) ) { isPossibleLeft = true; }
                     }
                     bool isPossibleRight = false;
                     for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                        if ( Irrep == bk_up->gIrrep( l_delta ) ) { isPossibleRight = true; }
                     }
                     if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                        for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                           if ( Irrep == bk_up->gIrrep( l_alpha ) ) {

                              int size = dimRup * dimRdown;
                              for ( int cnt = 0; cnt < size; cnt++ ) {
                                 temp[ cnt ] = 0.0;
                              }
                              for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                                 if ( Irrep == bk_up->gIrrep( l_delta ) ) {
                                    dcomplex * LblockRight = LTright[ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                                    dcomplex prefact       = Prob->gMxElement( l_alpha, theindex, theindex, l_delta ) - 2 * Prob->gMxElement( l_alpha, theindex, l_delta, theindex );
                                    zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                 }
                              }

                              int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 1, TwoSRdown, IRdown );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = factor;
                                 dcomplex beta  = 0.0; //set
                                 zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, temp, &dimRup, &beta, temp2, &dimLdown );

                                 alpha                 = 1.0;
                                 beta                  = 1.0; //add
                                 dcomplex * LblockLeft = LTleft[ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                                 zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, LblockLeft, &dimLup, temp2, &dimLdown, &beta, memHeff, &dimLup );
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

//4E4A
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::owner_specific_diagram( Prob->gL(), MPI_CHEMPS2_4E4A ) == MPIRANK ) && ( N1 == 1 ) ) {
#else
   if ( N1 == 1 ) {
#endif

      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( ( TwoJdown == 1 ) && abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  int fase               = phase( TwoSL + TwoSR + TwoJdown + TwoSLdown + TwoSRdown + 1 - TwoS2 );
                  const dcomplex factor1 = fase * sqrt( ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) * ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSLdown, TwoSR, TwoS2, TwoJ, 1, TwoSL ) * Wigner::wigner6j( TwoJdown, 1, TwoS2, TwoSR, TwoSLdown, TwoSRdown );

                  dcomplex factor2 = 0.0;
                  if ( TwoJ == TwoJdown ) {
                     fase    = phase( TwoSLdown + TwoSR + TwoJ + 3 + 2 * TwoS2 );
                     factor2 = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );
                  }

                  for ( int Irrep = 0; Irrep < ( bk_up->getNumberOfIrreps() ); Irrep++ ) {

                     int ILdown   = Irreps::directProd( IL, Irrep );
                     int IRdown   = Irreps::directProd( IR, Irrep );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                        bool isPossibleLeft = false;
                        for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                           if ( Irrep == bk_up->gIrrep( l_gamma ) ) { isPossibleLeft = true; }
                        }
                        bool isPossibleRight = false;
                        for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                           if ( Irrep == bk_up->gIrrep( l_beta ) ) { isPossibleRight = true; }
                        }
                        if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                           for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                              if ( Irrep == bk_up->gIrrep( l_gamma ) ) {

                                 int size = dimRup * dimRdown;
                                 for ( int cnt = 0; cnt < size; cnt++ ) {
                                    temp[ cnt ] = 0.0;
                                 }
                                 for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                                    if ( Irrep == bk_up->gIrrep( l_beta ) ) {
                                       dcomplex * LblockRight = Lright[ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                                       dcomplex prefact       = factor1 * Prob->gMxElement( l_gamma, theindex, theindex, l_beta );
                                       if ( TwoJ == TwoJdown ) { prefact += factor2 * Prob->gMxElement( l_gamma, theindex, l_beta, theindex ); }
                                       zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                    }
                                 }

                                 int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );
                                 if ( memSkappa != -1 ) {
                                    dcomplex alpha = 1.0;
                                    dcomplex beta  = 0.0; //set
                                    zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, temp, &dimRup, &beta, temp2, &dimLdown );

                                    beta                  = 1.0; //add
                                    dcomplex * LblockLeft = Lleft[ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                                    zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, LblockLeft, &dimLup, temp2, &dimLdown, &beta, memHeff, &dimLup );
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
   }

//4E4B
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::owner_specific_diagram( Prob->gL(), MPI_CHEMPS2_4E4B ) == MPIRANK ) && ( N1 == 2 ) ) {
#else
   if ( N1 == 2 ) {
#endif

      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSLdown + TwoSR - TwoS2 + 3 );
               const dcomplex factor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSL, TwoSR, TwoS2, TwoSRdown, TwoSLdown, 1 );

               for ( int Irrep = 0; Irrep < ( bk_up->getNumberOfIrreps() ); Irrep++ ) {

                  int ILdown   = Irreps::directProd( IL, Irrep );
                  int IRdown   = Irreps::directProd( IR, Irrep );
                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 1, TwoSRdown, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {
                     bool isPossibleLeft = false;
                     for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                        if ( Irrep == bk_up->gIrrep( l_gamma ) ) { isPossibleLeft = true; }
                     }
                     bool isPossibleRight = false;
                     for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                        if ( Irrep == bk_up->gIrrep( l_beta ) ) { isPossibleRight = true; }
                     }
                     if ( ( isPossibleLeft ) && ( isPossibleRight ) ) {

                        for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                           if ( Irrep == bk_up->gIrrep( l_gamma ) ) {

                              int size = dimRup * dimRdown;
                              for ( int cnt = 0; cnt < size; cnt++ ) {
                                 temp[ cnt ] = 0.0;
                              }
                              for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                                 if ( Irrep == bk_up->gIrrep( l_beta ) ) {
                                    dcomplex * LblockRight = Lright[ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                                    dcomplex prefact       = Prob->gMxElement( l_gamma, theindex, theindex, l_beta ) - 2 * Prob->gMxElement( l_gamma, theindex, l_beta, theindex );
                                    zaxpy_( &size, &prefact, LblockRight, &inc, temp, &inc );
                                 }
                              }

                              int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 1, TwoSRdown, IRdown );
                              if ( memSkappa != -1 ) {
                                 dcomplex alpha = factor;
                                 dcomplex beta  = 0.0; //set
                                 zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, temp, &dimRup, &beta, temp2, &dimLdown );

                                 alpha                 = 1.0;
                                 beta                  = 1.0; //add
                                 dcomplex * LblockLeft = Lleft[ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                                 zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, LblockLeft, &dimLup, temp2, &dimLdown, &beta, memHeff, &dimLup );
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
}

void CheMPS2::CHeffNS_1S::addDiagram4L1and4L2spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Aright, CTensorOperator *** ATright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

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

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
               {
                  int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );

                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 2, TwoSR, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex * blockA = ATright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
                        dcomplex beta     = 0.0; //set
                        dcomplex alpha    = factor;

                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockA, &dimRup, &beta, temp, &dimLdown );

                        beta              = 1.0; //add
                        alpha             = 1.0;
                        dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
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

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 2, TwoSR, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * blockA = ATright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
                           dcomplex beta     = 0.0; //set
                           dcomplex alpha    = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockA, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
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

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 2, TwoSR, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * blockA = Aright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                           dcomplex beta     = 0.0; //set
                           dcomplex alpha    = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockA, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
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

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
               {
                  int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown = Irreps::directProd( IR, Aright[ theindex - l_index ][ 0 ]->get_irrep() );

                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 2, TwoSR, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex * blockA = Aright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                        dcomplex beta     = 0.0; //set
                        dcomplex alpha    = factor;

                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockA, &dimRup, &beta, temp, &dimLdown );

                        beta              = 1.0; //add
                        alpha             = 1.0;
                        dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram4L1and4L2spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Bright, CTensorOperator *** BTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L1A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( 1 + TwoS2 - TwoJ );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 2, TwoSRdown, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * blockB = BTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                           dcomplex beta     = 0.0; //set
                           dcomplex alpha    = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockB, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4L1B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  int fase              = phase( TwoSR - TwoSRdown + TwoSLdown - TwoSL + TwoS2 - TwoJdown ); //bug fixed
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

                  for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
                     {
                        int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                        int IRdown = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );

                        int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );

                        if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                           int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR - 2, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex * blockB = BTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                              dcomplex beta     = 0.0; //set
                              dcomplex alpha    = factor;

                              zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockB, &dimRup, &beta, temp, &dimLdown );

                              beta              = 1.0; //add
                              alpha             = 1.0;
                              dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
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
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  int fase              = phase( 1 + TwoS2 - TwoJdown );
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

                  for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
                     {
                        int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                        int IRdown = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );

                        int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );

                        if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                           int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 2, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex * blockB = Bright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                              dcomplex beta     = 0.0; //set
                              dcomplex alpha    = factor;

                              zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockB, &dimRup, &beta, temp, &dimLdown );

                              beta              = 1.0; //add
                              alpha             = 1.0;
                              dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
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
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSRdown - TwoSR + TwoSL - TwoSLdown + TwoS2 - TwoJ );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_absigma( l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Bright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR + 2, TwoSRdown, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * blockB = Bright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                           dcomplex beta     = 0.0; //set
                           dcomplex alpha    = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, blockB, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram4L3and4L4spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Cright, CTensorOperator *** CTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

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

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
               {
                  int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );

                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex * ptr = Cright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );

                        dcomplex beta  = 0.0; //set
                        dcomplex alpha = factor;

                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                        beta              = 1.0; //add
                        alpha             = 1.0;
                        dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
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

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSR, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * ptr = Cright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );

                           dcomplex beta  = 0.0; //set
                           dcomplex alpha = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
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

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * ptr = CTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );

                           dcomplex beta  = 0.0; //set
                           dcomplex alpha = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
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

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
               {
                  int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown = Irreps::directProd( IR, Cright[ theindex - l_index ][ 0 ]->get_irrep() );

                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSR, IRdown );
                     if ( memSkappa != -1 ) {
                        dcomplex * ptr = CTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );

                        dcomplex beta  = 0.0; //set
                        dcomplex alpha = factor;

                        zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                        beta              = 1.0; //add
                        alpha             = 1.0;
                        dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram4L3and4L4spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Dright, CTensorOperator *** DTright, dcomplex * temp ) {

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   int N1   = NR - NL;
   int TwoS = ( N1 == 1 ) ? 1 : 0;

   int theindex = in->gIndex();
   int dimLup   = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   int TwoS2 = 0;
   int TwoJ  = TwoS;
   int inc   = 1;

   //4L3A.spin1
   if ( N1 == 1 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSL - TwoSLdown + 2 + TwoS2 - TwoJ );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSLdown + 1.0 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * ptr = Dright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex beta  = 0.0; //set
                           dcomplex alpha = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //4L3B.spin1
   if ( N1 == 2 ) {
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  int fase              = phase( TwoSR - TwoSRdown + 1 + TwoS2 - TwoJdown );
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoSRdown + 1.0 ) * ( TwoSLdown + 1.0 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

                  for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
                     {
                        int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                        int IRdown = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );

                        int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                        if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                           int memSkappa = in->gKappa( NL + 1, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex * ptr = Dright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                              dcomplex beta  = 0.0; //set
                              dcomplex alpha = factor;

                              zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                              beta              = 1.0; //add
                              alpha             = 1.0;
                              dcomplex * blockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
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
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            int TwoJstart = ( ( TwoSLdown != TwoSRdown ) || ( TwoS2 == 0 ) ) ? TwoS2 + 1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= TwoS2 + 1; TwoJdown += 2 ) {
               if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJdown ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

                  int fase              = phase( TwoSLdown + 2 - TwoSL + TwoS2 - TwoJdown );
                  const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSRdown, TwoSR, 1, TwoSLdown, TwoSL, 1, TwoJdown, TwoS2 );

                  for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
                     {
                        int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                        int IRdown = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );

                        int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                        int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                        if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                           int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );
                           if ( memSkappa != -1 ) {
                              dcomplex * ptr = DTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                              dcomplex beta  = 0.0; //set
                              dcomplex alpha = factor;

                              zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                              beta              = 1.0; //add
                              alpha             = 1.0;
                              dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                           }
                        }
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
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {
            if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoS2 ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

               int fase              = phase( TwoSR - TwoSRdown + 1 + TwoS2 - TwoJ );
               const dcomplex factor = fase * sqrt( 3.0 * ( TwoSL + 1.0 ) * ( TwoJ + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner9j( 2, TwoSR, TwoSRdown, 1, TwoSL, TwoSLdown, 1, TwoJ, TwoS2 );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                  if ( MPIchemps2::owner_cdf( Prob->gL(), l_index, theindex ) == MPIRANK )
#endif
                  {
                     int ILdown = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                     int IRdown = Irreps::directProd( IR, Dright[ theindex - l_index ][ 0 ]->get_irrep() );

                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        int memSkappa = in->gKappa( NL - 1, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );
                        if ( memSkappa != -1 ) {
                           dcomplex * ptr = DTright[ theindex - l_index ][ 0 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex beta  = 0.0; //set
                           dcomplex alpha = factor;

                           zgemm_( &notrans, &cotrans, &dimLdown, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimLdown, ptr, &dimRup, &beta, temp, &dimLdown );

                           beta              = 1.0; //add
                           alpha             = 1.0;
                           dcomplex * blockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );

                           zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimLdown, &alpha, blockL, &dimLup, temp, &dimLdown, &beta, memHeff, &dimLup );
                        }
                     }
                  }
               }
            }
         }
      }
   }
}