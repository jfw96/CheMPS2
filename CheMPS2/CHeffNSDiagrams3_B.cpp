#include <math.h>
#include <stdlib.h>

#include "CHeffNS.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Wigner.h"

void CheMPS2::CHeffNS::addDiagram3Aand3D( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS, CTensorQ * Qleft, CTensorQT * QTleft, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int theindex = denS->gIndex();
   int ILdown   = Irreps::directProd( IL, bk_up->gIrrep( theindex ) );
   int TwoS2    = ( N2 == 1 ) ? 1 : 0;

   int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   if ( N1 == 2 ) { //3A1A and 3D1
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 ) {

            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL + 1, TwoSLdown, ILdown, 1, N2, TwoJdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     int fase        = phase( TwoSL + TwoSR + 2 + TwoS2 );
                     dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSLdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta   = 1.0; //add
                     char notr       = 'N';

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

                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   if ( N1 == 1 ) { //3A1B
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {
            int dimLdown  = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
            int memSkappa = denS->gKappa( NL + 1, TwoSLdown, ILdown, 0, N2, TwoS2, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSL + TwoSR + 1 + TwoS2 );
               dcomplex factor   = sqrt( ( TwoSLdown + 1 ) * ( TwoJ + 1.0 ) ) * fase * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSL, TwoSLdown, TwoSR );
               dcomplex beta     = 1.0;
               dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   if ( N1 == 0 ) { //3A2A
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 ) {

            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL - 1, TwoSLdown, ILdown, 1, N2, TwoJdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     int fase          = phase( TwoSLdown + TwoSR + 1 + TwoS2 );
                     dcomplex factor   = fase * sqrt( ( TwoSL + 1 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta     = 1.0;
                     dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   if ( N1 == 1 ) { //3A2B ans 3D2
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS2 ) && ( TwoSLdown >= 0 ) ) {
            int dimLdown  = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
            int memSkappa = denS->gKappa( NL - 1, TwoSLdown, ILdown, 2, N2, TwoS2, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase        = phase( TwoSLdown + TwoSR + 2 + TwoS2 );
               dcomplex factor = fase * sqrt( ( TwoSL + 1 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSL, TwoSLdown, TwoSR );
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

               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram3Band3I( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS, CTensorQ * Qleft, CTensorQT * QTleft, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int theindex = denS->gIndex();
   int ILdown   = Irreps::directProd( IL, bk_up->gIrrep( theindex + 1 ) );
   int TwoS1    = ( N1 == 1 ) ? 1 : 0;

   int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   if ( N2 == 2 ) { //3B1A and 3I2
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 ) {

            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS1 == 0 ) ) ? 1 + TwoS1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS1; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL + 1, TwoSLdown, ILdown, N1, 1, TwoJdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     int fase        = phase( TwoSL + TwoSR + 3 - TwoJdown );
                     dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSLdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS1, 1, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta   = 1.0; //add

                     dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     int inc           = 1;
                     int size          = dimLup * dimLdown;
                     zcopy_( &size, BlockQ, &inc, temp, &inc );

                     for ( int l_index = 0; l_index < theindex; l_index++ ) {
                        if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex + 1 ) ) {
                           dcomplex alpha    = Prob->gMxElement( l_index, theindex + 1, theindex + 1, theindex + 1 );
                           dcomplex * BlockL = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                           zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                        }
                     }

                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   if ( N2 == 1 ) { //3B1B
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS1 ) && ( TwoSLdown >= 0 ) ) {
            int dimLdown  = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );
            int memSkappa = denS->gKappa( NL + 1, TwoSLdown, ILdown, N1, 0, TwoS1, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSL + TwoSR + 2 - TwoJ );
               dcomplex factor   = sqrt( ( TwoSLdown + 1 ) * ( TwoJ + 1.0 ) ) * fase * Wigner::wigner6j( TwoS1, TwoJ, 1, TwoSL, TwoSLdown, TwoSR );
               dcomplex beta     = 1.0;
               dcomplex * BlockQ = Qleft->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }

   if ( N2 == 0 ) { //3B2A
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
         if ( dimLdown > 0 ) {

            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS1 == 0 ) ) ? 1 + TwoS1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS1; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL - 1, TwoSLdown, ILdown, N1, 1, TwoJdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {
                     int fase          = phase( TwoSLdown + TwoSR + 2 - TwoJdown );
                     dcomplex factor   = fase * sqrt( ( TwoSL + 1 ) * ( TwoJdown + 1.0 ) ) * Wigner::wigner6j( TwoJdown, TwoS1, 1, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta     = 1.0;
                     dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, BlockQ, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   if ( N2 == 1 ) { //3B2B and 3I1
      for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSR ) <= TwoS1 ) && ( TwoSLdown >= 0 ) ) {
            int dimLdown  = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
            int memSkappa = denS->gKappa( NL - 1, TwoSLdown, ILdown, N1, 2, TwoS1, NR, TwoSR, IR );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSLdown + TwoSR + 3 - TwoJ );
               dcomplex factor   = fase * sqrt( ( TwoSL + 1 ) * ( TwoJ + 1.0 ) ) * Wigner::wigner6j( TwoS1, TwoJ, 1, TwoSL, TwoSLdown, TwoSR );
               dcomplex beta     = 1.0;
               dcomplex * BlockQ = QTleft->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
               int inc           = 1;
               int size          = dimLup * dimLdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = 0; l_index < theindex; l_index++ ) {
                  if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex + 1 ) ) {
                     dcomplex alpha    = Prob->gMxElement( l_index, theindex + 1, theindex + 1, theindex + 1 );
                     dcomplex * BlockL = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &factor, temp, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram3C( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS, CTensorQ ** Qleft, CTensorQT ** QTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int theindex = denS->gIndex();

   int dimRup = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   //First do 3C1
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase              = phase( TwoSLdown + TwoSR + TwoJ + 1 + ( ( N1 == 1 ) ? 2 : 0 ) + ( ( N2 == 1 ) ? 2 : 0 ) );
            const dcomplex factor = fase * sqrt( ( TwoSLdown + 1 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = theindex + 2; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_q( Prob->gL(), l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = denS->gKappa( NL + 1, TwoSLdown, ILdown, N1, N2, TwoJ, NR + 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 1, TwoSRdown, IRdown );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                     dcomplex * Qblock = Qleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * Lblock = Lright[ l_index - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );

                     dcomplex beta  = 0.0; //set
                     dcomplex alpha = factor;
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Qblock, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

                     beta  = 1.0; //add
                     alpha = 1.0;
                     zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Lblock, &dimRup, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }

   //Then do 3C2
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase            = phase( TwoSL + TwoSRdown + TwoJ + 1 + ( ( N1 == 1 ) ? 2 : 0 ) + ( ( N2 == 1 ) ? 2 : 0 ) );
            const double factor = fase * sqrt( ( TwoSL + 1 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = theindex + 2; l_index < Prob->gL(); l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_q( Prob->gL(), l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = denS->gKappa( NL - 1, TwoSLdown, ILdown, N1, N2, TwoJ, NR - 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 1, TwoSRdown, IRdown );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                     dcomplex * Qblock = QTleft[ l_index - theindex ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * Lblock = LTright[ l_index - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );

                     dcomplex beta  = 0.0; //set
                     dcomplex alpha = factor;
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Qblock, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

                     beta  = 1.0; //add
                     alpha = 1.0;
                     zgemm_( &notrans, &cotrans, &dimLup, &dimRup, &dimRdown, &alpha, temp, &dimLup, Lblock, &dimRup, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram3Eand3H( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS ) { //TwoJ = TwoJdown

   int theindex = denS->gIndex();

   if ( bk_up->gIrrep( theindex ) != bk_up->gIrrep( theindex + 1 ) ) { return; }

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int size = ( bk_down->gCurrentDim( theindex, NL, TwoSL, IL ) ) * ( bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR ) );
   int inc  = 1;

   if ( ( N1 == 2 ) && ( N2 == 0 ) ) { //3E1A

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 1, 1, 0, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 ) * Prob->gMxElement( theindex, theindex, theindex, theindex + 1 );
         zaxpy_( &size, &alpha, memS + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 2 ) && ( N2 == 1 ) ) { //3E1B and 3H1B

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 1, 2, 1, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         dcomplex alpha = -( Prob->gMxElement( theindex, theindex, theindex, theindex + 1 ) + Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex + 1 ) );
         zaxpy_( &size, &alpha, memS + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 1 ) && ( N2 == 1 ) && ( TwoJ == 0 ) ) { //3E2A and 3H1A

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 2, 0, 0, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 ) * Prob->gMxElement( theindex, theindex, theindex, theindex + 1 );
         zaxpy_( &size, &alpha, memS + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }

      memSkappa = denS->gKappa( NL, TwoSL, IL, 0, 2, 0, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 ) * Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex + 1 );
         zaxpy_( &size, &alpha, memS + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 1 ) && ( N2 == 2 ) ) { //3E2B and 3H2B

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 2, 1, 1, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         dcomplex alpha = -( Prob->gMxElement( theindex, theindex, theindex, theindex + 1 ) + Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex + 1 ) );
         zaxpy_( &size, &alpha, memS + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 0 ) && ( N2 == 2 ) ) { //3H2A

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 1, 1, 0, NR, TwoSR, IR );
      if ( memSkappa != -1 ) {
         dcomplex alpha = sqrt( 2.0 ) * Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex + 1 );
         zaxpy_( &size, &alpha, memS + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram3Kand3F( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS, CTensorQ * Qright, CTensorQT * QTright, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int theindex = denS->gIndex();
   int IRdown   = Irreps::directProd( IR, bk_up->gIrrep( theindex ) );
   int TwoS2    = ( N2 == 1 ) ? 1 : 0;

   int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   if ( N1 == 1 ) { //3K1A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = bk_down->gCurrentDim( theindex + 2, NR - 1, TwoSRdown, IRdown );
            int memSkappa = denS->gKappa( NL, TwoSL, IL, 0, N2, TwoS2, NR - 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSL + TwoSR + TwoJ + 2 * TwoS2 );
               dcomplex factor   = sqrt( ( TwoJ + 1 ) * ( TwoSR + 1.0 ) ) * fase * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSR, TwoSRdown, TwoSL );
               dcomplex beta     = 1.0; //add
               dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }

   if ( N1 == 2 ) { //3K1B and 3F1
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSL, IL, 1, N2, TwoJdown, NR - 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int fase        = phase( TwoSL + TwoSR + TwoJdown + 1 + 2 * TwoS2 );
                     dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSR + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta   = 1.0; //add

                     dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     int inc           = 1;
                     int size          = dimRup * dimRdown;
                     zcopy_( &size, BlockQ, &inc, temp, &inc );

                     for ( int l_index = theindex + 2; l_index < Prob->gL(); l_index++ ) {
                        if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
                           dcomplex alpha    = Prob->gMxElement( theindex, theindex, theindex, l_index );
                           dcomplex * BlockL = LTright[ l_index - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                           zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                        }
                     }

                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   if ( N1 == 0 ) { //3K2A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSL, IL, 1, N2, TwoJdown, NR + 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int fase          = phase( TwoSL + TwoSRdown + TwoJdown + 2 * TwoS2 );
                     dcomplex factor   = sqrt( ( TwoJdown + 1 ) * ( TwoSRdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS2, 1, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta     = 1.0; //add
                     dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   if ( N1 == 1 ) { //3K2B and 3F2
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS2 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = bk_down->gCurrentDim( theindex + 2, NR + 1, TwoSRdown, IRdown );
            int memSkappa = denS->gKappa( NL, TwoSL, IL, 2, N2, TwoS2, NR + 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase        = phase( TwoSL + TwoSRdown + TwoJ + 1 + 2 * TwoS2 );
               dcomplex factor = sqrt( ( TwoJ + 1 ) * ( TwoSRdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoS2, TwoJ, 1, TwoSR, TwoSRdown, TwoSL );
               dcomplex beta   = 1.0; //add

               dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               int inc           = 1;
               int size          = dimRup * dimRdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = theindex + 2; l_index < Prob->gL(); l_index++ ) {
                  if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex ) ) {
                     dcomplex alpha    = Prob->gMxElement( theindex, theindex, theindex, l_index );
                     dcomplex * BlockL = Lright[ l_index - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram3Land3G( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS, CTensorQ * Qright, CTensorQT * QTright, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int theindex = denS->gIndex();
   int IRdown   = Irreps::directProd( IR, bk_up->gIrrep( theindex + 1 ) );
   int TwoS1    = ( N1 == 1 ) ? 1 : 0;

   int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimRup = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   if ( N2 == 1 ) { //3L1A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS1 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = bk_down->gCurrentDim( theindex + 2, NR - 1, TwoSRdown, IRdown );
            int memSkappa = denS->gKappa( NL, TwoSL, IL, N1, 0, TwoS1, NR - 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase          = phase( TwoSL + TwoSR + TwoS1 + 1 );
               dcomplex factor   = sqrt( ( TwoJ + 1 ) * ( TwoSR + 1.0 ) ) * fase * Wigner::wigner6j( TwoS1, TwoJ, 1, TwoSR, TwoSRdown, TwoSL );
               dcomplex beta     = 1.0; //add
               dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }

   if ( N2 == 2 ) { //3L1B and 3G1
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS1 == 0 ) ) ? 1 + TwoS1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS1; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSL, IL, N1, 1, TwoJdown, NR - 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int fase        = phase( TwoSL + TwoSR + TwoS1 + 2 );
                     dcomplex factor = sqrt( ( TwoJdown + 1 ) * ( TwoSR + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS1, 1, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta   = 1.0; //add

                     dcomplex * BlockQ = QTright->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                     int inc           = 1;
                     int size          = dimRup * dimRdown;
                     zcopy_( &size, BlockQ, &inc, temp, &inc );

                     for ( int l_index = theindex + 2; l_index < Prob->gL(); l_index++ ) {
                        if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex + 1 ) ) {
                           dcomplex alpha    = Prob->gMxElement( theindex + 1, theindex + 1, theindex + 1, l_index );
                           dcomplex * BlockL = LTright[ l_index - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );
                           zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                        }
                     }

                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   if ( N2 == 0 ) { //3L2A
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 1, TwoSRdown, IRdown );
         if ( dimRdown > 0 ) {

            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS1 == 0 ) ) ? 1 + TwoS1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS1; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSL, IL, N1, 1, TwoJdown, NR + 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int fase          = phase( TwoSL + TwoSRdown + TwoS1 + 1 );
                     dcomplex factor   = sqrt( ( TwoJdown + 1 ) * ( TwoSRdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoJdown, TwoS1, 1, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta     = 1.0; //add
                     dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, BlockQ, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }

   if ( N2 == 1 ) { //3L2B and 3G2
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSL - TwoSRdown ) <= TwoS1 ) && ( TwoSRdown >= 0 ) ) {
            int dimRdown  = bk_down->gCurrentDim( theindex + 2, NR + 1, TwoSRdown, IRdown );
            int memSkappa = denS->gKappa( NL, TwoSL, IL, N1, 2, TwoS1, NR + 1, TwoSRdown, IRdown );
            if ( memSkappa != -1 ) {
               int fase        = phase( TwoSL + TwoSRdown + TwoS1 + 2 );
               dcomplex factor = sqrt( ( TwoJ + 1 ) * ( TwoSRdown + 1.0 ) ) * fase * Wigner::wigner6j( TwoS1, TwoJ, 1, TwoSR, TwoSRdown, TwoSL );
               dcomplex beta   = 1.0; //add

               dcomplex * BlockQ = Qright->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
               int inc           = 1;
               int size          = dimRup * dimRdown;
               zcopy_( &size, BlockQ, &inc, temp, &inc );

               for ( int l_index = theindex + 2; l_index < Prob->gL(); l_index++ ) {
                  if ( bk_up->gIrrep( l_index ) == bk_up->gIrrep( theindex + 1 ) ) {
                     dcomplex alpha    = Prob->gMxElement( theindex + 1, theindex + 1, theindex + 1, l_index );
                     dcomplex * BlockL = Lright[ l_index - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );
                     zaxpy_( &size, &alpha, BlockL, &inc, temp, &inc );
                  }
               }

               zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &factor, memS + denS->gKappa2index( memSkappa ), &dimL, temp, &dimRup, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram3J( const int ikappa, dcomplex * memS, dcomplex * memHeff, CSobject * denS, CTensorQ ** Qright, CTensorQT ** QTright, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denS->gNL( ikappa );
   int TwoSL = denS->gTwoSL( ikappa );
   int IL    = denS->gIL( ikappa );
   int N1    = denS->gN1( ikappa );
   int N2    = denS->gN2( ikappa );
   int TwoJ  = denS->gTwoJ( ikappa );
   int NR    = denS->gNR( ikappa );
   int TwoSR = denS->gTwoSR( ikappa );
   int IR    = denS->gIR( ikappa );

   int theindex = denS->gIndex();

   int dimRup = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );
   int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );

   //First do 3J2
   for ( int TwoSLdown = TwoSL - 1; TwoSLdown <= TwoSL + 1; TwoSLdown += 2 ) {
      for ( int TwoSRdown = TwoSR - 1; TwoSRdown <= TwoSR + 1; TwoSRdown += 2 ) {
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase            = phase( TwoSLdown + TwoSR + TwoJ + 1 + ( ( N1 == 1 ) ? 2 : 0 ) + ( ( N2 == 1 ) ? 2 : 0 ) );
            const double factor = fase * sqrt( ( TwoSLdown + 1 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_q( Prob->gL(), l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = denS->gKappa( NL + 1, TwoSLdown, ILdown, N1, N2, TwoJ, NR + 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {

                     int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 1, TwoSRdown, IRdown );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL + 1, TwoSLdown, ILdown );

                     dcomplex * Lblock = Lleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL + 1, TwoSLdown, ILdown );
                     dcomplex * Qblock = Qright[ theindex + 1 - l_index ]->gStorage( NR, TwoSR, IR, NR + 1, TwoSRdown, IRdown );

                     dcomplex beta  = 0.0; //set
                     dcomplex alpha = factor;
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

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
         if ( ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) && ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) ) {

            int fase            = phase( TwoSL + TwoSRdown + TwoJ + 1 + ( ( N1 == 1 ) ? 2 : 0 ) + ( ( N2 == 1 ) ? 2 : 0 ) );
            const double factor = fase * sqrt( ( TwoSL + 1 ) * ( TwoSR + 1.0 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoSRdown, TwoSLdown, 1 );

            for ( int l_index = 0; l_index < theindex; l_index++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
               if ( MPIchemps2::owner_q( Prob->gL(), l_index ) == MPIRANK )
#endif
               {
                  int ILdown    = Irreps::directProd( IL, bk_up->gIrrep( l_index ) );
                  int IRdown    = Irreps::directProd( IR, bk_up->gIrrep( l_index ) );
                  int memSkappa = denS->gKappa( NL - 1, TwoSLdown, ILdown, N1, N2, TwoJ, NR - 1, TwoSRdown, IRdown );
                  if ( memSkappa != -1 ) {
                     int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 1, TwoSRdown, IRdown );
                     int dimLdown = bk_down->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );

                     dcomplex * Lblock = LTleft[ theindex - 1 - l_index ]->gStorage( NL, TwoSL, IL, NL - 1, TwoSLdown, ILdown );
                     dcomplex * Qblock = QTright[ theindex + 1 - l_index ]->gStorage( NR, TwoSR, IR, NR - 1, TwoSRdown, IRdown );

                     dcomplex beta  = 0.0; //set
                     dcomplex alpha = factor;
                     zgemm_( &notrans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLup, memS + denS->gKappa2index( memSkappa ), &dimLdown, &beta, temp, &dimLup );

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
