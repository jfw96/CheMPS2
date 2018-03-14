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

void CheMPS2::CHeffNS_1S::addDiagram2a1spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** ATtensors, CTensorS0T **** S0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int theindex = out->gIndex();
   int dimL           = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR           = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {

            int ILdown    = Irreps::directProd( IL, S0Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
            int IRdown    = Irreps::directProd( IR, ATtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
            int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR - 2, TwoSR, IRdown );

            if ( memSkappa != -1 ) {

               int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
               int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );

               dcomplex * BlockS0 = S0Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
               dcomplex * BlockA  = ATtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

               dcomplex alpha = 1.0;
               dcomplex beta  = 0.0;
               zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockS0, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

               beta = 1.0;
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace, &dimL, BlockA, &dimR, &beta, memHeff, &dimL );
            }
         }
      }

   } else {

      for ( int l_gamma = theindex + 1; l_gamma < Prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < Prob->gL(); l_delta++ ) {

            int ILdown    = Irreps::directProd( IL, ATtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
            int IRdown    = Irreps::directProd( IR, S0Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
            int memSkappa = in->gKappa( NL - 2, TwoSL, ILdown, NR - 2, TwoSR, IRdown );

            if ( memSkappa != -1 ) {

               int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
               int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSR, IRdown );

               dcomplex * BlockA  = ATtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
               dcomplex * BlockS0 = S0Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
               dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

               dcomplex alpha = 1.0;
               dcomplex beta  = 0.0;
               zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockA, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

               beta = 1.0;
               zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace, &dimL, BlockS0, &dimR, &beta, memHeff, &dimL );
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2a1spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** BTtensors, CTensorS1T **** S1Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif
   const int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = bk_up->gIrrep( theindex );

   int dimL = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) ) {

               int fase                 = phase( TwoSRdown + TwoSL + TwoS + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, S1Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, BTtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
                        int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR - 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );

                           dcomplex * BlockS1 = S1Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                           dcomplex * BlockB  = BTtensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex alpha = thefactor;
                           dcomplex beta  = 0.0;

                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockS1, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           alpha = beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace, &dimL, BlockB, &dimR, &beta, memHeff, &dimL );
                        }
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

               int fase                 = phase( TwoSRdown + TwoSL + TwoS + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_gamma = theindex + 1; l_gamma < Prob->gL(); l_gamma++ ) {
                  for ( int l_delta = l_gamma + 1; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, BTtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, S1Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
                        int memSkappa = in->gKappa( NL - 2, TwoSLdown, ILdown, NR - 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSRdown, IRdown );

                           dcomplex * BlockB  = BTtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                           dcomplex * BlockS1 = S1Ttensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex alpha = thefactor;
                           dcomplex beta  = 0.0;

                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockB,
                                   &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           alpha = beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                                   &dimL, BlockS1, &dimR, &beta, memHeff, &dimL );
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2a2spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out,
                                              CTensorOperator **** Atensors, CTensorS0 **** S0tensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = bk_up->gIrrep( theindex );

   int dimL = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, S0tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, Atensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
               int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR + 2, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );

                  dcomplex * BlockS0 = S0tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                  dcomplex * BlockA  = Atensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                  dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0;

                  zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockS0, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                  beta = 1.0;
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace, &dimL, BlockA, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }

   } else {

      for ( int l_gamma = theindex + 1; l_gamma < Prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, Atensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, S0tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
               int memSkappa = in->gKappa( NL + 2, TwoSL, ILdown, NR + 2, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSR, IRdown );

                  dcomplex * BlockA  = Atensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                  dcomplex * BlockS0 = S0tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                  dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0;

                  zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockA, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                  beta = 1.0;
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace, &dimL, BlockS0, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2a2spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Btensors, CTensorS1 **** S1tensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = bk_up->gIrrep( theindex );

   int dimL = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) ) {

               int fase                 = phase( TwoSLdown + TwoSR + TwoS + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSLdown + 1.0 ) / ( TwoSR + 1 ) ) * ( TwoSRdown + 1 ) *
                                          Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, S1tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, Btensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->get_irrep() );
                        int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR + 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );

                           dcomplex * BlockS1 = S1tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                           dcomplex * BlockB  = Btensors[ theindex ][ l_beta - l_alpha ][ theindex - l_beta ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex alpha = thefactor;
                           dcomplex beta  = 0.0;

                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockS1,
                                   &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           alpha = beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                                   &dimL, BlockB, &dimR, &beta, memHeff, &dimL );
                        }
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

               for ( int l_gamma = theindex + 1; l_gamma < Prob->gL(); l_gamma++ ) {
                  for ( int l_delta = l_gamma + 1; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Btensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, S1tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->get_irrep() );
                        int memSkappa = in->gKappa( NL + 2, TwoSLdown, ILdown, NR + 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSRdown, IRdown );

                           dcomplex * BlockB  = Btensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                           dcomplex * BlockS1 = S1tensors[ theindex ][ l_delta - l_gamma ][ l_gamma - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex alpha = thefactor;
                           dcomplex beta  = 0.0;

                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockB,
                                   &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           alpha = beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                                   &dimL, BlockS1, &dimR, &beta, memHeff, &dimL );
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2a3spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Ctensors, CTensorOperator **** CTtensors, CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = bk_up->gIrrep( theindex );

   int dimL = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
         for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_gamma, l_alpha ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, F0tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, Ctensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->get_irrep() );
               int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                  //no transpose
                  dcomplex * BlockF0 = F0tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * ptr     = Ctensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0;

                  zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockF0,
                          &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                  beta = 1.0;

                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                          &dimL, ptr, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_alpha, l_gamma ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, F0Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, CTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->get_irrep() );
               int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                  //transpose
                  dcomplex * BlockF0 = F0Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * ptr     = CTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0;

                  zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, BlockF0,
                          &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                  beta = 1.0;
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                          &dimL, ptr, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }

   } else {

      for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
         for ( int l_beta = l_delta + 1; l_beta < Prob->gL(); l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_delta, l_beta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, Ctensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, F0tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->get_irrep() );
               int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                  //no transpose
                  dcomplex * ptr     = Ctensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * BlockF0 = F0tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0;

                  zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, ptr,
                          &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                  beta = 1.0;
                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                          &dimL, BlockF0, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }

      for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
         for ( int l_delta = l_beta; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_beta, l_delta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, CTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, F0Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->get_irrep() );
               int memSkappa = in->gKappa( NL, TwoSL, ILdown, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IRdown );

                  //transpose
                  dcomplex * ptr     = CTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * BlockF0 = F0Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0;

                  zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &alpha, ptr,
                          &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                  beta = 1.0;

                  zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &alpha, workspace,
                          &dimL, BlockF0, &dimR, &beta, memHeff, &dimL );
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2a3spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Dtensors, CTensorOperator **** DTtensors, CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int theindex = out->gIndex();

   int NL    = out->gNL( ikappa );
   int TwoSL = out->gTwoSL( ikappa );
   int IL    = out->gIL( ikappa );

   int NR    = out->gNR( ikappa );
   int TwoSR = out->gTwoSR( ikappa );
   int IR    = out->gIR( ikappa );

   const int N    = NR - NL;
   const int TwoS = ( N == 1 ) ? 1 : 0;
   const int I    = bk_up->gIrrep( theindex );

   int dimL = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoS ) ) {

               // 2 of 2
               int fase           = phase( TwoSLdown + TwoSRdown + TwoS + 2 );
               dcomplex prefactor = fase * sqrt( ( TwoSLdown + 1.0 ) * ( TwoSRdown + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                  for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_gamma, l_alpha ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, F1tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, Dtensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->get_irrep() );
                        int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                           //no transpose
                           dcomplex * BlockF1 = F1tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * ptr     = Dtensors[ theindex ][ l_alpha - l_gamma ][ theindex - l_alpha ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex beta = 0.0;

                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &prefactor, BlockF1,
                                   &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           beta = 1.0;

                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, workspace,
                                   &dimL, ptr, &dimR, &beta, memHeff, &dimL );
                        }
                     }
                  }
               }

               // 1 of 2
               fase      = phase( TwoSL + TwoSR + TwoS + 2 );
               prefactor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_alpha, l_gamma ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, F1Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, DTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->get_irrep() );
                        int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                           //transpose
                           dcomplex * BlockF1 = F1Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * ptr     = DTtensors[ theindex ][ l_gamma - l_alpha ][ theindex - l_gamma ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex beta = 0.0;
                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &prefactor, BlockF1,
                                   &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, workspace, &dimL, ptr,
                                   &dimR, &beta, memHeff, &dimL );
                        }
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

               // 2 of 2
               int fase           = phase( TwoSLdown + TwoSRdown + TwoS + 2 );
               dcomplex prefactor = fase * sqrt( ( TwoSLdown + 1.0 ) * ( TwoSRdown + 1.0 ) ) *
                                    Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_delta = theindex + 1; l_delta < Prob->gL(); l_delta++ ) {
                  for ( int l_beta = l_delta + 1; l_beta < Prob->gL(); l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_delta, l_beta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Dtensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, F1tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->get_irrep() );
                        int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                           //no transpose
                           dcomplex * ptr     = Dtensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * BlockF1 = F1tensors[ theindex ][ l_beta - l_delta ][ l_delta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex beta = 0.0;
                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &prefactor, ptr, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, workspace, &dimL, BlockF1, &dimR, &beta, memHeff, &dimL );
                        }
                     }
                  }
               }

               // 1 of 2
               fase      = phase( TwoSL + TwoSR + TwoS + 2 );
               prefactor = fase * sqrt( ( TwoSL + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1 ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoS, TwoSR, TwoSL, 2 );

               for ( int l_beta = theindex + 1; l_beta < Prob->gL(); l_beta++ ) {
                  for ( int l_delta = l_beta; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_beta, l_delta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, DTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, F1Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->get_irrep() );
                        int memSkappa = in->gKappa( NL, TwoSLdown, ILdown, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IRdown );

                           //transpose
                           dcomplex * ptr     = DTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * BlockF1 = F1Ttensors[ theindex ][ l_delta - l_beta ][ l_beta - theindex - 1 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockS  = in->gStorage() + in->gKappa2index( memSkappa );

                           dcomplex beta = 0.0;
                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &prefactor, ptr, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, workspace, &dimL, BlockF1, &dimR, &beta, memHeff, &dimL );
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2b1and2b2( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N == 0 ) {

      int theindex = out->gIndex();
      int NL       = out->gNL( ikappa );
      int TwoSL    = out->gTwoSL( ikappa );
      int IL       = out->gIL( ikappa );

      int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, IL );

      if ( dimLdown > 0 ) {

         int NR    = out->gNR( ikappa );
         int TwoSR = out->gTwoSR( ikappa );
         int IR    = out->gIR( ikappa );

         int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IR );

         int memSkappa = in->gKappa( NL - 2, TwoSL, IL, NR, TwoSR, IR );

         if ( memSkappa != -1 ) {

            dcomplex * BlockA = ATtensor->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, IL );
            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );

            dcomplex alpha = sqrt( 2.0 );
            dcomplex beta  = 1.0;
            zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, BlockA, &dimLup, BlockS, &dimLdown, &beta, memHeff, &dimLup );
         }
      }
   }

   if ( N == 2 ) {

      int theindex = out->gIndex();
      int NL       = out->gNL( ikappa );
      int TwoSL    = out->gTwoSL( ikappa );
      int IL       = out->gIL( ikappa );

      int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, IL );

      if ( dimLdown > 0 ) {

         int NR    = out->gNR( ikappa );
         int TwoSR = out->gTwoSR( ikappa );
         int IR    = out->gIR( ikappa );

         int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IR );

         int memSkappa = in->gKappa( NL + 2, TwoSL, IL, NR, TwoSR, IR );

         if ( memSkappa != -1 ) {

            dcomplex * BlockA = Atensor->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );
            dcomplex * BlockS = in->gStorage() + in->gKappa2index( memSkappa );
            dcomplex alpha    = sqrt( 2.0 );
            dcomplex beta     = 1.0;
            zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, BlockA, &dimLup, BlockS, &dimLdown, &beta, memHeff, &dimLup );
         }
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2b3spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N != 0 ) {

      int theindex = out->gIndex();

      int NL    = out->gNL( ikappa );
      int TwoSL = out->gTwoSL( ikappa );
      int IL    = out->gIL( ikappa );

      int dimLU = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR  = bk_down->gCurrentDim( theindex + 1, out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      int memSkappa = in->gKappa( out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ), out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
         dcomplex alpha    = ( ( N == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &notrans, &dimLU, &dimR, &dimLD, &alpha, Cblock, &dimLU, in->gStorage() + in->gKappa2index( memSkappa ), &dimLD, &beta, memHeff, &dimLU );
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2b3spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor ) {

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

      int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR   = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, IL );
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

void CheMPS2::CHeffNS_1S::addDiagram2e1and2e2( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );
   int N2 = 0;

   //    if ( N1 == 2 ) {

   //       int theindex = out->gIndex();
   //       int NL       = out->gNL( ikappa );
   //       int TwoSL    = out->gTwoSL( ikappa );
   //       int IL       = out->gIL( ikappa );
   //       int NR       = out->gNR( ikappa );
   //       int TwoSR    = out->gTwoSR( ikappa );
   //       int IR       = out->gIR( ikappa );

   //       int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   //       int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSR, IR );
   //       int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   //       int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IR );

   //       if ( memSkappa != -1 ) {

   //          dcomplex * BlockA = ATtensor->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IR );
   //          dcomplex alpha    = sqrt( 2.0 );
   //          dcomplex beta     = 1.0;
   //          zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
   //       }
   //    }

   if ( N1 == 2 ) {

      int theindex = out->gIndex();
      int NL       = out->gNL( ikappa );
      int TwoSL    = out->gTwoSL( ikappa );
      int IL       = out->gIL( ikappa );
      int NR       = out->gNR( ikappa );
      int TwoSR    = out->gTwoSR( ikappa );
      int IR       = out->gIR( ikappa );

      int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = bk_down->gCurrentDim( theindex + 1, NR - 2, TwoSR, IR );
      int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      int memSkappa = in->gKappa( NL, TwoSL, IL, NR - 2, TwoSR, IR );

      if ( memSkappa != -1 ) {

         dcomplex * BlockA = ATtensor->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IR );
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;

         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }

   // if ( N1 == 0 ) {

   //    int theindex = out->gIndex();
   //    int NL       = out->gNL( ikappa );
   //    int TwoSL    = out->gTwoSL( ikappa );
   //    int IL       = out->gIL( ikappa );
   //    int NR       = out->gNR( ikappa );
   //    int TwoSR    = out->gTwoSR( ikappa );
   //    int IR       = out->gIR( ikappa );

   //    int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   //    int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSR, IR );
   //    int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   //    int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IR );

   //    if ( memSkappa != -1 ) {

   //       dcomplex * BlockA = Atensor->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IR );
   //       dcomplex alpha    = sqrt( 2.0 );
   //       dcomplex beta     = 1.0;
   //       zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
   //    }
   // }

   if ( N1 == 0 ) {

      int theindex = out->gIndex();
      int NL       = out->gNL( ikappa );
      int TwoSL    = out->gTwoSL( ikappa );
      int IL       = out->gIL( ikappa );
      int NR       = out->gNR( ikappa );
      int TwoSR    = out->gTwoSR( ikappa );
      int IR       = out->gIR( ikappa );

      int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = bk_down->gCurrentDim( theindex + 1, NR + 2, TwoSR, IR );
      int dimRup   = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      int memSkappa = in->gKappa( NL, TwoSL, IL, NR + 2, TwoSR, IR );

      if ( memSkappa != -1 ) {

         dcomplex * BlockA = Atensor->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IR );
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;

         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2e3spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N1 != 0 ) {

      int theindex = out->gIndex();

      int NR    = out->gNR( ikappa );
      int TwoSR = out->gTwoSR( ikappa );
      int IR    = out->gIR( ikappa );

      int dimRU = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int dimRD = bk_down->gCurrentDim( theindex + 1, NR, TwoSR, IR );
      int dimL  = bk_down->gCurrentDim( theindex, out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ) );

      int memSkappa = in->gKappa( out->gNL( ikappa ), out->gTwoSL( ikappa ), out->gIL( ikappa ), out->gNR( ikappa ), out->gTwoSR( ikappa ), out->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         dcomplex alpha    = ( ( N1 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRU, &dimRD, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Cblock, &dimRU, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CHeffNS_1S::addDiagram2e3spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor ) {

   // char cotrans = 'C';
   // char notrans = 'N';

   // int N1 = out->gNR( ikappa ) - out->gNL( ikappa );
   // if ( N1 == 1 ) {

   //    int theindex = out->gIndex();

   //    int NL    = out->gNL( ikappa );
   //    int TwoSL = out->gTwoSL( ikappa );
   //    int IL    = out->gIL( ikappa );
   //    int NR    = out->gNR( ikappa );
   //    int TwoSR = out->gTwoSR( ikappa );
   //    int IR    = out->gIR( ikappa );
   //    int TwoJ  = ( N1 == 1 ) ? 1 : 0;
   //    int N2    = 0;

   //    int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
   //    int dimRup = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

   //    for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

   //       int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IR );

   //       if ( dimRdown > 0 ) {

   //          dcomplex * Dblock = DTtensor->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IR );

   //          int TwoS2     = ( N2 == 1 ) ? 1 : 0;
   //          int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
   //          for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
   //             if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

   //                int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IR );

   //                if ( memSkappa != -1 ) {

   //                   int fase       = phase( TwoSRdown + TwoSL + 2 * TwoJ + TwoS2 + 1 );
   //                   dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1.0 ) * ( TwoJdown + 1.0 ) / ( TwoSR + 1.0 ) ) * ( TwoSRdown + 1.0 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSR, TwoSRdown, TwoSL );
   //                   dcomplex beta  = 1.0;
   //                   zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Dblock, &dimRup, &beta, memHeff, &dimL );
   //                }
   //             }
   //          }
   //       }
   //    }
   // }

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = out->gNR( ikappa ) - out->gNL( ikappa );

   if ( N1 == 1 ) {

      int TwoS1 = ( N1 == 1 ) ? 1 : 0;

      int theindex = in->gIndex();

      int NL    = in->gNL( ikappa );
      int TwoSL = in->gTwoSL( ikappa );
      int IL    = in->gIL( ikappa );
      int NR    = in->gNR( ikappa );
      int TwoSR = in->gTwoSR( ikappa );
      int IR    = in->gIR( ikappa );

      int TwoJ  = TwoS1;
      int TwoS2 = 0;
      int N2    = 0;

      int N1 = NR - NL;

      int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRup = bk_up->gCurrentDim( theindex + 1, NR, TwoSR, IR );

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 1, NR, TwoSRdown, IR );

         if ( dimRdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IR );

            int TwoS2     = ( N2 == 1 ) ? 1 : 0;
            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = in->gKappa( NL, TwoSL, IL, NR, TwoSRdown, IR );

                  if ( memSkappa != -1 ) {

                     int fase       = phase( TwoSRdown + TwoSL + 2 * TwoJ + TwoS2 + 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1 ) * ( TwoJdown + 1 ) * ( TwoSRdown + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta  = 1.0;

                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, in->gStorage() + in->gKappa2index( memSkappa ), &dimL, Dblock, &dimRdown, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}
