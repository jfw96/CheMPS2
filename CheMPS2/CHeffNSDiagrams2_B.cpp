
#include <math.h>
#include <stdlib.h>

#include "CHeffNS.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Wigner.h"

void CheMPS2::CHeffNS::addDiagram2a1spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP,
                                           CTensorOperator **** ATtensors, CTensorS0T **** S0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denP->gNL( ikappa );
   int TwoSL = denP->gTwoSL( ikappa );
   int IL    = denP->gIL( ikappa );

   int NR    = denP->gNR( ikappa );
   int TwoSR = denP->gTwoSR( ikappa );
   int IR    = denP->gIR( ikappa );

   int N1   = denP->gN1( ikappa );
   int N2   = denP->gN2( ikappa );
   int TwoJ = denP->gTwoJ( ikappa );

   int theindex = denP->gIndex();
   int dimL     = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, S0Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, ATtensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->get_irrep() );
               int memSkappa = denS->gKappa( NL - 2, TwoSL, ILdown, N1, N2, TwoJ, NR - 2, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 2, TwoSR, IRdown );

                  dcomplex * BlockS0 = S0Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
                  dcomplex * BlockA  = ATtensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
                  dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

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

      for ( int l_gamma = theindex + 2; l_gamma < Prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, ATtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, S0Ttensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]->get_irrep() );
               int memSkappa = denS->gKappa( NL - 2, TwoSL, ILdown, N1, N2, TwoJ, NR - 2, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 2, TwoSR, IRdown );

                  dcomplex * BlockA  = ATtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, ILdown );
                  dcomplex * BlockS0 = S0Ttensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IRdown );
                  dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

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

void CheMPS2::CHeffNS::addDiagram2a2spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP,
                                           CTensorOperator **** Atensors, CTensorS0 **** S0tensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denP->gNL( ikappa );
   int TwoSL = denP->gTwoSL( ikappa );
   int IL    = denP->gIL( ikappa );

   int NR    = denP->gNR( ikappa );
   int TwoSR = denP->gTwoSR( ikappa );
   int IR    = denP->gIR( ikappa );

   int N1   = denP->gN1( ikappa );
   int N2   = denP->gN2( ikappa );
   int TwoJ = denP->gTwoJ( ikappa );

   int theindex = denP->gIndex();
   int dimL     = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
         for ( int l_beta = l_alpha; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, S0tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, Atensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->get_irrep() );
               int memSkappa = denS->gKappa( NL + 2, TwoSL, ILdown, N1, N2, TwoJ, NR + 2, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 2, TwoSR, IRdown );

                  dcomplex * BlockS0 = S0tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                  dcomplex * BlockA  = Atensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                  dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

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

      for ( int l_gamma = theindex + 2; l_gamma < Prob->gL(); l_gamma++ ) {
         for ( int l_delta = l_gamma; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, Atensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, S0tensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]->get_irrep() );
               int memSkappa = denS->gKappa( NL + 2, TwoSL, ILdown, N1, N2, TwoJ, NR + 2, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 2, TwoSR, IRdown );

                  dcomplex * BlockA  = Atensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, ILdown );
                  dcomplex * BlockS0 = S0tensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IRdown );
                  dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

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

void CheMPS2::CHeffNS::addDiagram2a1spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator **** BTtensors, CTensorS1T **** S1Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denP->gNL( ikappa );
   int TwoSL = denP->gTwoSL( ikappa );
   int IL    = denP->gIL( ikappa );

   int NR    = denP->gNR( ikappa );
   int TwoSR = denP->gTwoSR( ikappa );
   int IR    = denP->gIR( ikappa );

   int N1   = denP->gN1( ikappa );
   int N2   = denP->gN2( ikappa );
   int TwoJ = denP->gTwoJ( ikappa );

   int theindex = denP->gIndex();
   int dimL     = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) ) {

               int fase                 = phase( TwoSRdown + TwoSL + TwoJ + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSR + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, S1Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, BTtensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL - 2, TwoSLdown, ILdown, N1, N2, TwoJ, NR - 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 2, TwoSRdown, IRdown );

                           dcomplex * BlockS1 = S1Ttensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                           dcomplex * BlockB  = BTtensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                           dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

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

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) ) {

               int fase                 = phase( TwoSRdown + TwoSL + TwoJ + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSR + 1 ) * ( TwoSL + 1.0 ) ) *
                                          Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_gamma = theindex + 2; l_gamma < Prob->gL(); l_gamma++ ) {
                  for ( int l_delta = l_gamma + 1; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, BTtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, S1Ttensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL - 2, TwoSLdown, ILdown, N1, N2, TwoJ, NR - 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 2, TwoSRdown, IRdown );

                           dcomplex * BlockB = BTtensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]
                                                   ->gStorage( NL, TwoSL, IL, NL - 2, TwoSLdown, ILdown );
                           dcomplex * BlockS1 = S1Ttensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]
                                                    ->gStorage( NR, TwoSR, IR, NR - 2, TwoSRdown, IRdown );
                           dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

void CheMPS2::CHeffNS::addDiagram2a2spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator **** Btensors, CTensorS1 **** S1tensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denP->gNL( ikappa );
   int TwoSL = denP->gTwoSL( ikappa );
   int IL    = denP->gIL( ikappa );

   int NR    = denP->gNR( ikappa );
   int TwoSR = denP->gTwoSR( ikappa );
   int IR    = denP->gIR( ikappa );

   int N1   = denP->gN1( ikappa );
   int N2   = denP->gN2( ikappa );
   int TwoJ = denP->gTwoJ( ikappa );

   int theindex = denP->gIndex();
   int dimL     = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) ) {

               int fase                 = phase( TwoSLdown + TwoSR + TwoJ + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSLdown + 1.0 ) ) *
                                          Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_beta = l_alpha + 1; l_beta < theindex; l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_alpha, l_beta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, S1tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, Btensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL + 2, TwoSLdown, ILdown, N1, N2, TwoJ, NR + 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 2, TwoSRdown, IRdown );

                           dcomplex * BlockS1 = S1tensors[ theindex - 1 ][ l_beta - l_alpha ][ theindex - 1 - l_beta ]
                                                    ->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                           dcomplex * BlockB = Btensors[ theindex + 1 ][ l_beta - l_alpha ][ theindex + 1 - l_beta ]
                                                   ->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                           dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) ) {

               int fase                 = phase( TwoSLdown + TwoSR + TwoJ + 2 );
               const dcomplex thefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSLdown + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_gamma = theindex + 2; l_gamma < Prob->gL(); l_gamma++ ) {
                  for ( int l_delta = l_gamma + 1; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_absigma( l_gamma, l_delta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Btensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, S1tensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL + 2, TwoSLdown, ILdown, N1, N2, TwoJ, NR + 2, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 2, TwoSRdown, IRdown );

                           dcomplex * BlockB = Btensors[ theindex - 1 ][ l_delta - l_gamma ][ l_gamma - theindex ]
                                                   ->gStorage( NL, TwoSL, IL, NL + 2, TwoSLdown, ILdown );
                           dcomplex * BlockS1 = S1tensors[ theindex + 1 ][ l_delta - l_gamma ][ l_gamma - theindex - 2 ]
                                                    ->gStorage( NR, TwoSR, IR, NR + 2, TwoSRdown, IRdown );
                           dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

void CheMPS2::CHeffNS::addDiagram2a3spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator **** Ctensors, CTensorOperator **** CTtensors, CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denP->gNL( ikappa );
   int TwoSL = denP->gTwoSL( ikappa );
   int IL    = denP->gIL( ikappa );

   int NR    = denP->gNR( ikappa );
   int TwoSR = denP->gTwoSR( ikappa );
   int IR    = denP->gIR( ikappa );

   int N1   = denP->gN1( ikappa );
   int N2   = denP->gN2( ikappa );
   int TwoJ = denP->gTwoJ( ikappa );

   int theindex = denP->gIndex();
   int dimL     = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
         for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_gamma, l_alpha ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, F0tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, Ctensors[ theindex + 1 ][ l_alpha - l_gamma ][ theindex + 1 - l_alpha ]->get_irrep() );
               int memSkappa = denS->gKappa( NL, TwoSL, ILdown, N1, N2, TwoJ, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IRdown );

                  //no transpose
                  dcomplex * ptr = Ctensors[ theindex + 1 ][ l_alpha - l_gamma ][ theindex + 1 - l_alpha ]
                                       ->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockF0 = F0tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]
                                           ->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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
               int IRdown    = Irreps::directProd( IR, CTtensors[ theindex + 1 ][ l_gamma - l_alpha ][ theindex + 1 - l_gamma ]->get_irrep() );
               int memSkappa = denS->gKappa( NL, TwoSL, ILdown, N1, N2, TwoJ, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IRdown );

                  //transpose
                  dcomplex * ptr = CTtensors[ theindex + 1 ][ l_gamma - l_alpha ][ theindex + 1 - l_gamma ]
                                       ->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockF0 = F0Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]
                                           ->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

      for ( int l_delta = theindex + 2; l_delta < Prob->gL(); l_delta++ ) {
         for ( int l_beta = l_delta + 1; l_beta < Prob->gL(); l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_delta, l_beta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, Ctensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, F0tensors[ theindex + 1 ][ l_beta - l_delta ][ l_delta - theindex - 2 ]->get_irrep() );
               int memSkappa = denS->gKappa( NL, TwoSL, ILdown, N1, N2, TwoJ, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IRdown );

                  //no transpose
                  dcomplex * ptr = Ctensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]
                                       ->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * BlockF0 = F0tensors[ theindex + 1 ][ l_beta - l_delta ][ l_delta - theindex - 2 ]
                                           ->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

      for ( int l_beta = theindex + 2; l_beta < Prob->gL(); l_beta++ ) {
         for ( int l_delta = l_beta; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIchemps2::owner_cdf( Prob->gL(), l_beta, l_delta ) == MPIRANK )
#endif
            {
               int ILdown    = Irreps::directProd( IL, CTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->get_irrep() );
               int IRdown    = Irreps::directProd( IR, F0Ttensors[ theindex + 1 ][ l_delta - l_beta ][ l_beta - theindex - 2 ]->get_irrep() );
               int memSkappa = denS->gKappa( NL, TwoSL, ILdown, N1, N2, TwoJ, NR, TwoSR, IRdown );

               if ( memSkappa != -1 ) {

                  int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSL, ILdown );
                  int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IRdown );

                  //transpose
                  dcomplex * ptr = CTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]
                                       ->gStorage( NL, TwoSL, IL, NL, TwoSL, ILdown );
                  dcomplex * BlockF0 = F0Ttensors[ theindex + 1 ][ l_delta - l_beta ][ l_beta - theindex - 2 ]
                                           ->gStorage( NR, TwoSR, IR, NR, TwoSR, IRdown );
                  dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

void CheMPS2::CHeffNS::addDiagram2a3spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator **** Dtensors, CTensorOperator **** DTtensors, CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors, dcomplex * workspace ) {

   char cotrans = 'C';
   char notrans = 'N';

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   int NL    = denP->gNL( ikappa );
   int TwoSL = denP->gTwoSL( ikappa );
   int IL    = denP->gIL( ikappa );

   int NR    = denP->gNR( ikappa );
   int TwoSR = denP->gTwoSR( ikappa );
   int IR    = denP->gIR( ikappa );

   int N1   = denP->gN1( ikappa );
   int N2   = denP->gN2( ikappa );
   int TwoJ = denP->gTwoJ( ikappa );

   int theindex = denP->gIndex();
   int dimL     = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
   int dimR     = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

   const bool leftSum = ( theindex < Prob->gL() * 0.5 ) ? true : false;

   if ( leftSum ) {

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {
         for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) ) {

               int fase           = phase( TwoSLdown + TwoSRdown + TwoJ + 2 );
               dcomplex prefactor = fase * sqrt( ( TwoSR + 1 ) * ( TwoSLdown + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_gamma = 0; l_gamma < theindex; l_gamma++ ) {
                  for ( int l_alpha = l_gamma + 1; l_alpha < theindex; l_alpha++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_gamma, l_alpha ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, F1tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, Dtensors[ theindex + 1 ][ l_alpha - l_gamma ][ theindex + 1 - l_alpha ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL, TwoSLdown, ILdown, N1, N2, TwoJ, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSRdown, IRdown );

                           //no transpose
                           dcomplex * ptr = Dtensors[ theindex + 1 ][ l_alpha - l_gamma ][ theindex + 1 - l_alpha ]
                                                ->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockF1 = F1tensors[ theindex - 1 ][ l_alpha - l_gamma ][ theindex - 1 - l_alpha ]
                                                    ->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

               fase      = phase( TwoSL + TwoSR + TwoJ + 2 );
               prefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_alpha = 0; l_alpha < theindex; l_alpha++ ) {
                  for ( int l_gamma = l_alpha; l_gamma < theindex; l_gamma++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_alpha, l_gamma ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, F1Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, DTtensors[ theindex + 1 ][ l_gamma - l_alpha ][ theindex + 1 - l_gamma ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL, TwoSLdown, ILdown, N1, N2, TwoJ, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSRdown, IRdown );

                           //transpose
                           dcomplex * ptr = DTtensors[ theindex + 1 ][ l_gamma - l_alpha ][ theindex + 1 - l_gamma ]
                                                ->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockF1 = F1Ttensors[ theindex - 1 ][ l_gamma - l_alpha ][ theindex - 1 - l_gamma ]
                                                    ->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

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

            if ( ( TwoSLdown >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= TwoJ ) ) {

               int fase           = phase( TwoSLdown + TwoSRdown + TwoJ + 2 );
               dcomplex prefactor = fase * sqrt( ( TwoSR + 1 ) * ( TwoSLdown + 1.0 ) ) *
                                    Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_delta = theindex + 2; l_delta < Prob->gL(); l_delta++ ) {
                  for ( int l_beta = l_delta + 1; l_beta < Prob->gL(); l_beta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_delta, l_beta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, Dtensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, F1tensors[ theindex + 1 ][ l_beta - l_delta ][ l_delta - theindex - 2 ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL, TwoSLdown, ILdown, N1, N2, TwoJ, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSRdown, IRdown );

                           //no transpose
                           dcomplex * ptr     = Dtensors[ theindex - 1 ][ l_beta - l_delta ][ l_delta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * BlockF1 = F1tensors[ theindex + 1 ][ l_beta - l_delta ][ l_delta - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

                           dcomplex beta = 0.0;
                           zgemm_( &notrans, &notrans, &dimL, &dimRdown, &dimLdown, &prefactor, ptr, &dimL, BlockS, &dimLdown, &beta, workspace, &dimL );

                           beta = 1.0;
                           zgemm_( &notrans, &cotrans, &dimL, &dimR, &dimRdown, &beta, workspace, &dimL, BlockF1, &dimR, &beta, memHeff, &dimL );
                        }
                     }
                  }
               }

               fase      = phase( TwoSL + TwoSR + TwoJ + 2 );
               prefactor = fase * sqrt( ( TwoSRdown + 1 ) * ( TwoSL + 1.0 ) ) * Wigner::wigner6j( TwoSLdown, TwoSRdown, TwoJ, TwoSR, TwoSL, 2 );

               for ( int l_beta = theindex + 2; l_beta < Prob->gL(); l_beta++ ) {
                  for ( int l_delta = l_beta; l_delta < Prob->gL(); l_delta++ ) {

#ifdef CHEMPS2_MPI_COMPILATION
                     if ( MPIchemps2::owner_cdf( Prob->gL(), l_beta, l_delta ) == MPIRANK )
#endif
                     {
                        int ILdown    = Irreps::directProd( IL, DTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->get_irrep() );
                        int IRdown    = Irreps::directProd( IR, F1Ttensors[ theindex + 1 ][ l_delta - l_beta ][ l_beta - theindex - 2 ]->get_irrep() );
                        int memSkappa = denS->gKappa( NL, TwoSLdown, ILdown, N1, N2, TwoJ, NR, TwoSRdown, IRdown );

                        if ( memSkappa != -1 ) {

                           int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, ILdown );
                           int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSRdown, IRdown );

                           //transpose
                           dcomplex * ptr     = DTtensors[ theindex - 1 ][ l_delta - l_beta ][ l_beta - theindex ]->gStorage( NL, TwoSL, IL, NL, TwoSLdown, ILdown );
                           dcomplex * BlockF1 = F1Ttensors[ theindex + 1 ][ l_delta - l_beta ][ l_beta - theindex - 2 ]->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IRdown );
                           dcomplex * BlockS  = denS->gStorage() + denS->gKappa2index( memSkappa );

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

void CheMPS2::CHeffNS::addDiagram2b1and2b2( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = denP->gN1( ikappa );

   if ( N1 == 0 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );

      int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, IL );

      if ( dimLdown > 0 ) {

         int NR    = denP->gNR( ikappa );
         int TwoSR = denP->gTwoSR( ikappa );
         int IR    = denP->gIR( ikappa );

         int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

         int memSkappa = denS->gKappa( NL - 2, TwoSL, IL, 2, denP->gN2( ikappa ), denP->gTwoJ( ikappa ), NR, TwoSR, IR );

         if ( memSkappa != -1 ) {

            dcomplex * BlockA = ATtensor->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, IL );
            dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );

            dcomplex alpha = sqrt( 2.0 );
            dcomplex beta  = 1.0;
            zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, BlockA, &dimLup, BlockS, &dimLdown, &beta, memHeff, &dimLup );
         }
      }
   }

   if ( N1 == 2 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );

      int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, IL );

      if ( dimLdown > 0 ) {

         int NR    = denP->gNR( ikappa );
         int TwoSR = denP->gTwoSR( ikappa );
         int IR    = denP->gIR( ikappa );

         int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

         int memSkappa = denS->gKappa( NL + 2, TwoSL, IL, 0, denP->gN2( ikappa ), denP->gTwoJ( ikappa ), NR, TwoSR, IR );

         if ( memSkappa != -1 ) {

            dcomplex * BlockA = Atensor->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );
            dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );
            dcomplex alpha    = sqrt( 2.0 );
            dcomplex beta     = 1.0;
            zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, BlockA, &dimLup, BlockS, &dimLdown, &beta, memHeff, &dimLup );
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2c1and2c2( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N2 = denP->gN2( ikappa );

   if ( N2 == 0 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );

      int dimLdown = bk_down->gCurrentDim( theindex, NL - 2, TwoSL, IL );

      if ( dimLdown > 0 ) {

         int NR    = denP->gNR( ikappa );
         int TwoSR = denP->gTwoSR( ikappa );
         int IR    = denP->gIR( ikappa );

         int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

         int memSkappa = denS->gKappa( NL - 2, TwoSL, IL, denP->gN1( ikappa ), 2, denP->gTwoJ( ikappa ), NR, TwoSR, IR );

         if ( memSkappa != -1 ) {

            dcomplex * BlockA = ATtensor->gStorage( NL, TwoSL, IL, NL - 2, TwoSL, IL );
            dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );
            dcomplex alpha    = sqrt( 2.0 );
            dcomplex beta     = 1.0;

            zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, BlockA, &dimLup, BlockS, &dimLdown, &beta, memHeff, &dimLup );
         }
      }
   }

   if ( N2 == 2 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );

      int dimLdown = bk_down->gCurrentDim( theindex, NL + 2, TwoSL, IL );

      if ( dimLdown > 0 ) {

         int NR    = denP->gNR( ikappa );
         int TwoSR = denP->gTwoSR( ikappa );
         int IR    = denP->gIR( ikappa );

         int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
         int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

         int memSkappa = denS->gKappa( NL + 2, TwoSL, IL, denP->gN1( ikappa ), 0, denP->gTwoJ( ikappa ), NR, TwoSR, IR );

         if ( memSkappa != -1 ) {

            dcomplex * BlockA = Atensor->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );
            dcomplex * BlockS = denS->gStorage() + denS->gKappa2index( memSkappa );
            dcomplex alpha    = sqrt( 2.0 );
            dcomplex beta     = 1.0;
            zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, BlockA, &dimLup, BlockS, &dimLdown, &beta, memHeff, &dimLup );
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2dall( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP ) {

   const int N1       = denP->gN1( ikappa );
   const int N2       = denP->gN2( ikappa );
   const int theindex = denP->gIndex();
   int size           = bk_down->gCurrentDim( theindex, denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ) ) * bk_down->gCurrentDim( theindex + 2, denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );
   int inc            = 1;

   if ( ( N1 == 2 ) && ( N2 == 0 ) ) { //2d1

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), 0, 2, 0, denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex factor = Prob->gMxElement( theindex, theindex, theindex + 1, theindex + 1 );
         zaxpy_( &size, &factor, denS->gStorage() + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 0 ) && ( N2 == 2 ) ) { //2d2

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), 2, 0, 0, denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex factor = Prob->gMxElement( theindex, theindex, theindex + 1, theindex + 1 );
         zaxpy_( &size, &factor, denS->gStorage() + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 2 ) && ( N2 == 2 ) ) { //2d3a

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex factor = 4 * Prob->gMxElement( theindex, theindex + 1, theindex, theindex + 1 ) - 2 * Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex );
         zaxpy_( &size, &factor, denS->gStorage() + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 1 ) && ( N2 == 1 ) ) { //2d3b

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         int fase        = ( denS->gTwoJ( ikappa ) == 0 ) ? 1 : -1;
         dcomplex factor = Prob->gMxElement( theindex, theindex + 1, theindex, theindex + 1 ) + fase * Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex );
         zaxpy_( &size, &factor, denS->gStorage() + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 2 ) && ( N2 == 1 ) ) { //2d3c

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex factor = 2 * Prob->gMxElement( theindex, theindex + 1, theindex, theindex + 1 ) - Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex );
         zaxpy_( &size, &factor, denS->gStorage() + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }

   if ( ( N1 == 1 ) && ( N2 == 2 ) ) { //2d3d

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex factor = 2 * Prob->gMxElement( theindex, theindex + 1, theindex, theindex + 1 ) - Prob->gMxElement( theindex, theindex + 1, theindex + 1, theindex );
         zaxpy_( &size, &factor, denS->gStorage() + denS->gKappa2index( memSkappa ), &inc, memHeff, &inc );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2e1and2e2( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = denP->gN1( ikappa );

   if ( N1 == 2 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );
      int NR       = denP->gNR( ikappa );
      int TwoSR    = denP->gTwoSR( ikappa );
      int IR       = denP->gIR( ikappa );

      int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 2, TwoSR, IR );
      int dimRup   = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 0, denP->gN2( ikappa ), denP->gTwoJ( ikappa ), NR - 2, TwoSR, IR );

      if ( memSkappa != -1 ) {

         dcomplex * BlockA = ATtensor->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IR );
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }

   if ( N1 == 0 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );
      int NR       = denP->gNR( ikappa );
      int TwoSR    = denP->gTwoSR( ikappa );
      int IR       = denP->gIR( ikappa );

      int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 2, TwoSR, IR );
      int dimRup   = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      int memSkappa = denS->gKappa( NL, TwoSL, IL, 2, denP->gN2( ikappa ), denP->gTwoJ( ikappa ), NR + 2, TwoSR, IR );

      if ( memSkappa != -1 ) {

         dcomplex * BlockA = Atensor->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IR );
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2f1and2f2( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Atensor, CTensorOperator * ATtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N2 = denP->gN2( ikappa );

   if ( N2 == 2 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );
      int NR       = denP->gNR( ikappa );
      int TwoSR    = denP->gTwoSR( ikappa );
      int IR       = denP->gIR( ikappa );

      int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = bk_down->gCurrentDim( theindex + 2, NR - 2, TwoSR, IR );
      int dimRup   = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      int memSkappa = denS->gKappa( NL, TwoSL, IL, denP->gN1( ikappa ), 0, denP->gTwoJ( ikappa ), NR - 2, TwoSR, IR );

      if ( memSkappa != -1 ) {

         dcomplex * BlockA = ATtensor->gStorage( NR, TwoSR, IR, NR - 2, TwoSR, IR );
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }

   if ( N2 == 0 ) {

      int theindex = denP->gIndex();
      int NL       = denP->gNL( ikappa );
      int TwoSL    = denP->gTwoSL( ikappa );
      int IL       = denP->gIL( ikappa );
      int NR       = denP->gNR( ikappa );
      int TwoSR    = denP->gTwoSR( ikappa );
      int IR       = denP->gIR( ikappa );

      int dimL     = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRdown = bk_down->gCurrentDim( theindex + 2, NR + 2, TwoSR, IR );
      int dimRup   = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      int memSkappa = denS->gKappa( NL, TwoSL, IL, denP->gN1( ikappa ), 2, denP->gTwoJ( ikappa ), NR + 2, TwoSR, IR );

      if ( memSkappa != -1 ) {

         dcomplex * BlockA = Atensor->gStorage( NR, TwoSR, IR, NR + 2, TwoSR, IR );
         dcomplex alpha    = sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, BlockA, &dimRup, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2b3spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = denP->gN1( ikappa );

   if ( N1 != 0 ) {

      int theindex = denP->gIndex();

      int NL    = denP->gNL( ikappa );
      int TwoSL = denP->gTwoSL( ikappa );
      int IL    = denP->gIL( ikappa );

      int dimLU = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR  = bk_down->gCurrentDim( theindex + 2, denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
         dcomplex alpha    = ( ( N1 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &notrans, &dimLU, &dimR, &dimLD, &alpha, Cblock, &dimLU, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimLD, &beta, memHeff, &dimLU );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2c3spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N2 = denP->gN2( ikappa );

   if ( N2 != 0 ) {

      int theindex = denP->gIndex();

      int NL    = denP->gNL( ikappa );
      int TwoSL = denP->gTwoSL( ikappa );
      int IL    = denP->gIL( ikappa );

      int dimLU = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimLD = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR  = bk_down->gCurrentDim( theindex + 2, denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
         dcomplex alpha    = ( ( N2 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &notrans, &dimLU, &dimR, &dimLD, &alpha, Cblock, &dimLU, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimLD, &beta, memHeff, &dimLU );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2e3spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = denP->gN1( ikappa );

   if ( N1 != 0 ) {

      int theindex = denP->gIndex();

      int NR    = denP->gNR( ikappa );
      int TwoSR = denP->gTwoSR( ikappa );
      int IR    = denP->gIR( ikappa );

      int dimRU = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );
      int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );
      int dimL  = bk_down->gCurrentDim( theindex, denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ) );

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         dcomplex alpha    = ( ( N1 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRU, &dimRD, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, Cblock, &dimRU, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2f3spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * CTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N2 = denP->gN2( ikappa );

   if ( N2 != 0 ) {

      int theindex = denP->gIndex();

      int NR    = denP->gNR( ikappa );
      int TwoSR = denP->gTwoSR( ikappa );
      int IR    = denP->gIR( ikappa );

      int dimRU = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );
      int dimRD = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );
      int dimL  = bk_down->gCurrentDim( theindex, denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ) );

      int memSkappa = denS->gKappa( denP->gNL( ikappa ), denP->gTwoSL( ikappa ), denP->gIL( ikappa ), denP->gN1( ikappa ), denP->gN2( ikappa ), denP->gTwoJ( ikappa ), denP->gNR( ikappa ), denP->gTwoSR( ikappa ), denP->gIR( ikappa ) );

      if ( memSkappa != -1 ) {
         dcomplex * Cblock = CTtensor->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
         dcomplex alpha    = ( ( N2 == 2 ) ? 1.0 : 0.5 ) * sqrt( 2.0 );
         dcomplex beta     = 1.0;
         zgemm_( &notrans, &cotrans, &dimL, &dimRU, &dimRD, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, Cblock, &dimRU, &beta, memHeff, &dimL );
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2b3spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = denP->gN1( ikappa );
   if ( N1 == 1 ) {

      int theindex = denP->gIndex();

      int NL    = denP->gNL( ikappa );
      int TwoSL = denP->gTwoSL( ikappa );
      int IL    = denP->gIL( ikappa );
      int NR    = denP->gNR( ikappa );
      int TwoSR = denP->gTwoSR( ikappa );
      int IR    = denP->gIR( ikappa );
      int TwoJ  = denP->gTwoJ( ikappa );
      int N2    = denP->gN2( ikappa );

      int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, IL );
         if ( dimLdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NL, TwoSL, IL, NL, TwoSLdown, IL );

            int TwoS2     = ( N2 == 1 ) ? 1 : 0;
            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;

            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSLdown, IL, N1, N2, TwoJdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {

                     int fase       = phase( TwoSLdown + TwoSR + TwoJ + TwoS2 + TwoJdown - 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1 ) * ( TwoJdown + 1 ) * ( TwoSL + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta  = 1.0;
                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Dblock, &dimLup, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2c3spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N2 = denP->gN2( ikappa );

   if ( N2 == 1 ) {

      int theindex = denP->gIndex();

      int NL    = denP->gNL( ikappa );
      int TwoSL = denP->gTwoSL( ikappa );
      int IL    = denP->gIL( ikappa );
      int NR    = denP->gNR( ikappa );
      int TwoSR = denP->gTwoSR( ikappa );
      int IR    = denP->gIR( ikappa );
      int TwoJ  = denP->gTwoJ( ikappa );
      int N1    = denP->gN1( ikappa );

      int dimLup = bk_up->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimR   = bk_down->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      for ( int TwoSLdown = TwoSL - 2; TwoSLdown <= TwoSL + 2; TwoSLdown += 2 ) {

         int dimLdown = bk_down->gCurrentDim( theindex, NL, TwoSLdown, IL );

         if ( dimLdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NL, TwoSL, IL, NL, TwoSLdown, IL );

            int TwoS1     = ( N1 == 1 ) ? 1 : 0;
            int TwoJstart = ( ( TwoSR != TwoSLdown ) || ( TwoS1 == 0 ) ) ? 1 + TwoS1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS1; TwoJdown += 2 ) {
               if ( abs( TwoSLdown - TwoSR ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSLdown, IL, N1, N2, TwoJdown, NR, TwoSR, IR );
                  if ( memSkappa != -1 ) {

                     int fase       = phase( TwoSLdown + TwoSR + 2 * TwoJ + TwoS1 - 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1 ) * ( TwoJdown + 1 ) * ( TwoSL + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS1 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSL, TwoSLdown, TwoSR );
                     dcomplex beta  = 1.0;
                     zgemm_( &notrans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Dblock, &dimLup, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimLdown, &beta, memHeff, &dimLup );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2e3spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N1 = denP->gN1( ikappa );
   if ( N1 == 1 ) {

      int theindex = denP->gIndex();

      int NL    = denP->gNL( ikappa );
      int TwoSL = denP->gTwoSL( ikappa );
      int IL    = denP->gIL( ikappa );
      int NR    = denP->gNR( ikappa );
      int TwoSR = denP->gTwoSR( ikappa );
      int IR    = denP->gIR( ikappa );
      int TwoJ  = denP->gTwoJ( ikappa );
      int N2    = denP->gN2( ikappa );

      int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRup = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSRdown, IR );

         if ( dimRdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IR );

            int TwoS2     = ( N2 == 1 ) ? 1 : 0;
            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS2 == 0 ) ) ? 1 + TwoS2 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS2; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSL, IL, N1, N2, TwoJdown, NR, TwoSRdown, IR );

                  if ( memSkappa != -1 ) {

                     int fase       = phase( TwoSRdown + TwoSL + 2 * TwoJ + TwoS2 + 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1 ) * ( TwoJdown + 1 ) * ( TwoSRdown + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS2 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta  = 1.0;
                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, Dblock, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CHeffNS::addDiagram2f3spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * DTtensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   int N2 = denP->gN2( ikappa );
   if ( N2 == 1 ) {

      int theindex = denP->gIndex();

      int NL    = denP->gNL( ikappa );
      int TwoSL = denP->gTwoSL( ikappa );
      int IL    = denP->gIL( ikappa );
      int NR    = denP->gNR( ikappa );
      int TwoSR = denP->gTwoSR( ikappa );
      int IR    = denP->gIR( ikappa );
      int TwoJ  = denP->gTwoJ( ikappa );
      int N1    = denP->gN1( ikappa );

      int dimL   = bk_down->gCurrentDim( theindex, NL, TwoSL, IL );
      int dimRup = bk_up->gCurrentDim( theindex + 2, NR, TwoSR, IR );

      for ( int TwoSRdown = TwoSR - 2; TwoSRdown <= TwoSR + 2; TwoSRdown += 2 ) {

         int dimRdown = bk_down->gCurrentDim( theindex + 2, NR, TwoSRdown, IR );
         if ( dimRdown > 0 ) {

            dcomplex * Dblock = DTtensor->gStorage( NR, TwoSR, IR, NR, TwoSRdown, IR );

            int TwoS1     = ( N1 == 1 ) ? 1 : 0;
            int TwoJstart = ( ( TwoSRdown != TwoSL ) || ( TwoS1 == 0 ) ) ? 1 + TwoS1 : 0;
            for ( int TwoJdown = TwoJstart; TwoJdown <= 1 + TwoS1; TwoJdown += 2 ) {
               if ( abs( TwoSL - TwoSRdown ) <= TwoJdown ) {

                  int memSkappa = denS->gKappa( NL, TwoSL, IL, N1, N2, TwoJdown, NR, TwoSRdown, IR );

                  if ( memSkappa != -1 ) {

                     int fase       = phase( TwoSRdown + TwoSL + TwoJ + TwoS1 + TwoJdown + 1 );
                     dcomplex alpha = fase * sqrt( 3.0 * ( TwoJ + 1 ) * ( TwoJdown + 1 ) * ( TwoSRdown + 1 ) ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, 1, 1, TwoS1 ) * Wigner::wigner6j( TwoJdown, TwoJ, 2, TwoSR, TwoSRdown, TwoSL );
                     dcomplex beta  = 1.0;

                     zgemm_( &notrans, &cotrans, &dimL, &dimRup, &dimRdown, &alpha, denS->gStorage() + denS->gKappa2index( memSkappa ), &dimL, Dblock, &dimRup, &beta, memHeff, &dimL );
                  }
               }
            }
         }
      }
   }
}