
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CTensorX.h"
#include "Lapack.h"
#include "Wigner.h"

CheMPS2::CTensorX::CTensorX( const int boundary_index, const bool moving_right, const SyBookkeeper * bk_up,
                             const SyBookkeeper * bk_down, const Problem * Prob )
    : CTensorOperator( boundary_index, // the index
                       0,              // two_j
                       0,              // n_elec
                       0,              // n_irrep
                       moving_right,   // direction
                       true,           // prime_last (doesn't matter for spin-0 tensors)
                       false,          // jw_phase (four 2nd quantized operators)
                       bk_up,          // upper bookkeeper
                       bk_down         // lower bookeeper
      ) {
   this->Prob = Prob;
}

CheMPS2::CTensorX::~CTensorX() {}

void CheMPS2::CTensorX::update( CTensorT * denTup, CTensorT * denTdown ) {
   if ( moving_right ) {
// PARALLEL
#pragma omp parallel for schedule( dynamic )
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         makenewRight( ikappa, denTup, denTdown, NULL, NULL );
      }
   } else {
// PARALLEL
#pragma omp parallel for schedule( dynamic )
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         makenewLeft( ikappa, denTup, denTdown, NULL, NULL );
      }
   }
}

void CheMPS2::CTensorX::update( CTensorT * denTup, CTensorT * denTdown, CTensorO * overlap,
                                CTensorL ** Ltensors, CTensorLT ** LtensorsT,
                                CTensorOperator * Xtensor, CTensorQ * Qtensor, CTensorQT * QtensorT,
                                CTensorOperator * Atensor, CTensorOperator * AtensorT,
                                CTensorOperator * CtensorT, CTensorOperator * DtensorT ) {
   clear();

   if ( moving_right ) {
// PARALLEL
#pragma omp parallel
      {
         const bool doOtherThings = ( index > 1 ) ? true : false;
         const int dimLU          = ( doOtherThings ) ? bk_up->gMaxDimAtBound( index - 1 ) : 0;
         const int dimLD          = ( doOtherThings ) ? bk_down->gMaxDimAtBound( index - 1 ) : 0;
         const int dimRU          = ( doOtherThings ) ? bk_up->gMaxDimAtBound( index ) : 0;
         const int dimRD          = ( doOtherThings ) ? bk_down->gMaxDimAtBound( index ) : 0;
         dcomplex * workmemLULD   = ( doOtherThings ) ? new dcomplex[ dimLU * dimLD ] : NULL;
         dcomplex * workmemRULD   = ( doOtherThings ) ? new dcomplex[ dimRU * dimLD ] : NULL;
         dcomplex * workmemRURD   = ( doOtherThings ) ? new dcomplex[ dimRU * dimRD ] : NULL;

#pragma omp for schedule( dynamic )
         for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {

            makenewRight( ikappa, denTup, denTdown, overlap, workmemRULD );

            if ( doOtherThings ) {
               update_moving_right( ikappa, Xtensor, denTup, denTdown, workmemRULD );

               addTermQLRight( ikappa, denTup, denTdown, Ltensors, Qtensor, workmemRURD, workmemRULD, workmemLULD );

               addTermQTLTRight( ikappa, denTup, denTdown, LtensorsT, QtensorT, workmemRURD, workmemRULD, workmemLULD );

               addTermARight( ikappa, denTup, denTdown, Atensor, workmemRURD, workmemRULD );

               addTermATRight( ikappa, denTup, denTdown, AtensorT, workmemRURD, workmemRULD );

               addTermCRight( ikappa, denTup, denTdown, CtensorT, workmemRULD );

               addTermDRight( ikappa, denTup, denTdown, DtensorT, workmemRULD );
            }
         }

         if ( doOtherThings ) {
            delete[] workmemLULD;
            delete[] workmemRULD;
            delete[] workmemRURD;
         }
      }
   } else {
// PARALLEL
#pragma omp parallel
      {
         const bool doOtherThings = ( index < Prob->gL() - 1 ) ? true : false;
         const int dimLU          = ( doOtherThings ) ? bk_up->gMaxDimAtBound( index ) : 0;
         const int dimLD          = ( doOtherThings ) ? bk_down->gMaxDimAtBound( index ) : 0;
         const int dimRU          = ( doOtherThings ) ? bk_up->gMaxDimAtBound( index + 1 ) : 0;
         const int dimRD          = ( doOtherThings ) ? bk_down->gMaxDimAtBound( index + 1 ) : 0;
         dcomplex * workmemLULD   = ( doOtherThings ) ? new dcomplex[ dimLU * dimLD ] : NULL;
         dcomplex * workmemLURD   = ( doOtherThings ) ? new dcomplex[ dimLU * dimRD ] : NULL;
         dcomplex * workmemRURD   = ( doOtherThings ) ? new dcomplex[ dimRU * dimRD ] : NULL;

#pragma omp for schedule( dynamic )
         for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {

            makenewLeft( ikappa, denTup, denTdown, overlap, workmemLURD );

            if ( doOtherThings ) {
               update_moving_left( ikappa, Xtensor, denTup, denTdown, workmemLURD );

               addTermQLLeft( ikappa, denTup, denTdown, Ltensors, Qtensor, workmemLULD, workmemLURD, workmemRURD );

               addTermQTLTLeft( ikappa, denTup, denTdown, LtensorsT, QtensorT, workmemLULD, workmemLURD, workmemRURD );

               addTermALeft( ikappa, denTup, denTdown, Atensor, workmemLURD, workmemLULD );

               addTermATLeft( ikappa, denTup, denTdown, AtensorT, workmemLURD, workmemLULD );

               addTermCLeft( ikappa, denTup, denTdown, CtensorT, workmemLURD );

               addTermDLeft( ikappa, denTup, denTdown, DtensorT, workmemLURD );
            }
         }
         if ( doOtherThings ) {
            delete[] workmemLULD;
            delete[] workmemLURD;
            delete[] workmemRURD;
         }
      }
   }
}

void CheMPS2::CTensorX::makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                                      CTensorO * overlap, dcomplex * workmem ) {

   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int NL    = NRU - 2;
   int TwoSL = TwoSRU;
   int IL    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRU, IRU );
   int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
   int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );

   dcomplex alpha = Prob->gMxElement( index - 1, index - 1, index - 1, index - 1 );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 && ( abs( alpha ) > 0.0 ) ) {
      if ( overlap == NULL ) {
         assert( dimLU == dimLD );

         dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );

         dcomplex beta = 0.0; // because there's only 1 term contributing per kappa,
                              // we might as well set it i.o. adding

         zgemm_( &cotrans, &notrans, &dimRU, &dimRD, &dimLU, &alpha,
                 BlockTup, &dimLU, BlockTdown, &dimLU, &beta, storage + kappa2index[ ikappa ], &dimRU );
      } else {
         dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );
         dcomplex * BlockPrev  = overlap->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

         dcomplex set = 0.0;
         zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha,
                 BlockTup, &dimLU, BlockPrev, &dimLU, &set, workmem, &dimRU );

         alpha        = 1.0;
         dcomplex one = 1.0;
         zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &one,
                 workmem, &dimRU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimRU );
      }

   } else {
      for ( int cnt = kappa2index[ ikappa ]; cnt < kappa2index[ ikappa + 1 ]; cnt++ ) {
         storage[ cnt ] = 0.0;
      }
   }
}

void CheMPS2::CTensorX::makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                                     CTensorO * overlap, dcomplex * workmem ) {

   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   const int NR    = NLU + 2;
   const int TwoSR = TwoSLU;
   const int IR    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );
   int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );

   dcomplex alpha = Prob->gMxElement( index, index, index, index );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 && ( abs( alpha ) > 0.0 ) ) {
      if ( overlap == NULL ) {
         assert( dimRU == dimRD );

         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );

         dcomplex beta = 0.0; // set, not add (only 1 term)
         zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRU, &alpha,
                 BlockTup, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );

      } else {
         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );
         dcomplex * BlockO     = overlap->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

         dcomplex beta = 0.0; // set
         zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha,
                 BlockTup, &dimLU, BlockO, &dimRU, &beta, workmem, &dimLU );

         alpha = 1.0;
         beta  = 1.0;
         zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha,
                 workmem, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
      }

   } else {
      for ( int cnt = kappa2index[ ikappa ]; cnt < kappa2index[ ikappa + 1 ]; cnt++ ) {
         storage[ cnt ] = 0.0;
      }
   }
}

void CheMPS2::CTensorX::addTermQLRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                                        CTensorL ** Lprev, CTensorQ * Qprev, dcomplex * workmemRR,
                                        dcomplex * workmemLR, dcomplex * workmemLL ) {
   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   if ( dimRU > 0 && dimRD > 0 ) {

      int dimTot = dimRU * dimRD;
      for ( int cnt = 0; cnt < dimTot; cnt++ ) {
         workmemRR[ cnt ] = 0.0;
      }

      char cotrans = 'C';
      char notrans = 'N';

      for ( int geval = 0; geval < 4; geval++ ) {
         int NLU, TwoSLU, ILU, NLD, TwoSLD, ILD;
         switch ( geval ) {
            case 0:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU;
               TwoSLD = TwoSRU;
               ILD    = IRU;
               break;
            case 1:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU + 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU;
               TwoSLD = TwoSRU;
               ILD    = IRU;
               break;
            case 2:
               NLU    = NRU - 2;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU - 1;
               TwoSLD = TwoSRU - 1;
               ILD    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               break;
            case 3:
               NLU    = NRU - 2;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU - 1;
               TwoSLD = TwoSRU + 1;
               ILD    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               break;
         }
         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );

         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockQ     = Qprev->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

            dcomplex factor;
            dcomplex * ptr;
            dcomplex * ptrTmp;
            if ( geval < 2 ) {
               factor = 1.0;
               ptr    = BlockQ;

            } else {
               int fase = ( ( ( ( TwoSRU + 1 - TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               factor   = fase * sqrt( ( TwoSLD + 1.0 ) / ( TwoSRU + 1.0 ) );

               int dimLUD = dimLU * dimLD;
               int inc    = 1;
               ptrTmp     = workmemLL;
               zcopy_( &dimLUD, BlockQ, &inc, ptrTmp, &inc );

               for ( int loca = 0; loca < index - 1; loca++ ) {
                  if ( bk_up->gIrrep( index - 1 ) == bk_up->gIrrep( loca ) ) {
                     dcomplex alpha    = Prob->gMxElement( loca, index - 1, index - 1, index - 1 );
                     dcomplex * BlockL = Lprev[ index - 2 - loca ]->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
                     zaxpy_( &dimLUD, &alpha, BlockL, &inc, ptrTmp, &inc );
                  }
               }
               ptr = ptrTmp;
            }

            // factor * Tup^T * L --> mem2 //set
            dcomplex beta = 0.0;
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &factor,
                    BlockTup, &dimLU, ptr, &dimLU, &beta, workmemLR, &dimRU );

            // mem2 * Tdown --> mem //add
            factor = 1.0;
            beta   = 1.0;
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &factor,
                    workmemLR, &dimRU, BlockTdown, &dimLD, &beta, workmemRR, &dimRU );
         }
      }

      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemRR, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermQTLTRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorLT ** LprevT,
                                          CTensorQT * QprevT, dcomplex * workmemRR, dcomplex * workmemLR, dcomplex * workmemLL ) {

   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   if ( dimRU > 0 && dimRD > 0 ) {

      int dimTot = dimRU * dimRD;
      for ( int cnt = 0; cnt < dimTot; cnt++ ) {
         workmemRR[ cnt ] = 0.0;
      }

      char cotrans = 'C';
      char notrans = 'N';

      for ( int geval = 0; geval < 4; geval++ ) {
         int NLU, TwoSLU, ILU, NLD, TwoSLD, ILD;
         switch ( geval ) {
            case 0:
               NLU    = NRU;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU - 1;
               TwoSLD = TwoSRU + 1;
               ILD    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               ;
               break;
            case 1:
               NLU    = NRU;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU - 1;
               TwoSLD = TwoSRU - 1;
               ILD    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               ;
               break;
            case 2:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU + 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU - 2;
               TwoSLD = TwoSRU;
               ILD    = IRU;
               break;
            case 3:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU - 2;
               TwoSLD = TwoSRU;
               ILD    = IRU;
               break;
         }
         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );

         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockQ     = QprevT->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

            dcomplex factor;
            dcomplex * ptr;
            dcomplex * ptrTmp;
            if ( geval < 2 ) {
               factor = 1.0;
               ptr    = BlockQ;

            } else {
               int fase = ( ( ( ( TwoSRU + 1 - TwoSLU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               factor   = fase * sqrt( ( TwoSLU + 1.0 ) / ( TwoSRD + 1.0 ) );

               int dimLUD = dimLU * dimLD;
               int inc    = 1;
               ptrTmp     = workmemLL;
               zcopy_( &dimLUD, BlockQ, &inc, ptrTmp, &inc );

               for ( int loca = 0; loca < index - 1; loca++ ) {
                  if ( bk_up->gIrrep( index - 1 ) == bk_up->gIrrep( loca ) ) {
                     dcomplex alpha =
                         Prob->gMxElement( loca, index - 1, index - 1, index - 1 );
                     dcomplex * BlockL = LprevT[ index - 2 - loca ]->gStorage(
                         NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
                     zaxpy_( &dimLUD, &alpha, BlockL, &inc, ptrTmp, &inc );
                  }
               }
               ptr = ptrTmp;
            }

            // factor * Tup^T * L --> mem2 //set
            dcomplex beta = 0.0;
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &factor,
                    BlockTup, &dimLU, ptr, &dimLU, &beta, workmemLR, &dimRU );

            // mem2 * Tdown --> mem //add
            factor = 1.0;
            beta   = 1.0;
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &factor,
                    workmemLR, &dimRU, BlockTdown, &dimLD, &beta, workmemRR, &dimRU );
         }
      }

      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemRR, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermQLLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorL ** Lprev,
                                       CTensorQ * Qprev, dcomplex * workmemLL, dcomplex * workmemLR, dcomplex * workmemRR ) {

   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   if ( dimLU > 0 && dimLD > 0 ) {
      int dimTot = dimLU * dimLD;
      for ( int cnt = 0; cnt < dimTot; cnt++ ) {
         workmemLL[ cnt ] = 0.0;
      }

      char cotrans = 'C';
      char notrans = 'N';

      for ( int geval = 0; geval < 4; geval++ ) {
         int NRU, TwoSRU, IRU, NRD, TwoSRD, IRD;
         switch ( geval ) {
            case 0:
               NRU    = NLU;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU + 1;
               TwoSRD = TwoSLU - 1;
               IRD    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               break;
            case 1:
               NRU    = NLU;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU + 1;
               TwoSRD = TwoSLU + 1;
               IRD    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               break;
            case 2:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU - 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU + 2;
               TwoSRD = TwoSLU;
               IRD    = ILU;
               break;
            case 3:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU + 2;
               TwoSRD = TwoSLU;
               IRD    = ILU;
               break;
         }
         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );

         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockQ     = Qprev->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

            dcomplex factor;
            dcomplex * ptr;
            dcomplex * ptrTmp;
            if ( geval < 2 ) {
               factor = ( TwoSRD + 1.0 ) / ( TwoSLU + 1.0 );
               ptr    = BlockQ;

            } else {
               int fase = ( ( ( ( TwoSLU + 1 - TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               factor   = fase * sqrt( ( TwoSRU + 1.0 ) / ( TwoSLU + 1.0 ) );

               int dimRUD = dimRU * dimRD;
               ptrTmp     = workmemRR;
               int inc    = 1;

               zcopy_( &dimRUD, BlockQ, &inc, ptrTmp, &inc );

               for ( int loca = index + 1; loca < Prob->gL(); loca++ ) {
                  if ( bk_up->gIrrep( index ) == bk_up->gIrrep( loca ) ) {
                     dcomplex alpha    = Prob->gMxElement( index, index, index, loca );
                     dcomplex * BlockL = Lprev[ loca - index - 1 ]->gStorage(
                         NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
                     zaxpy_( &dimRUD, &alpha, BlockL, &inc, ptrTmp, &inc );
                  }
               }
               ptr = ptrTmp;
            }

            // factor * Tup * L --> mem2 //set
            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &factor,
                    BlockTup, &dimLU, ptr, &dimRU, &beta, workmemLR, &dimLU );

            // mem2 * Tdown^T --> mem //add
            factor = 1.0;
            beta   = 1.0;
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &factor,
                    workmemLR, &dimLU, BlockTdown, &dimLD, &beta, workmemLL, &dimLU );
         }
      }

      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemLL, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermQTLTLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorLT ** Lprev,
                                         CTensorQT * Qprev, dcomplex * workmemLL, dcomplex * workmemLR, dcomplex * workmemRR ) {

   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   if ( dimLU > 0 && dimLD > 0 ) {
      int dimTot = dimLU * dimLD;
      for ( int cnt = 0; cnt < dimTot; cnt++ ) {
         workmemLL[ cnt ] = 0.0;
      }

      char cotrans = 'C';
      char notrans = 'N';

      for ( int geval = 0; geval < 4; geval++ ) {
         int NRU, TwoSRU, IRU, NRD, TwoSRD, IRD;
         switch ( geval ) {
            case 0:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU;
               TwoSRD = TwoSLU;
               IRD    = ILU;
               break;
            case 1:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU - 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU;
               TwoSRD = TwoSLU;
               IRD    = ILU;
               break;
            case 2:
               NRU    = NLU + 2;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU + 1;
               TwoSRD = TwoSLU + 1;
               IRD    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               break;
            case 3:
               NRU    = NLU + 2;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU + 1;
               TwoSRD = TwoSLU - 1;
               IRD    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               break;
         }
         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );

         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockQ     = Qprev->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

            dcomplex factor;
            dcomplex * ptr;
            dcomplex * ptrTmp;
            if ( geval < 2 ) {
               factor = ( TwoSRU + 1.0 ) / ( TwoSLD + 1.0 );
               ptr    = BlockQ;

            } else {
               int fase = ( ( ( ( TwoSLU + 1 - TwoSRD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               factor   = fase * sqrt( ( TwoSRD + 1.0 ) / ( TwoSLD + 1.0 ) );

               int dimRUD = dimRU * dimRD;
               ptrTmp     = workmemRR;
               int inc    = 1;
               zcopy_( &dimRUD, BlockQ, &inc, ptrTmp, &inc );

               for ( int loca = index + 1; loca < Prob->gL(); loca++ ) {
                  if ( bk_up->gIrrep( index ) == bk_up->gIrrep( loca ) ) {
                     dcomplex alpha    = Prob->gMxElement( index, index, index, loca );
                     dcomplex * BlockL = Lprev[ loca - index - 1 ]->gStorage(
                         NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
                     zaxpy_( &dimRUD, &alpha, BlockL, &inc, ptrTmp, &inc );
                  }
               }
               ptr = ptrTmp;
            }

            // factor * Tup * L --> mem2 //set
            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &factor,
                    BlockTup, &dimLU, ptr, &dimRU, &beta, workmemLR, &dimLU );

            // mem2 * Tdown^T --> mem //add
            factor = 1.0;
            beta   = 1.0;
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &factor,
                    workmemLR, &dimLU, BlockTdown, &dimLD, &beta, workmemLL, &dimLU );
         }
      }

      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemLL, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermARight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * Aprev,
                                       dcomplex * workmemRR, dcomplex * workmemLR ) {
   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int NLU    = NRU - 2;
   int TwoSLU = TwoSRU;
   int ILU    = IRU;

   int NLD    = NRU;
   int TwoSLD = TwoSRU;
   int ILD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );
   int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 ) {
      dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
      dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
      dcomplex * BlockA     = Aprev->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

      // factor * Tup^T * A --> mem2 //set
      dcomplex factor = sqrt( 2.0 );
      dcomplex beta   = 0.0; // set
      zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &factor, BlockTup, &dimLU, BlockA, &dimLU, &beta, workmemLR, &dimRU );

      // mem2 * Tdown --> mem //set
      factor = 1.0;
      zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &factor,
              workmemLR, &dimRU, BlockTdown, &dimLD, &beta, workmemRR, &dimRU );

      int dimTot     = dimRD * dimRU;
      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemRR, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermATRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * AprevT,
                                        dcomplex * workmemRR, dcomplex * workmemLR ) {
   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int NLU    = NRU;
   int TwoSLU = TwoSRU;
   int ILU    = IRU;

   int NLD    = NRU - 2;
   int TwoSLD = TwoSRU;
   int ILD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );
   int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 ) {
      dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
      dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
      dcomplex * BlockA     = AprevT->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

      // factor * Tup^T * A --> mem2 //set
      dcomplex factor = sqrt( 2.0 );
      dcomplex beta   = 0.0; // set
      zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &factor,
              BlockTup, &dimLU, BlockA, &dimLU, &beta, workmemLR, &dimRU );

      // mem2 * Tdown --> mem //set
      factor = 1.0;

      zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &factor,
              workmemLR, &dimRU, BlockTdown, &dimLD, &beta, workmemRR, &dimRU );

      int dimTot     = dimRD * dimRU;
      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemRR, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermALeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * Aprev,
                                      dcomplex * workmemLR, dcomplex * workmemLL ) {

   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   const int NRU    = NLU;
   const int TwoSRU = TwoSLU;
   const int IRU    = ILU;

   const int NRD    = NLU + 2;
   const int TwoSRD = TwoSLU;
   const int IRD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );
   int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 ) {
      dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
      dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
      dcomplex * BlockA     = Aprev->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

      // factor * Tup * A --> mem2 //set
      dcomplex factor = sqrt( 2.0 );
      dcomplex beta   = 0.0; // set
      zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &factor,
              BlockTup, &dimLU, BlockA, &dimRU, &beta, workmemLR, &dimLU );

      // mem2 * Tdown^T --> mem //set
      factor = 1.0;
      zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &factor,
              workmemLR, &dimLU, BlockTdown, &dimLD, &beta, workmemLL, &dimLU );

      int dimTot     = dimLU * dimLD;
      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemLL, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermATLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * Aprev,
                                       dcomplex * workmemLR, dcomplex * workmemLL ) {

   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   const int NRU    = NLU + 2;
   const int TwoSRU = TwoSLU;
   const int IRU    = ILU;

   const int NRD    = NLU;
   const int TwoSRD = TwoSLU;
   const int IRD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );
   int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 ) {
      dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
      dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
      dcomplex * BlockA     = Aprev->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

      // factor * Tup * A --> mem2 //set
      dcomplex factor = sqrt( 2.0 );
      dcomplex beta   = 0.0; // set
      zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &factor,
              BlockTup, &dimLU, BlockA, &dimRU, &beta, workmemLR, &dimLU );

      // mem2 * Tdown^T --> mem //set
      factor = 1.0;
      zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &factor,
              workmemLR, &dimLU, BlockTdown, &dimLD, &beta, workmemLL, &dimLU );

      int dimTot     = dimLU * dimLD;
      int inc        = 1;
      dcomplex alpha = 1.0;
      zaxpy_( &dimTot, &alpha, workmemLL, &inc, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CTensorX::addTermCRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * denCT,
                                       dcomplex * workmemLR ) {
   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 ) {
      for ( int geval = 0; geval < 3; geval++ ) {
         int NL, TwoSL, IL;
         switch ( geval ) {
            case 0:
               NL    = NRU - 1;
               TwoSL = TwoSRU - 1;
               IL    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               break;
            case 1:
               NL    = NRU - 1;
               TwoSL = TwoSRU + 1;
               IL    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               break;
            case 2:
               NL    = NRU - 2;
               TwoSL = TwoSRU;
               IL    = IRU;
               break;
         }
         int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
         int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );
            dcomplex * BlockC     = denCT->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

            dcomplex factor = ( geval < 2 ) ? sqrt( 0.5 ) : sqrt( 2.0 );
            dcomplex beta   = 0.0; // set

            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &factor,
                    BlockTup, &dimLU, BlockC, &dimLU, &beta, workmemLR, &dimRU );

            // mem2 * Tdown --> mem //set
            factor = 1.0;
            beta   = 1.0;
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &factor,
                    workmemLR, &dimRU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorX::addTermCLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * denC,
                                      dcomplex * workmemLR ) {

   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimLU > 0 && dimLD > 0 ) {
      for ( int geval = 0; geval < 3; geval++ ) {
         int NR, TwoSR, IR;
         switch ( geval ) {
            case 0:
               NR    = NLU + 1;
               TwoSR = TwoSLU - 1;
               IR    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               break;
            case 1:
               NR    = NLU + 1;
               TwoSR = TwoSLU + 1;
               IR    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               break;
            case 2:
               NR    = NLU + 2;
               TwoSR = TwoSLU;
               IR    = ILU;
               break;
         }
         int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            dcomplex * BlockTup =
                denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
            dcomplex * BlockTdown =
                denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );
            dcomplex * BlockC = denC->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

            dcomplex factor = ( geval < 2 )
                                  ? ( sqrt( 0.5 ) * ( TwoSR + 1.0 ) / ( TwoSLU + 1.0 ) )
                                  : sqrt( 2.0 );
            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &factor,
                    BlockTup, &dimLU, BlockC, &dimRU, &beta, workmemLR, &dimLU );

            factor = 1.0;
            beta   = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &factor,
                    workmemLR, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}

void CheMPS2::CTensorX::addTermDRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * denDT,
                                       dcomplex * workmemLR ) {

   int NRU    = sector_nelec_up[ ikappa ];
   int TwoSRU = sector_spin_up[ ikappa ];
   int IRU    = sector_irrep_up[ ikappa ];

   int NRD    = NRU;
   int TwoSRD = TwoSRU;
   int IRD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   const int IL = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
   const int NL = NRU - 1;

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 ) {
      for ( int geval = 0; geval < 4; geval++ ) {
         int TwoSLU, TwoSLD;
         switch ( geval ) {
            case 0:
               TwoSLU = TwoSRU - 1;
               TwoSLD = TwoSRU - 1;
               break;
            case 1:
               TwoSLU = TwoSRU + 1;
               TwoSLD = TwoSRU - 1;
               break;
            case 2:
               TwoSLU = TwoSRU - 1;
               TwoSLD = TwoSRU + 1;
               break;
            case 3:
               TwoSLU = TwoSRU + 1;
               TwoSLD = TwoSRU + 1;
               break;
         }

         int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSLU, IL );
         int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSLD, IL );

         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NL, TwoSLU, IL, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSLD, IL, NRD, TwoSRD, IRD );
            dcomplex * BlockD     = denDT->gStorage( NL, TwoSLU, IL, NL, TwoSLD, IL );

            int fase =
                ( ( ( ( TwoSLD + sector_spin_up[ ikappa ] + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factor = fase * sqrt( 3.0 * ( TwoSLU + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSRD );

            dcomplex beta = 0.0; // set
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &factor,
                    BlockTup, &dimLU, BlockD, &dimLU, &beta, workmemLR, &dimRU );

            factor = 1.0;
            beta   = 1.0; // add
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &factor,
                    workmemLR, &dimRU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorX::addTermDLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorOperator * denD,
                                      dcomplex * workmemLR ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   const int NR = NLU + 1;
   const int IR = Irreps::directProd( ILU, bk_up->gIrrep( index ) );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimLU > 0 && dimLD > 0 ) {
      for ( int geval = 0; geval < 4; geval++ ) {
         int TwoSRU, TwoSRD;
         switch ( geval ) {
            case 0:
               TwoSRU = TwoSLU - 1;
               TwoSRD = TwoSLU - 1;
               break;
            case 1:
               TwoSRU = TwoSLU + 1;
               TwoSRD = TwoSLU - 1;
               break;
            case 2:
               TwoSRU = TwoSLU - 1;
               TwoSRD = TwoSLU + 1;
               break;
            case 3:
               TwoSRU = TwoSLU + 1;
               TwoSRD = TwoSLU + 1;
               break;
         }

         int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSRU, IR );
         int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSRD, IR );

         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSRU, IR );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSRD, IR );
            dcomplex * BlockD     = denD->gStorage( NR, TwoSRU, IR, NR, TwoSRD, IR );

            int fase        = ( ( ( ( TwoSLU + TwoSRD + 3 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factor = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) * ( ( TwoSRD + 1.0 ) / ( TwoSLU + 1.0 ) ) *
                              Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSLU );
            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &factor,
                    BlockTup, &dimLU, BlockD, &dimRU, &beta, workmemLR, &dimLU );

            factor = 1.0;
            beta   = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &factor,
                    workmemLR, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}
