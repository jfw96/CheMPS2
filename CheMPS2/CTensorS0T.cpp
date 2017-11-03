
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CTensorS0T.h"
#include "Lapack.h"

CheMPS2::CTensorS0T::CTensorS0T( const int boundary_index, const int Idiff,
                                 const bool moving_right,
                                 const CheMPS2::SyBookkeeper * book_up,
                                 const CheMPS2::SyBookkeeper * book_down )
    : CTensorOperator( boundary_index, // index
                       0,              // two_j
                       -2,             // n_elec
                       Idiff,          // irrep
                       moving_right,   // direction
                       false,          // prime_last (doesn't matter for spin-0)
                       false,          // jw_phase (two 2nd quantized operators)
                       book_up,        // upper bookeeper
                       book_down       // lower bookeeper
                       ) {}

CheMPS2::CTensorS0T::~CTensorS0T() {}

void CheMPS2::CTensorS0T::makenew( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   clear();

   if ( moving_right ) {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         makenewRight( ikappa, denTup, denTdown, previous, workmem );
      }
   } else {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         makenewLeft( ikappa, denTup, denTdown, previous, workmem );
      }
   }
}

void CheMPS2::CTensorS0T::makenew( CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   clear();

   if ( moving_right ) {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         makenewRight( ikappa, denL, denTup, denTdown, workmem );
      }
   } else {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         makenewLeft( ikappa, denL, denTup, denTdown, workmem );
      }
   }
}

void CheMPS2::CTensorS0T::makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU - 2;
   const int TwoSRD = TwoSRU;
   const int IRD    = IRU;

   const int NL    = NRU - 2;
   const int TwoSL = TwoSRU;
   const int IL    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );
   int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
   int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 ) {
      if ( previous == NULL ) {
         assert( dimLU == dimLD );

         dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );

         dcomplex alpha = sqrt( 2.0 );
         dcomplex beta  = 1.0; // add
         zgemm_( &cotrans, &notrans, &dimRU, &dimRD, &dimLU, &alpha, BlockTup, &dimLU, BlockTdown, &dimLU, &beta, storage + kappa2index[ ikappa ], &dimRU );

      } else {
         dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );
         dcomplex * BlockPrev  = previous->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

         dcomplex alpha = sqrt( 2.0 );

         dcomplex set = 0.0;
         zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, BlockPrev, &dimLU, &set, workmem, &dimRU );

         dcomplex one = 1.0;
         zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &one, workmem, &dimRU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimRU );
      }
   }
}

void CheMPS2::CTensorS0T::makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU - 2;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   const int NR    = NLU;
   const int TwoSR = TwoSLU;
   const int IR    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );
   int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );
   int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 && dimLU > 0 && dimLD > 0 ) {
      if ( previous == NULL ) {
         assert( dimRU == dimRD );

         dcomplex * BlockTup =
             denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
         dcomplex * BlockTdown =
             denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );

         dcomplex alpha = sqrt( 2.0 );
         dcomplex beta  = 1.0; // add
         zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRU, &alpha, BlockTup, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );

      } else {
         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );
         dcomplex * BlockPrev  = previous->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

         dcomplex alpha = sqrt( 2.0 );

         dcomplex set = 0.0;
         zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, BlockPrev, &dimRU, &set, workmem, &dimLU );

         dcomplex one = 1.0;
         zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &one, workmem, &dimLU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimLU );
      }
   }
}

void CheMPS2::CTensorS0T::makenewRight( const int ikappa, CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU - 2;
   const int TwoSRD = TwoSRU;
   const int IRD    = Irreps::directProd( n_irrep, IRU );

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimRU > 0 && dimRD > 0 ) {
      for ( int geval = 0; geval < 4; geval++ ) {
         int NLU, TwoSLU, ILU, NLD, TwoSLD, ILD;
         switch ( geval ) {
            case 0:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU - 2;
               TwoSLD = TwoSRU;
               ILD    = IRD;
               break;
            case 1:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU + 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU - 2;
               TwoSLD = TwoSRU;
               ILD    = IRD;
               break;
            case 2:
               NLU    = NRU - 2;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU - 3;
               TwoSLD = TwoSRU + 1;
               ILD    = Irreps::directProd( IRU, denL->get_irrep() );
               break;
            case 3:
               NLU    = NRU - 2;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU - 3;
               TwoSLD = TwoSRU - 1;
               ILD    = Irreps::directProd( IRU, denL->get_irrep() );
               break;
         }

         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockL     = denL->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

            // factor * Tup^T * L -> mem
            dcomplex alpha;
            if ( geval <= 1 ) {
               int fase = ( ( ( ( TwoSRD - TwoSLU + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               alpha    = fase * sqrt( 0.5 * ( TwoSLU + 1.0 ) / ( TwoSRD + 1.0 ) );
            } else {
               alpha = -sqrt( 0.5 );
            }

            dcomplex beta = 0.0; // set
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, BlockL, &dimLU, &beta, workmem, &dimRU );

            // mem * Tdown -> storage
            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem, &dimRU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorS0T::makenewLeft( const int ikappa, CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU - 2;
   const int TwoSLD = TwoSLU;
   const int ILD    = Irreps::directProd( ILU, n_irrep );

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   char cotrans = 'C';
   char notrans = 'N';

   if ( dimLU > 0 && dimLD > 0 ) {
      for ( int geval = 0; geval < 4; geval++ ) {
         int NRU, TwoSRU, IRU, NRD, TwoSRD, IRD;
         switch ( geval ) {
            case 0:
               NRU    = NLU;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU - 1;
               TwoSRD = TwoSLU - 1;
               IRD    = Irreps::directProd( ILU, denL->get_irrep() );
               break;
            case 1:
               NRU    = NLU;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU - 1;
               TwoSRD = TwoSLU + 1;
               IRD    = Irreps::directProd( ILU, denL->get_irrep() );
               break;
            case 2:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU;
               TwoSRD = TwoSLU;
               IRD    = ILD;
               break;
            case 3:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU - 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU;
               TwoSRD = TwoSLU;
               IRD    = ILD;
               break;
         }

         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            dcomplex * BlockTup =
                denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown =
                denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockL =
                denL->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

            // factor * Tup * L -> mem
            dcomplex alpha = 1.0;
            if ( geval <= 1 ) {
               int fase = ( ( ( ( TwoSLU - TwoSRD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               alpha    = fase * sqrt( 0.5 * ( TwoSRD + 1.0 ) / ( TwoSLU + 1.0 ) );
            } else {
               alpha = -sqrt( 0.5 ) * ( TwoSRU + 1.0 ) / ( TwoSLU + 1.0 );
            }
            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, BlockL, &dimRU, &beta, workmem, &dimLU );

            // mem * Tdown^T -> storage
            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}
