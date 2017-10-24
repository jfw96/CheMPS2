
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CTensorS0.h"
#include "Lapack.h"

CheMPS2::CTensorS0::CTensorS0( const int boundary_index, const int Idiff,
                               const bool moving_right,
                               const CheMPS2::SyBookkeeper * book_up,
                               const CheMPS2::SyBookkeeper * book_down )
    : CTensorOperator( boundary_index, // index
                       0,              // two_j
                       2,              // n_elec
                       Idiff,          // irrep
                       moving_right,   // direction
                       true,           // prime_last (doesn't matter for spin-0)
                       false,          // jw_phase (two 2nd quantized operators)
                       book_up,        // upper bookkeeper
                       book_down       // lower bookkeeper
                       ) {}

CheMPS2::CTensorS0::~CTensorS0() {}

void CheMPS2::CTensorS0::makenew( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
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

void CheMPS2::CTensorS0::makenew( CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
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

void CheMPS2::CTensorS0::makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU + 2;
   const int TwoSRD = TwoSRU;
   const int IRD    = IRU;

   const int NL    = NRU;
   const int TwoSL = TwoSRU;
   const int IL    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
   int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );

   char cotrans = 'C';
   char notrans = 'N';

   if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
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
         dcomplex set   = 0.0;
         zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, BlockPrev, &dimLU, &set, workmem, &dimRU );

         dcomplex one = 1.0;
         zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &one, workmem, &dimRU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimRU );
      }
   }
}

void CheMPS2::CTensorS0::makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU + 2;
   const int TwoSLD = TwoSLU;
   const int ILD    = ILU;

   const int NR    = NLU + 2;
   const int TwoSR = TwoSLU;
   const int IR    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );
   int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );

   char cotrans = 'C';
   char notrans = 'N';

   if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
      if ( previous == NULL ) {
         assert( dimRU == dimRD );

         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );

         dcomplex alpha = sqrt( 2.0 );
         dcomplex beta  = 1.0; // add
         zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRU, &alpha, BlockTup, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );

      } else {
         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );
         dcomplex * BlockPrev  = previous->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

         dcomplex alpha = sqrt( 2.0 );
         dcomplex beta  = 1.0; // add

         dcomplex set = 0.0;
         zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, BlockPrev, &dimRU, &set, workmem, &dimLU );

         dcomplex one = 1.0;
         zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &one, workmem, &dimLU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimLU );
      }
   }
}

void CheMPS2::CTensorS0::makenewRight( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU + 2;
   const int TwoSRD = TwoSRU;
   const int IRD    = Irreps::directProd( n_irrep, IRU );

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   char cotrans = 'C';
   char notrans = 'N';

   for ( int geval = 0; geval < 4; geval++ ) {
      int NLU, TwoSLU, ILU, NLD, TwoSLD, ILD;
      switch ( geval ) {
         case 0:
            NLU    = NRU;
            TwoSLU = TwoSRU;
            ILU    = IRU;
            NLD    = NRU + 1;
            TwoSLD = TwoSRU - 1;
            ILD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
         case 1:
            NLU    = NRU;
            TwoSLU = TwoSRU;
            ILU    = IRU;
            NLD    = NRU + 1;
            TwoSLD = TwoSRU + 1;
            ILD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
         case 2:
            NLU    = NRU - 1;
            TwoSLU = TwoSRU - 1;
            ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
            NLD    = NRU;
            TwoSLD = TwoSRU;
            ILD    = IRD;
            break;
         case 3:
            NLU    = NRU - 1;
            TwoSLU = TwoSRU + 1;
            ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
            NLD    = NRU;
            TwoSLD = TwoSRU;
            ILD    = IRD;
            break;
      }

      int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
      int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
      if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NLU + 1, TwoSLD, ILD, NRD, TwoSRD, IRD );
         dcomplex * BlockL     = denL->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

         // factor * Tup^T * L -> mem
         dcomplex alpha;
         if ( geval <= 1 ) {
            int fase = ( ( ( ( TwoSRU - TwoSLD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            alpha    = fase * sqrt( 0.5 * ( TwoSLD + 1.0 ) / ( TwoSRU + 1.0 ) );
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

void CheMPS2::CTensorS0::makenewLeft( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU + 2;
   const int TwoSLD = TwoSLU;
   const int ILD    = Irreps::directProd( n_irrep, ILU );

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   char cotrans = 'C';
   char notrans = 'N';

   for ( int geval = 0; geval < 4; geval++ ) {
      int NRU, TwoSRU, IRU, NRD, TwoSRD, IRD;
      switch ( geval ) {
         case 0:
            NRU    = NLU + 1;
            TwoSRU = TwoSLU - 1;
            IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
            NRD    = NLU + 2;
            TwoSRD = TwoSLU;
            IRD    = ILD;
            break;
         case 1:
            NRU    = NLU + 1;
            TwoSRU = TwoSLU + 1;
            IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
            NRD    = NLU + 2;
            TwoSRD = TwoSLU;
            IRD    = ILD;
            break;
         case 2:
            NRU    = NLU + 2;
            TwoSRU = TwoSLU;
            IRU    = ILU;
            NRD    = NLU + 3;
            TwoSRD = TwoSLU - 1;
            IRD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
         case 3:
            NRU    = NLU + 2;
            TwoSRU = TwoSLU;
            IRU    = ILU;
            NRD    = NLU + 3;
            TwoSRD = TwoSLU + 1;
            IRD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
      }
      int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
      int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
      if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
         dcomplex * BlockL     = denL->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

         // factor * Tup * L -> mem
         dcomplex alpha = 1.0;
         if ( geval <= 1 ) {
            int fase = ( ( ( ( TwoSLU - TwoSRU + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            alpha    = fase * sqrt( 0.5 * ( TwoSRU + 1.0 ) / ( TwoSLU + 1.0 ) );
         } else {
            alpha = -sqrt( 0.5 ) * ( TwoSRD + 1.0 ) / ( TwoSLU + 1.0 );
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
