
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "CTensorF1.h"
#include "Lapack.h"
#include "Wigner.h"

CheMPS2::CTensorF1::CTensorF1( const int boundary_index, const int Idiff,
                               const bool moving_right,
                               const CheMPS2::SyBookkeeper * book_up,
                               const CheMPS2::SyBookkeeper * book_down )
    : CTensorOperator( boundary_index, // index
                       2,              // two_j
                       0,              // n_elec
                       Idiff,          // irrep
                       moving_right,   // direction
                       moving_right,   // prime_last
                       false,          // jw_phase (two 2nd quantized operators)
                       book_up,        // upper bookkeeper
                       book_down       // lower bookeeper
                       ) {}

CheMPS2::CTensorF1::~CTensorF1() {}

void CheMPS2::CTensorF1::makenew( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
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

void CheMPS2::CTensorF1::makenew( CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
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

void CheMPS2::CTensorF1::makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU;
   const int TwoSRD = sector_spin_down[ ikappa ];
   const int IRD    = IRU;

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   char cotrans = 'C';
   char notrans = 'N';

   for ( int geval = 0; geval < 2; geval++ ) {
      int NL, TwoSL, IL;
      switch ( geval ) {
         case 0:
            NL    = NRU - 1;
            TwoSL = TwoSRU + 1;
            IL    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
            break;
         case 1:
            NL    = NRU - 1;
            TwoSL = TwoSRU - 1;
            IL    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
            break;
      }

      if ( ( TwoSL >= 0 ) && ( abs( TwoSRD - TwoSL ) < 2 ) ) {
         int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
         int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            if ( previous == NULL ) {
               assert( dimLU == dimLD );

               dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
               dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );

               int fase       = ( ( ( ( TwoSL + TwoSRD + 3 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               dcomplex alpha = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSL );
               dcomplex beta  = 1.0; // add
               zgemm_( &cotrans, &notrans, &dimRU, &dimRD, &dimLU, &alpha, BlockTup, &dimLU, BlockTdown, &dimLU, &beta, storage + kappa2index[ ikappa ], &dimRU );

            } else {
               dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
               dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );
               dcomplex * BlockPrev  = previous->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

               int fase       = ( ( ( ( TwoSL + TwoSRD + 3 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               dcomplex alpha = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSL );

               dcomplex set = 0.0;
               zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, BlockPrev, &dimLU, &set, workmem, &dimRU );
               dcomplex one = 1.0;
               zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &one, workmem, &dimRU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimRU );
            }
         }
      }
   }
}

void CheMPS2::CTensorF1::makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = sector_spin_down[ ikappa ];
   const int ILD    = ILU;

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   const int IR = Irreps::directProd( ILU, bk_up->gIrrep( index ) );

   char cotrans = 'C';
   char notrans = 'N';

   for ( int geval = 0; geval < 2; geval++ ) {
      int NR, TwoSR, IR;
      switch ( geval ) {
         case 0:
            NR    = NLU + 1;
            TwoSR = TwoSLU + 1;
            IR    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
            break;
         case 1:
            NR    = NLU + 1;
            TwoSR = TwoSLU - 1;
            IR    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
            break;
      }

      if ( ( TwoSR >= 0 ) && ( abs( TwoSLD - TwoSR ) < 2 ) ) {
         int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            if ( previous == NULL ) {
               assert( dimRU == dimRD );

               dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
               dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );

               int fase       = ( ( ( ( TwoSLD + TwoSR + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               dcomplex alpha = fase * sqrt( 3.0 / ( TwoSLU + 1.0 ) ) * ( TwoSR + 1 ) * Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSR );

               dcomplex beta = 1.0; // add
               zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRU, &alpha, BlockTup, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );

            } else {
               dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
               dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );
               dcomplex * BlockPrev  = previous->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

               int fase       = ( ( ( ( TwoSLD + TwoSR + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               dcomplex alpha = fase * sqrt( 3.0 / ( TwoSLU + 1.0 ) ) * ( TwoSR + 1 ) * Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSR );

               dcomplex set = 0.0;
               zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, BlockPrev, &dimRU, &set, workmem, &dimLU );
               dcomplex one = 1.0;
               zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &one, workmem, &dimLU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimLU );
            }
         }
      }
   }
}

void CheMPS2::CTensorF1::makenewRight( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU;
   const int TwoSRD = sector_spin_down[ ikappa ];
   const int IRD    = Irreps::directProd( n_irrep, IRU );

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

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
            TwoSLD = TwoSRD;
            ILD    = IRD;
            break;
         case 1:
            NLU    = NRU - 1;
            TwoSLU = TwoSRU + 1;
            ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
            NLD    = NRU;
            TwoSLD = TwoSRD;
            ILD    = IRD;
            break;
         case 2:
            NLU    = NRU - 2;
            TwoSLU = TwoSRU;
            ILU    = IRU;
            NLD    = NRU - 1;
            TwoSLD = TwoSRD + 1;
            ILD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
         case 3:
            NLU    = NRU - 2;
            TwoSLU = TwoSRU;
            ILU    = IRU;
            NLD    = NRU - 1;
            TwoSLD = TwoSRD - 1;
            ILD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
      }

      int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
      int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
      if ( ( dimLU > 0 ) && ( dimLD > 0 ) && ( abs( TwoSLU - TwoSLD ) < 2 ) ) {
         dcomplex * BlockTup =
             denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown =
             denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
         dcomplex * BlockL =
             denL->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

         // factor * Tup^T * L -> mem
         dcomplex alpha;
         if ( geval <= 1 ) {
            int fase = ( ( ( ( TwoSLU + TwoSRD + 3 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            alpha    = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSLU );
         } else {
            int fase = ( ( ( ( TwoSRU + TwoSRD + 2 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            alpha    = fase * sqrt( 3.0 * ( TwoSLD + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSLD );
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

void CheMPS2::CTensorF1::makenewLeft( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU;
   const int TwoSLD = sector_spin_down[ ikappa ];
   const int ILD    = Irreps::directProd( ILU, n_irrep );

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

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
            TwoSRD = TwoSLD + 1;
            IRD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
         case 1:
            NRU    = NLU;
            TwoSRU = TwoSLU;
            IRU    = ILU;
            NRD    = NLU + 1;
            TwoSRD = TwoSLD - 1;
            IRD    = Irreps::directProd( ILU, denL->get_irrep() );
            break;
         case 2:
            NRU    = NLU + 1;
            TwoSRU = TwoSLU + 1;
            IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
            NRD    = NLU + 2;
            TwoSRD = TwoSLD;
            IRD    = ILD;
            break;
         case 3:
            NRU    = NLU + 1;
            TwoSRU = TwoSLU - 1;
            IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
            NRD    = NLU + 2;
            TwoSRD = TwoSLD;
            IRD    = ILD;
            break;
      }

      int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
      int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
      if ( ( dimRU > 0 ) && ( dimRD > 0 ) && ( abs( TwoSRU - TwoSRD ) < 2 ) ) {
         dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
         dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
         dcomplex * BlockL     = denL->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

         // factor * Tup * L -> mem
         dcomplex alpha;
         if ( geval <= 1 ) {
            int fase = ( ( ( ( TwoSLD + TwoSRD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            alpha    = fase * sqrt( 3.0 / ( TwoSLU + 1.0 ) ) * ( TwoSRD + 1 ) *
                    Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSRD );
         } else {
            int fase = ( ( ( TwoSLU ) % 2 ) != 0 ) ? -1 : 1;
            alpha    = fase *
                    sqrt( 3.0 * ( TwoSRU + 1.0 ) * ( TwoSLD + 1.0 ) / ( TwoSLU + 1.0 ) ) *
                    Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSRU );
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
