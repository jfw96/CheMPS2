
#include <math.h>
#include <stdlib.h>

#include "CTensorS1.h"
#include "Lapack.h"
#include "Wigner.h"

CheMPS2::CTensorS1::CTensorS1( const int boundary_index, const int Idiff,
                               const bool moving_right,
                               const SyBookkeeper * book_up,
                               const SyBookkeeper * book_down )
    : CTensorOperator( boundary_index, // index
                       2,              // two_j
                       2,              // n_elec
                       Idiff,          // irrep
                       moving_right,   // direction
                       true,           // prime_last
                       false,          // jw_phase (two 2nd quantized operators)
                       book_up,        // upper bookkeeper
                       book_down       // lower bookkeeper
                       ) {}

CheMPS2::CTensorS1::~CTensorS1() {}

void CheMPS2::CTensorS1::makenew( CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
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

void CheMPS2::CTensorS1::makenewRight( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];

   const int NRD    = NRU + 2;
   const int TwoSRD = sector_spin_down[ ikappa ];
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
               NLU    = NRU;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU + 1;
               TwoSLD = TwoSRD - 1;
               ILD    = Irreps::directProd( IRU, denL->get_irrep() );
               break;
            case 1:
               NLU    = NRU;
               TwoSLU = TwoSRU;
               ILU    = IRU;
               NLD    = NRU + 1;
               TwoSLD = TwoSRD + 1;
               ILD    = Irreps::directProd( IRU, denL->get_irrep() );
               break;
            case 2:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU;
               TwoSLD = TwoSRD;
               ILD    = IRD;
               break;
            case 3:
               NLU    = NRU - 1;
               TwoSLU = TwoSRU + 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               NLD    = NRU;
               TwoSLD = TwoSRD;
               ILD    = IRD;
               break;
         }

         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) && ( abs( TwoSLU - TwoSLD ) < 2 ) ) {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockL     = denL->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

            // factor * Tup^T * L -> mem
            dcomplex alpha = 1.0;
            if ( geval <= 1 ) {
               int fase = ( ( ( ( TwoSRU + TwoSRD + 2 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               alpha    = fase * sqrt( 3.0 * ( TwoSLD + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSLD );
            } else {
               int fase = ( ( ( ( TwoSLU + TwoSRD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               alpha    = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSRD, TwoSLU );
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

void CheMPS2::CTensorS1::makenewLeft( const int ikappa, CTensorL * denL,
                                      CTensorT * denTup,
                                      CTensorT * denTdown, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];

   const int NLD    = NLU + 2;
   const int TwoSLD = sector_spin_down[ ikappa ];
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
               NRU    = NLU + 1;
               TwoSRU = TwoSLU - 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU + 2;
               TwoSRD = TwoSLD;
               IRD    = ILD;
               break;
            case 1:
               NRU    = NLU + 1;
               TwoSRU = TwoSLU + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               NRD    = NLU + 2;
               TwoSRD = TwoSLD;
               IRD    = ILD;
               break;
            case 2:
               NRU    = NLU + 2;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU + 3;
               TwoSRD = TwoSLD - 1;
               IRD    = Irreps::directProd( ILU, denL->get_irrep() );
               break;
            case 3:
               NRU    = NLU + 2;
               TwoSRU = TwoSLU;
               IRU    = ILU;
               NRD    = NLU + 3;
               TwoSRD = TwoSLD + 1;
               IRD    = Irreps::directProd( ILU, denL->get_irrep() );
               break;
         }

         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) && ( abs( TwoSRD - TwoSRU ) < 2 ) ) {
            dcomplex * BlockTup =
                denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown =
                denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
            dcomplex * BlockL =
                denL->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

            // factor * Tup * L -> mem
            dcomplex alpha = 1.0;
            if ( geval <= 1 ) {
               int fase = ( ( ( ( TwoSLU + TwoSLD + 2 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               alpha    = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) *
                       Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSRU );
            } else {
               int fase = ( ( ( ( TwoSLU + TwoSRD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               alpha    = fase * sqrt( 3.0 / ( TwoSLD + 1.0 ) ) * ( TwoSRD + 1 ) *
                       Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSLD, TwoSRD );
            }

            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, BlockL, &dimRU, &beta, workmem, &dimLU );

            // mem * Tdown -> storage
            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem, &dimLU, BlockTdown, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}
