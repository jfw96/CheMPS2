
#include <algorithm>
#include <math.h>
#include <stdlib.h>

#include "CTensorO.h"
#include "Lapack.h"

CheMPS2::CTensorO::CTensorO( const int boundary_index, const bool moving_right,
                             const CheMPS2::SyBookkeeper * book_up,
                             const CheMPS2::SyBookkeeper * book_down )
    : CTensorOperator( boundary_index, // the index
                       0,              // two_j
                       0,              // n_elec
                       0,              // n_irrep
                       moving_right,   // the direction
                       true,           // prime_last (doesn't matter for spin-0 tensors)
                       false,          // jw_phase (no operators)
                       book_up,        // upper bookkeeper
                       book_down       // lower bookkeeper
                       ) {}

CheMPS2::CTensorO::~CTensorO() {}

void CheMPS2::CTensorO::update_ownmem( CTensorT * mps_tensor_up, CTensorT * mps_tensor_down, CTensorO * previous ) {
   clear();

   if ( moving_right ) {
      const int dimL = std::max( bk_up->gMaxDimAtBound( index - 1 ), bk_down->gMaxDimAtBound( index - 1 ) );
      const int dimR = std::max( bk_up->gMaxDimAtBound( index ), bk_down->gMaxDimAtBound( index ) );

#pragma omp parallel
      {
         dcomplex * workmem = new dcomplex[ dimL * dimR ];

#pragma omp for schedule( dynamic )
         for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
            update_moving_right( ikappa, previous, mps_tensor_up, mps_tensor_down, workmem );
         }

         delete[] workmem;
      }
   } else {
      const int dimL = std::max( bk_up->gMaxDimAtBound( index ), bk_down->gMaxDimAtBound( index ) );
      const int dimR = std::max( bk_up->gMaxDimAtBound( index + 1 ), bk_down->gMaxDimAtBound( index + 1 ) );

#pragma omp parallel
      {
         dcomplex * workmem = new dcomplex[ dimL * dimR ];

#pragma omp for schedule( dynamic )
         for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
            update_moving_left( ikappa, previous, mps_tensor_up, mps_tensor_down, workmem );
         }

         delete[] workmem;
      }
   }
}

void CheMPS2::CTensorO::create( CTensorT * mps_tensor_up, CTensorT * mps_tensor_down ) {
   clear();

   if ( moving_right ) {
#pragma omp parallel for schedule( dynamic )
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         create_right( ikappa, mps_tensor_up, mps_tensor_down );
      }
   } else {
#pragma omp parallel for schedule( dynamic )
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         create_left( ikappa, mps_tensor_up, mps_tensor_down );
      }
   }
}

void CheMPS2::CTensorO::create_right( const int ikappa, CTensorT * mps_tensor_up, CTensorT * mps_tensor_down ) {
   const int NR    = sector_nelec_up[ ikappa ];
   const int IR    = sector_irrep_up[ ikappa ];
   const int TwoSR = sector_spin_up[ ikappa ];

   int dimRU = bk_up->gCurrentDim( index, NR, TwoSR, IR );
   int dimRD = bk_down->gCurrentDim( index, NR, TwoSR, IR );

   if ( dimRU > 0 && dimRD > 0 ) {

      for ( int geval = 0; geval < 4; geval++ ) {
         int IL, TwoSL, NL;
         switch ( geval ) {
            case 0:
               NL    = NR;
               TwoSL = TwoSR;
               IL    = IR;
               break;
            case 1:
               NL    = NR - 2;
               TwoSL = TwoSR;
               IL    = IR;
               break;
            case 2:
               NL    = NR - 1;
               TwoSL = TwoSR - 1;
               IL    = Irreps::directProd( IR, bk_up->gIrrep( index - 1 ) );
               break;
            case 3:
               NL    = NR - 1;
               TwoSL = TwoSR + 1;
               IL    = Irreps::directProd( IR, bk_up->gIrrep( index - 1 ) );
               break;
         }

         int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
         int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) && ( dimLU == dimLD ) ) {
            dcomplex alpha = 1.0;
            dcomplex beta  = 1.0; // add
            dcomplex * Tup = mps_tensor_up->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
            dcomplex * Tdo = mps_tensor_down->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

            char cotrans = 'C';
            char notrans = 'N';
            zgemm_( &cotrans, &notrans, &dimRU, &dimRD, &dimLU, &alpha, Tup, &dimLU,
                    Tdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorO::create_left( const int ikappa, CTensorT * mps_tensor_up, CTensorT * mps_tensor_down ) {
   const int NL    = sector_nelec_up[ ikappa ];
   const int IL    = sector_irrep_up[ ikappa ];
   const int TwoSL = sector_spin_up[ ikappa ];

   int dimLU = bk_up->gCurrentDim( index, NL, TwoSL, IL );
   int dimLD = bk_down->gCurrentDim( index, NL, TwoSL, IL );

   if ( dimLU > 0 && dimLD > 0 ) {
      for ( int geval = 0; geval < 4; geval++ ) {
         int IR, TwoSR, NR;
         switch ( geval ) {
            case 0:
               NR    = NL;
               TwoSR = TwoSL;
               IR    = IL;
               break;
            case 1:
               NR    = NL + 2;
               TwoSR = TwoSL;
               IR    = IL;
               break;
            case 2:
               NR    = NL + 1;
               TwoSR = TwoSL - 1;
               IR    = Irreps::directProd( IL, bk_up->gIrrep( index ) );
               break;
            case 3:
               NR    = NL + 1;
               TwoSR = TwoSL + 1;
               IR    = Irreps::directProd( IL, bk_up->gIrrep( index ) );
               break;
         }

         int dimRU = bk_up->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimRD = bk_down->gCurrentDim( index + 1, NR, TwoSR, IR );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) && ( dimRU == dimRD ) ) {
            dcomplex * Tup = mps_tensor_up->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
            dcomplex * Tdo = mps_tensor_down->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
            char cotrans   = 'C';
            char notrans   = 'N';
            dcomplex alpha = ( ( geval > 1 ) ? ( ( TwoSR + 1.0 ) / ( TwoSL + 1 ) ) : 1.0 );
            dcomplex beta  = 1.0; // add

            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRU, &alpha, Tup, &dimLU,
                    Tdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}
