
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "CTensorOperator.h"
#include "Lapack.h"
#include "Logger.h"
#include "Special.h"
#include "Wigner.h"
#include "iostream"

CheMPS2::CTensorOperator::CTensorOperator(
    const int boundary_index, const int two_j, const int n_elec,
    const int n_irrep, const bool moving_right, const bool prime_last,
    const bool jw_phase, const CheMPS2::SyBookkeeper * bk_up,
    const CheMPS2::SyBookkeeper * bk_down )
    : CTensor() {
   // Copy the variables
   this->index        = boundary_index;
   this->two_j        = two_j;
   this->n_elec       = n_elec;
   this->n_irrep      = n_irrep;
   this->moving_right = moving_right;
   this->prime_last   = prime_last;
   this->jw_phase     = jw_phase;
   this->bk_up        = bk_up;
   this->bk_down      = bk_down;

   assert( two_j >= 0 );
   assert( n_irrep >= 0 );
   assert( n_irrep < bk_up->getNumberOfIrreps() );

   nKappa = 0;
   for ( int n_up = bk_up->gNmin( index ); n_up <= bk_up->gNmax( index ); n_up++ ) {
      for ( int TwoSU = bk_up->gTwoSmin( index, n_up );
            TwoSU <= bk_up->gTwoSmax( index, n_up ); TwoSU += 2 ) {
         for ( int IRU = 0; IRU < bk_up->getNumberOfIrreps(); IRU++ ) {
            const int dimU = bk_up->gCurrentDim( index, n_up, TwoSU, IRU );
            if ( dimU > 0 ) {
               const int IRD    = Irreps::directProd( n_irrep, IRU );
               const int n_down = n_up + n_elec;
               for ( int TwoSD = TwoSU - two_j; TwoSD <= TwoSU + two_j; TwoSD += 2 ) {
                  if ( TwoSD >= 0 ) {
                     const int dimD = bk_down->gCurrentDim( index, n_down, TwoSD, IRD );
                     if ( dimD > 0 ) {
                        nKappa++;
                     }
                  }
               }
            }
         }
      }
   }

   sector_nelec_up  = new int[ nKappa ];
   sector_irrep_up  = new int[ nKappa ];
   sector_spin_up   = new int[ nKappa ];
   sector_spin_down = ( ( two_j == 0 ) ? sector_spin_up : new int[ nKappa ] );
   kappa2index      = new int[ nKappa + 1 ];
   kappa2index[ 0 ] = 0;

   nKappa = 0;
   for ( int n_up = bk_up->gNmin( index ); n_up <= bk_up->gNmax( index ); n_up++ ) {
      for ( int TwoSU = bk_up->gTwoSmin( index, n_up );
            TwoSU <= bk_up->gTwoSmax( index, n_up ); TwoSU += 2 ) {
         for ( int IRU = 0; IRU < bk_up->getNumberOfIrreps(); IRU++ ) {
            const int dimU = bk_up->gCurrentDim( index, n_up, TwoSU, IRU );
            if ( dimU > 0 ) {
               const int IRD    = Irreps::directProd( n_irrep, IRU );
               const int n_down = n_up + n_elec;
               for ( int TwoSD = TwoSU - two_j; TwoSD <= TwoSU + two_j; TwoSD += 2 ) {
                  if ( TwoSD >= 0 ) {
                     const int dimD = bk_down->gCurrentDim( index, n_down, TwoSD, IRD );
                     if ( dimD > 0 ) {
                        sector_nelec_up[ nKappa ]  = n_up;
                        sector_irrep_up[ nKappa ]  = IRU;
                        sector_spin_up[ nKappa ]   = TwoSU;
                        sector_spin_down[ nKappa ] = TwoSD;
                        kappa2index[ nKappa + 1 ]  = kappa2index[ nKappa ] + dimU * dimD;
                        nKappa++;
                     }
                  }
               }
            }
         }
      }
   }

   storage = new dcomplex[ kappa2index[ nKappa ] ];
}

CheMPS2::CTensorOperator::~CTensorOperator() {
   delete[] sector_nelec_up;
   delete[] sector_irrep_up;
   delete[] sector_spin_up;
   delete[] kappa2index;
   delete[] storage;
   if ( two_j != 0 ) {
      delete[] sector_spin_down;
   }
}

int CheMPS2::CTensorOperator::gNKappa() const { return nKappa; }

dcomplex * CheMPS2::CTensorOperator::gStorage() { return storage; }

int CheMPS2::CTensorOperator::gKappa( const int N1, const int TwoS1, const int I1,
                                      const int N2, const int TwoS2, const int I2 ) const {
   if ( Irreps::directProd( I1, n_irrep ) != I2 ) {
      return -1;
   }
   if ( N2 != N1 + n_elec ) {
      return -1;
   }
   if ( abs( TwoS1 - TwoS2 ) > two_j ) {
      return -1;
   }

   if ( two_j == 0 ) {
      for ( int cnt = 0; cnt < nKappa; cnt++ ) {
         if ( ( sector_nelec_up[ cnt ] == N1 ) && ( sector_spin_up[ cnt ] == TwoS1 ) &&
              ( sector_irrep_up[ cnt ] == I1 ) ) {
            return cnt;
         }
      }
   } else {
      for ( int cnt = 0; cnt < nKappa; cnt++ ) {
         if ( ( sector_nelec_up[ cnt ] == N1 ) && ( sector_spin_up[ cnt ] == TwoS1 ) &&
              ( sector_irrep_up[ cnt ] == I1 ) && ( sector_spin_down[ cnt ] == TwoS2 ) ) {
            return cnt;
         }
      }
   }

   return -1;
}

int CheMPS2::CTensorOperator::gKappa2index( const int kappa ) const {
   return kappa2index[ kappa ];
}

dcomplex * CheMPS2::CTensorOperator::gStorage( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 ) {
   int kappa = gKappa( N1, TwoS1, I1, N2, TwoS2, I2 );
   if ( kappa == -1 ) {
      return NULL;
   }
   return storage + kappa2index[ kappa ];
}

int CheMPS2::CTensorOperator::gIndex() const { return index; }

int CheMPS2::CTensorOperator::get_2j() const { return two_j; }

int CheMPS2::CTensorOperator::get_nelec() const { return n_elec; }

int CheMPS2::CTensorOperator::get_irrep() const { return n_irrep; }

void CheMPS2::CTensorOperator::clear() {
   for ( int cnt = 0; cnt < kappa2index[ nKappa ]; cnt++ ) {
      storage[ cnt ] = 0.0;
   }
}

void CheMPS2::CTensorOperator::update( CTensorOperator * previous, CTensorT * den_up, CTensorT * den_down, dcomplex * workmem ) {
   clear();

   if ( moving_right ) {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         update_moving_right( ikappa, previous, den_up, den_down, workmem );
      }
   } else {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         update_moving_left( ikappa, previous, den_up, den_down, workmem );
      }
   }
}

void CheMPS2::CTensorOperator::update_moving_right( const int ikappa, CTensorOperator * previous,
                                                    CTensorT * den_up, CTensorT * den_down, dcomplex * workmem ) {
   const int NRU    = sector_nelec_up[ ikappa ];
   const int NRD    = NRU + n_elec;
   const int TwoSRU = sector_spin_up[ ikappa ];
   const int TwoSRD = sector_spin_down[ ikappa ];
   const int IRU    = sector_irrep_up[ ikappa ];
   const int IRD    = Irreps::directProd( IRU, n_irrep );

   int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );

   int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

   if ( dimRU > 0 && dimRD > 0 ) {
      for ( int geval = 0; geval < 6; geval++ ) {
         int NLU, NLD, TwoSLU, TwoSLD, ILU, ILD;
         switch ( geval ) {
            case 0: // MPS tensor sector (I,J,N) = (0,0,0)
               TwoSLU = TwoSRU;
               TwoSLD = TwoSRD;
               NLU    = NRU;
               NLD    = NRD;
               ILU    = IRU;
               ILD    = IRD;
               break;
            case 1: // MPS tensor sector (I,J,N) = (0,0,2)
               TwoSLU = TwoSRU;
               TwoSLD = TwoSRD;
               NLU    = NRU - 2;
               NLD    = NRD - 2;
               ILU    = IRU;
               ILD    = IRD;
               break;
            case 2: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSLU = TwoSRU - 1;
               TwoSLD = TwoSRD - 1;
               NLU    = NRU - 1;
               NLD    = NRD - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               ILD    = Irreps::directProd( IRD, bk_up->gIrrep( index - 1 ) );
               break;
            case 3: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSLU = TwoSRU - 1;
               TwoSLD = TwoSRD + 1;
               NLU    = NRU - 1;
               NLD    = NRD - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               ILD    = Irreps::directProd( IRD, bk_up->gIrrep( index - 1 ) );
               break;
            case 4: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSLU = TwoSRU + 1;
               TwoSLD = TwoSRD - 1;
               NLU    = NRU - 1;
               NLD    = NRD - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               ILD    = Irreps::directProd( IRD, bk_up->gIrrep( index - 1 ) );
               break;
            case 5: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSLU = TwoSRU + 1;
               TwoSLD = TwoSRD + 1;
               NLU    = NRU - 1;
               NLD    = NRD - 1;
               ILU    = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );
               ILD    = Irreps::directProd( IRD, bk_up->gIrrep( index - 1 ) );
               break;
         }

         if ( abs( TwoSLU - TwoSLD ) <= two_j ) {
            int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
            int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );

            if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
               dcomplex * block_up   = den_up->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
               dcomplex * block_down = den_down->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
               dcomplex * left_block = previous->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

               // Prefactor
               dcomplex alpha = 1.0;
               if ( geval >= 2 ) {
                  if ( two_j == 0 ) {
                     alpha = ( ( jw_phase ) ? -1.0 : 1.0 );
                  } else {
                     if ( prime_last ) {
                        alpha = Special::phase( TwoSRU + TwoSLD + two_j + ( ( jw_phase ) ? 3 : 1 ) ) *
                                sqrt( ( TwoSLD + 1.0 ) * ( TwoSRU + 1.0 ) ) * Wigner::wigner6j( TwoSLU, TwoSLD, two_j, TwoSRD, TwoSRU, 1 );
                     } else {
                        alpha = Special::phase( TwoSRD + TwoSLU + two_j + ( ( jw_phase ) ? 3 : 1 ) ) *
                                sqrt( ( TwoSLU + 1.0 ) * ( TwoSRD + 1.0 ) ) * Wigner::wigner6j( TwoSLD, TwoSLU, two_j, TwoSRU, TwoSRD, 1 );
                     }
                  }
               }

               char cotrans  = 'C';
               char notrans  = 'N';
               dcomplex beta = 0.0; // set
               zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, block_up, &dimLU, left_block, &dimLU, &beta, workmem, &dimRU );

               alpha = 1.0;
               beta  = 1.0; // add
               zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem, &dimRU, block_down, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
            }
         }
      }
   }
}

void CheMPS2::CTensorOperator::update_moving_left( const int ikappa, CTensorOperator * previous, CTensorT * den_up, CTensorT * den_down, dcomplex * workmem ) {
   const int NLU    = sector_nelec_up[ ikappa ];
   const int NLD    = NLU + n_elec;
   const int TwoSLU = sector_spin_up[ ikappa ];
   const int TwoSLD = sector_spin_down[ ikappa ];
   const int ILU    = sector_irrep_up[ ikappa ];
   const int ILD    = Irreps::directProd( ILU, n_irrep );

   int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
   int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

   if ( dimLU > 0 && dimLD > 0 ) {
      for ( int geval = 0; geval < 6; geval++ ) {
         int NRU, NRD, TwoSRU, TwoSRD, IRU, IRD;
         switch ( geval ) {
            case 0: // MPS tensor sector (I,J,N) = (0,0,0)
               TwoSRU = TwoSLU;
               TwoSRD = TwoSLD;
               NRU    = NLU;
               NRD    = NLD;
               IRU    = ILU;
               IRD    = ILD;
               break;
            case 1: // MPS tensor sector (I,J,N) = (0,0,2)
               TwoSRU = TwoSLU;
               TwoSRD = TwoSLD;
               NRU    = NLU + 2;
               NRD    = NLD + 2;
               IRU    = ILU;
               IRD    = ILD;
               break;
            case 2: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSRU = TwoSLU - 1;
               TwoSRD = TwoSLD - 1;
               NRU    = NLU + 1;
               NRD    = NLD + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               IRD    = Irreps::directProd( ILD, bk_up->gIrrep( index ) );
               break;
            case 3: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSRU = TwoSLU - 1;
               TwoSRD = TwoSLD + 1;
               NRU    = NLU + 1;
               NRD    = NLD + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               IRD    = Irreps::directProd( ILD, bk_up->gIrrep( index ) );
               break;
            case 4: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSRU = TwoSLU + 1;
               TwoSRD = TwoSLD - 1;
               NRU    = NLU + 1;
               NRD    = NLD + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               IRD    = Irreps::directProd( ILD, bk_up->gIrrep( index ) );
               break;
            case 5: // MPS tensor sector (I,J,N) = (Ilocal,1/2,1)
               TwoSRU = TwoSLU + 1;
               TwoSRD = TwoSLD + 1;
               NRU    = NLU + 1;
               NRD    = NLD + 1;
               IRU    = Irreps::directProd( ILU, bk_up->gIrrep( index ) );
               IRD    = Irreps::directProd( ILD, bk_up->gIrrep( index ) );
               break;
         }

         if ( abs( TwoSRU - TwoSRD ) <= two_j ) {
            int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
            int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );

            if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
               dcomplex * block_up    = den_up->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
               dcomplex * block_down  = den_down->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );
               dcomplex * right_block = previous->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

               // Prefactor
               dcomplex alpha = 1.0;
               if ( geval >= 2 ) {
                  if ( two_j == 0 ) {
                     alpha = ( ( jw_phase ) ? -1.0 : 1.0 ) * ( ( TwoSRU + 1.0 ) / ( TwoSLU + 1 ) );
                  } else {
                     if ( prime_last ) {
                        alpha = Special::phase( TwoSRU + TwoSLD + two_j + ( ( jw_phase ) ? 3 : 1 ) ) *
                                ( TwoSRD + 1 ) * sqrt( ( TwoSRU + 1.0 ) / ( TwoSLD + 1 ) ) * Wigner::wigner6j( TwoSRU, TwoSRD, two_j, TwoSLD, TwoSLU, 1 );
                     } else {
                        alpha = Special::phase( TwoSRD + TwoSLU + two_j + ( ( jw_phase ) ? 3 : 1 ) ) *
                                ( TwoSRU + 1 ) * sqrt( ( TwoSRD + 1.0 ) / ( TwoSLU + 1 ) ) * Wigner::wigner6j( TwoSRD, TwoSRU, two_j, TwoSLU, TwoSLD, 1 );
                     }
                  }
               }

               // prefactor * block_up * right_block --> mem
               char cotrans  = 'C';
               char notrans  = 'N';
               dcomplex beta = 0.0; // set
               zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, block_up, &dimLU, right_block, &dimRU, &beta, workmem, &dimLU );

               // mem * block_down^T --> storage
               alpha = 1.0;
               beta  = 1.0; // add
               zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem, &dimLU, block_down, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
            }
         }
      }
   }
}

void CheMPS2::CTensorOperator::zaxpy( dcomplex alpha, CTensorOperator * to_add ) {
   assert( nKappa == to_add->gNKappa() );
   assert( kappa2index[ nKappa ] == to_add->gKappa2index( to_add->gNKappa() ) );
   int inc = 1;
   zaxpy_( kappa2index + nKappa, &alpha, to_add->gStorage(), &inc, storage, &inc );
}

void CheMPS2::CTensorOperator::zaxpy_tensorCD( dcomplex alpha, CTensorOperator * to_add ) {
   assert( nKappa == to_add->gNKappa() );
   assert( kappa2index[ nKappa ] == to_add->gKappa2index( to_add->gNKappa() ) );
   assert( n_elec == 0 );
   assert( ( two_j == 0 ) || ( two_j == 2 ) );

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int IRU   = sector_irrep_up[ ikappa ];
      const int IRD   = Irreps::directProd( IRU, n_irrep );
      const int TwoSU = sector_spin_up[ ikappa ];
      const int TwoSD = sector_spin_down[ ikappa ];
      const int N     = sector_nelec_up[ ikappa ];

      const int dim_up   = bk_up->gCurrentDim( index, N, TwoSU, IRU );
      const int dim_down = bk_down->gCurrentDim( index, N, TwoSD, IRD );

      if ( dim_up > 0 && dim_down > 0 ) {
         dcomplex prefactor = alpha;

         if ( TwoSU != TwoSD ) {
            prefactor *= Special::phase( TwoSU - TwoSD ) * sqrt( ( moving_right ) ? ( ( TwoSU + 1.0 ) / ( TwoSD + 1 ) )
                                                                                  : ( ( TwoSD + 1.0 ) / ( TwoSU + 1 ) ) );
         }

         dcomplex * block = to_add->gStorage( N, TwoSU, IRU, N, TwoSD, IRD );

         for ( int irow = 0; irow < dim_up; irow++ ) {
            for ( int icol = 0; icol < dim_down; icol++ ) {
               storage[ kappa2index[ ikappa ] + irow + dim_up * icol ] +=
                   prefactor * block[ irow + dim_up * icol ];
            }
         }
      }
   }
}

void CheMPS2::CTensorOperator::zaxpy_tensorCTDT( dcomplex alpha, CTensorOperator * to_add ) {
   assert( nKappa == to_add->gNKappa() );
   assert( kappa2index[ nKappa ] == to_add->gKappa2index( to_add->gNKappa() ) );
   assert( n_elec == 0 );
   assert( ( two_j == 0 ) || ( two_j == 2 ) );

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int IRU   = sector_irrep_up[ ikappa ];
      const int IRD   = Irreps::directProd( IRU, n_irrep );
      const int TwoSU = sector_spin_up[ ikappa ];
      const int TwoSD = sector_spin_down[ ikappa ];
      const int N     = sector_nelec_up[ ikappa ];

      const int dim_up   = bk_up->gCurrentDim( index, N, TwoSU, IRU );
      const int dim_down = bk_down->gCurrentDim( index, N, TwoSD, IRD );

      if ( dim_up > 0 && dim_down > 0 ) {
         dcomplex prefactor = alpha;

         if ( TwoSU != TwoSD ) {
            prefactor *= Special::phase( TwoSD - TwoSU ) * sqrt( ( moving_right ) ? ( ( TwoSD + 1.0 ) / ( TwoSU + 1 ) )
                                                                                  : ( ( TwoSU + 1.0 ) / ( TwoSD + 1 ) ) );
         }

         dcomplex * block = to_add->gStorage( N, TwoSU, IRU, N, TwoSD, IRD );
         for ( int irow = 0; irow < dim_up; irow++ ) {
            for ( int icol = 0; icol < dim_down; icol++ ) {
               storage[ kappa2index[ ikappa ] + irow + dim_up * icol ] += prefactor * block[ irow + dim_up * icol ];
            }
         }
      }
   }
}

void CheMPS2::CTensorOperator::zaxpy_transpose_tensorCD( dcomplex alpha, CTensorOperator * to_add ) {
   assert( nKappa == to_add->gNKappa() );
   assert( kappa2index[ nKappa ] == to_add->gKappa2index( to_add->gNKappa() ) );
   assert( n_elec == 0 );
   assert( ( two_j == 0 ) || ( two_j == 2 ) );

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int IRU   = sector_irrep_up[ ikappa ];
      const int IRD   = Irreps::directProd( IRU, n_irrep );
      const int TwoSU = sector_spin_up[ ikappa ];
      const int TwoSD = sector_spin_down[ ikappa ];
      const int N     = sector_nelec_up[ ikappa ];

      const int dimU = bk_up->gCurrentDim( index, N, TwoSU, IRU );
      const int dimD = bk_down->gCurrentDim( index, N, TwoSD, IRD );

      if ( dimU > 0 && dimD > 0 ) {
         dcomplex prefactor = alpha;
         if ( TwoSU != TwoSD ) {
            prefactor *= Special::phase( TwoSU - TwoSD ) * sqrt( ( moving_right ) ? ( ( TwoSU + 1.0 ) / ( TwoSD + 1 ) )
                                                                                  : ( ( TwoSD + 1.0 ) / ( TwoSU + 1 ) ) );
         }

         dcomplex * block = to_add->gStorage( N, TwoSD, IRD, N, TwoSU, IRU );
         for ( int irow = 0; irow < dimU; irow++ ) {
            for ( int icol = 0; icol < dimD; icol++ ) {
               storage[ kappa2index[ ikappa ] + irow + dimU * icol ] += prefactor * std::conj( block[ icol + dimD * irow ] );
            }
         }
      }
   }
}

dcomplex CheMPS2::CTensorOperator::inproduct( CTensorOperator * buddy, const char trans ) const {
   if ( buddy == NULL ) {
      return 0.0;
   }

   assert( get_2j() == buddy->get_2j() );
   assert( n_elec == buddy->get_nelec() );
   assert( n_irrep == buddy->get_irrep() );

   dcomplex value = 0.0;
   int inc        = 1;

   if ( trans == 'N' ) {
      zdotc_( kappa2index + nKappa, storage, &inc, buddy->gStorage(), &inc );
   } else {
      assert( n_elec == 0 );
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         const int N     = sector_nelec_up[ ikappa ];
         const int TwoJU = sector_spin_up[ ikappa ];
         const int TwoJD = sector_spin_down[ ikappa ];
         const int IRU   = sector_irrep_up[ ikappa ];
         const int IRD   = Irreps::directProd( IRU, n_irrep );

         dcomplex * my_block    = storage + kappa2index[ ikappa ];
         dcomplex * buddy_block = buddy->gStorage( N, TwoJD, IRD, N, TwoJU, IRU );

         const int dimU = bk_up->gCurrentDim( index, N, TwoJU, IRU );
         const int dimD = bk_down->gCurrentDim( index, N, TwoJD, IRD );

         if ( dimU > 0 && dimD > 0 ) {
            dcomplex temp = 0.0;
            for ( int row = 0; row < dimU; row++ ) {
               for ( int col = 0; col < dimD; col++ ) {
                  temp += my_block[ row + dimU * col ] * std::conj( buddy_block[ col + dimD * row ] );
               }
            }

            const dcomplex prefactor =
                ( ( get_2j() == 0 ) ? 1.0 : ( sqrt( ( TwoJU + 1.0 ) / ( TwoJD + 1 ) ) * Special::phase( TwoJU - TwoJD ) ) );
            value += prefactor * temp;
         }
      }
   }

   return value;
}

std::ostream & CheMPS2::operator<<( std::ostream & os, const CheMPS2::CTensorOperator & tns ) {
   os << CheMPS2::hashline;

   os << "CTensorOperator with " << tns.gNKappa() << " symmetry blocks: " << std::endl;
   os << "TwoJ:    " << tns.two_j << std::endl;
   os << "nEle:    " << tns.n_elec << std::endl;
   os << "nIrr:    " << tns.n_irrep << std::endl;
   for ( int ikappa = 0; ikappa < tns.nKappa; ++ikappa ) {
      const int Nup = tns.sector_nelec_up[ ikappa ];
      const int Iup = tns.sector_irrep_up[ ikappa ];
      const int Sup = tns.sector_spin_up[ ikappa ];
      const int Sdn = tns.sector_spin_down[ ikappa ];

      os << "Block number " << ikappa << std::endl;
      os << "Nup:    " << Nup << std::endl;
      os << "Sup:    " << Sup << std::endl;
      os << "Iup:    " << Iup << std::endl;
      os << "Ndn:    " << Nup + tns.n_elec << std::endl;
      os << "Sdn:    " << Sdn << std::endl;
      os << "Idn:    " << Irreps::directProd( tns.n_irrep, Iup ) << std::endl;

      for ( int i = tns.kappa2index[ ikappa ]; i < tns.kappa2index[ ikappa + 1 ]; ++i ) {
         os << tns.storage[ i ] << std::endl;
      }
   }
   os << CheMPS2::hashline;
   return os;
}