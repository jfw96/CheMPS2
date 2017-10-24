
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "CTensorQ.h"
#include "Lapack.h"
#include "Wigner.h"

CheMPS2::CTensorQ::CTensorQ( const int boundary_index, const int Idiff,
                             const bool moving_right,
                             const SyBookkeeper * book_up,
                             const SyBookkeeper * book_down,
                             const Problem * Prob,
                             const int site )
    : CTensorOperator( boundary_index, // index
                       1,              // two_j
                       1,              // n_elec
                       Idiff,          // irrep
                       moving_right,   // direction
                       true,           // prime_last
                       true,           // jw_phase (three 2nd quantized operators)
                       book_up,        // upper bookkeeper
                       book_down       // lower bookkeeper
                       ),
      site( site ) {
   this->Prob = Prob;
}

CheMPS2::CTensorQ::~CTensorQ() {}

void CheMPS2::CTensorQ::AddTermSimple( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   if ( ( moving_right ) && ( bk_up->gIrrep( denTup->gIndex() ) == n_irrep ) ) {
      AddTermSimpleRight( denTup, denTdown, previous, workmem );
   }
   if ( ( !moving_right ) && ( bk_up->gIrrep( denTup->gIndex() ) == n_irrep ) ) {
      AddTermSimpleLeft( denTup, denTdown, previous, workmem );
   }
}

void CheMPS2::CTensorQ::AddTermSimpleRight( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const double mxElement =
       Prob->gMxElement( index - 1, index - 1, index - 1, site );

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NRU    = sector_nelec_up[ ikappa ];
      const int TwoSRU = sector_spin_up[ ikappa ];
      const int IRU    = sector_irrep_up[ ikappa ];

      const int NRD    = NRU + 1;
      const int TwoSRD = sector_spin_down[ ikappa ];
      const int IRD    = Irreps::directProd( n_irrep, IRU );

      const int NL    = NRU - 1;
      const int TwoSL = TwoSRD;
      const int IL    = IRD;

      int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
      int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

      int dimLU = bk_up->gCurrentDim( index - 1, NL, TwoSL, IL );
      int dimLD = bk_down->gCurrentDim( index - 1, NL, TwoSL, IL );

      char cotrans = 'C';
      char notrans = 'N';

      if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
         if ( previous == NULL ) {
            assert( dimLU == dimLD );

            dcomplex * BlockTup = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );

            int fase       = ( ( ( ( TwoSRD + 1 - TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex alpha = fase * mxElement * sqrt( ( TwoSRU + 1.0 ) / ( TwoSRD + 1.0 ) );
            dcomplex beta  = 1.0; // add
            zgemm_( &cotrans, &notrans, &dimRU, &dimRD, &dimLU, &alpha, BlockTup, &dimLU, BlockTdo, &dimLU, &beta, storage + kappa2index[ ikappa ], &dimRU );

         } else {
            dcomplex * BlockTup   = denTup->gStorage( NL, TwoSL, IL, NRU, TwoSRU, IRU );
            dcomplex * BlockTdown = denTdown->gStorage( NL, TwoSL, IL, NRD, TwoSRD, IRD );
            dcomplex * BlockPrev  = previous->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

            int fase       = ( ( ( ( TwoSRD + 1 - TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex alpha = fase * mxElement * sqrt( ( TwoSRU + 1.0 ) / ( TwoSRD + 1.0 ) );

            dcomplex set = 0.0;
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, BlockPrev, &dimLU, &set, workmem, &dimRU );

            dcomplex one = 1.0;
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &one, workmem, &dimRU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermSimpleLeft( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem ) {
   const double mxElement = Prob->gMxElement( site, index, index, index );

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NLU    = sector_nelec_up[ ikappa ];
      const int TwoSLU = sector_spin_up[ ikappa ];
      const int ILU    = sector_irrep_up[ ikappa ];

      const int NLD    = NLU + 1;
      const int TwoSLD = sector_spin_down[ ikappa ];
      const int ILD    = Irreps::directProd( n_irrep, ILU );

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

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );

            int fase       = ( ( ( ( TwoSLU + 1 - TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex alpha = fase * mxElement * sqrt( ( TwoSLU + 1.0 ) / ( TwoSLD + 1.0 ) );
            dcomplex beta  = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRU, &alpha, BlockTup, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         } else {
            dcomplex * BlockTup   = denTup->gStorage( NLU, TwoSLU, ILU, NR, TwoSR, IR );
            dcomplex * BlockTdown = denTdown->gStorage( NLD, TwoSLD, ILD, NR, TwoSR, IR );
            dcomplex * BlockPrev  = previous->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

            int fase       = ( ( ( ( TwoSLU + 1 - TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex alpha = fase * mxElement * sqrt( ( TwoSLU + 1.0 ) / ( TwoSLD + 1.0 ) );

            dcomplex set = 0.0;
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, BlockPrev, &dimRU, &set, workmem, &dimLU );

            dcomplex one = 1.0;
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &one, workmem, &dimLU, BlockTdown, &dimLD, &one, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermsL( CTensorL ** Ltensors, CTensorLT ** LtensorsT, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {
   if ( moving_right ) {
      AddTermsLRight( Ltensors, LtensorsT, denTup, denTdown, workmem, workmem2 );
   } else {
      AddTermsLLeft( Ltensors, LtensorsT, denTup, denTdown, workmem, workmem2 );
   }
}

void CheMPS2::CTensorQ::AddTermsLRight( CTensorL ** Ltensors, CTensorLT ** LtensorsT, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   bool OneToAdd = false;
   for ( int loca = 0; loca < index - 1; loca++ ) {
      if ( Ltensors[ index - 2 - loca ]->get_irrep() == n_irrep ||
           LtensorsT[ index - 2 - loca ]->get_irrep() == n_irrep ) {
         OneToAdd = true;
      }
   }

   char cotrans = 'C';
   char notrans = 'N';

   if ( OneToAdd ) {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         const int NRU    = sector_nelec_up[ ikappa ];
         const int TwoSRU = sector_spin_up[ ikappa ];
         const int IRU    = sector_irrep_up[ ikappa ];

         const int NRD    = NRU + 1;
         const int TwoSRD = sector_spin_down[ ikappa ];
         const int IRD    = Irreps::directProd( IRU, n_irrep );

         int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

         // case 1
         int NLU    = NRU;
         int TwoSLU = TwoSRU;
         int ILU    = IRU;

         int NLD    = NRU - 1;
         int TwoSLD = TwoSRD;
         int ILD    = IRD;

         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            int dimLUxLD = dimLU * dimLD;
            for ( int cnt = 0; cnt < dimLUxLD; cnt++ ) {
               workmem[ cnt ] = 0.0;
            }

            for ( int loca = 0; loca < index - 1; loca++ ) {
               if ( LtensorsT[ index - 2 - loca ]->get_irrep() == n_irrep ) {
                  dcomplex * BlockL = LtensorsT[ index - 2 - loca ]->gStorage(
                      NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
                  dcomplex alpha = Prob->gMxElement( loca, site, index - 1, index - 1 );
                  int inc        = 1;
                  zaxpy_( &dimLUxLD, &alpha, BlockL, &inc, workmem, &inc );
               }
            }

            int fase       = ( ( ( ( TwoSRD + 1 - TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex alpha = fase * sqrt( ( TwoSRU + 1.0 ) / ( TwoSRD + 1.0 ) );
            dcomplex beta  = 0.0; // set

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            // factor * Tup^T * LT --> mem2
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, workmem, &dimLU, &beta, workmem2, &dimRU );

            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }

         // case 2
         NLU    = NRU - 2;
         TwoSLU = TwoSRU;
         ILU    = IRU;

         NLD    = NRU - 1;
         TwoSLD = TwoSRD;
         ILD    = IRD;

         dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            int dimLUxLD = dimLU * dimLD;
            for ( int cnt = 0; cnt < dimLUxLD; cnt++ ) {
               workmem[ cnt ] = 0.0;
            }

            for ( int loca = 0; loca < index - 1; loca++ ) {
               if ( Ltensors[ index - 2 - loca ]->get_irrep() == n_irrep ) {
                  dcomplex * BlockL = Ltensors[ index - 2 - loca ]->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
                  dcomplex alpha    = 2 * Prob->gMxElement( loca, index - 1, site, index - 1 ) - Prob->gMxElement( loca, index - 1, index - 1, site );
                  int inc           = 1;
                  zaxpy_( &dimLUxLD, &alpha, BlockL, &inc, workmem, &inc );
               }
            }

            dcomplex alpha = 1.0; // factor = 1 in this case
            dcomplex beta  = 0.0; // set

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            // factor * Tup^T * L --> mem2
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, workmem, &dimLU, &beta, workmem2, &dimRU );

            beta = 1.0; // add
            // mem2 * Tdo --> storage
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }

         // case 3
         NLU = NRU - 1;
         ILU = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );

         NLD = NRU;
         ILD = Irreps::directProd( IRD, bk_up->gIrrep( index - 1 ) );

         for ( TwoSLU = TwoSRU - 1; TwoSLU <= TwoSRU + 1; TwoSLU += 2 ) {
            for ( TwoSLD = TwoSRD - 1; TwoSLD <= TwoSRD + 1; TwoSLD += 2 ) {
               if ( ( TwoSLD >= 0 ) && ( TwoSLU >= 0 ) && ( abs( TwoSLD - TwoSLU ) < 2 ) ) {
                  dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
                  dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
                  if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
                     int fase        = ( ( ( ( TwoSRU + TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
                     dcomplex factor = fase * sqrt( ( TwoSLD + 1 ) * ( TwoSRU + 1.0 ) ) * Wigner::wigner6j( TwoSRU, TwoSRD, 1, TwoSLD, TwoSLU, 1 );

                     int dimLUxLD = dimLU * dimLD;
                     for ( int cnt = 0; cnt < dimLUxLD; cnt++ ) {
                        workmem[ cnt ] = 0.0;
                     }

                     for ( int loca = 0; loca < index - 1; loca++ ) {
                        if ( Ltensors[ index - 2 - loca ]->get_irrep() == n_irrep ) {
                           dcomplex * BlockL = Ltensors[ index - 2 - loca ]->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
                           dcomplex val      = Prob->gMxElement( loca, index - 1, site, index - 1 );
                           dcomplex alpha    = factor * Prob->gMxElement( loca, index - 1, site, index - 1 );
                           if ( TwoSLD == TwoSRU ) {
                              alpha += Prob->gMxElement( loca, index - 1, index - 1, site );
                           }
                           int inc = 1;
                           zaxpy_( &dimLUxLD, &alpha, BlockL, &inc, workmem, &inc );
                        }
                     }

                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; // set

                     dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
                     dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

                     // Tup^T * mem --> mem2
                     zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, workmem, &dimLU, &beta, workmem2, &dimRU );
                     beta = 1.0; // add
                     // mem2 * Tdo --> storage
                     zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermsLLeft( CTensorL ** Ltensors, CTensorLT ** LtensorsT, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   bool OneToAdd = false;
   for ( int loca = index + 1; loca < Prob->gL(); loca++ ) {
      if ( Ltensors[ loca - index - 1 ]->get_irrep() == n_irrep ||
           LtensorsT[ loca - index - 1 ]->get_irrep() == n_irrep ) {
         OneToAdd = true;
      }
   }

   char cotrans = 'C';
   char notrans = 'N';

   if ( OneToAdd ) {
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         const int NLU    = sector_nelec_up[ ikappa ];
         const int TwoSLU = sector_spin_up[ ikappa ];
         const int ILU    = sector_irrep_up[ ikappa ];

         const int NLD    = NLU + 1;
         const int TwoSLD = sector_spin_down[ ikappa ];
         const int ILD    = Irreps::directProd( ILU, n_irrep );

         int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

         // case 1
         int NRU    = NLU + 2;
         int TwoSRU = TwoSLU;
         int IRU    = ILU;

         int NRD    = NLU + 1;
         int TwoSRD = TwoSLD;
         int IRD    = ILD;

         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            int dimRUxRD = dimRU * dimRD;
            for ( int cnt = 0; cnt < dimRUxRD; cnt++ ) {
               workmem[ cnt ] = 0.0;
            }

            for ( int loca = index + 1; loca < Prob->gL(); loca++ ) {
               if ( LtensorsT[ loca - index - 1 ]->get_irrep() == n_irrep ) {
                  dcomplex * BlockL = LtensorsT[ loca - index - 1 ]->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
                  dcomplex alpha    = Prob->gMxElement( site, loca, index, index );
                  int inc           = 1;
                  zaxpy_( &dimRUxRD, &alpha, BlockL, &inc, workmem, &inc );
               }
            }

            int fase       = ( ( ( ( TwoSLU + 1 - TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex alpha = fase * sqrt( ( TwoSLU + 1.0 ) / ( TwoSLD + 1.0 ) );
            dcomplex beta  = 0.0; // set

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            // factor * Tup * LT --> mem2
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, workmem, &dimRU, &beta, workmem2, &dimLU );
            alpha = 1.0;
            beta  = 1.0; // add
            // mem2 * Tdo^T --> storage
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }

         // case 2
         NRU    = NLU + 2;
         TwoSRU = TwoSLU;
         IRU    = ILU;

         NRD    = NLU + 3;
         TwoSRD = TwoSLD;
         IRD    = ILD;

         dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            int dimRUxRD = dimRU * dimRD;
            for ( int cnt = 0; cnt < dimRUxRD; cnt++ ) {
               workmem[ cnt ] = 0.0;
            }

            for ( int loca = index + 1; loca < Prob->gL(); loca++ ) {
               if ( Ltensors[ loca - index - 1 ]->get_irrep() == n_irrep ) {
                  dcomplex * BlockL = Ltensors[ loca - index - 1 ]->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
                  dcomplex alpha    = 2 * Prob->gMxElement( site, index, loca, index ) - Prob->gMxElement( site, index, index, loca );
                  int inc           = 1;
                  zaxpy_( &dimRUxRD, &alpha, BlockL, &inc, workmem, &inc );
               }
            }

            dcomplex alpha = 1.0; // factor = 1 in this case
            dcomplex beta  = 0.0; // set

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            // factor * Tup * L --> mem2
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, workmem, &dimRU, &beta, workmem2, &dimLU );
            beta = 1.0; // add
            // mem2 * Tdo^T --> storage
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }

         // case 3
         NRU = NLU + 1;
         IRU = Irreps::directProd( ILU, bk_up->gIrrep( index ) );

         NRD = NLU + 2;
         IRD = Irreps::directProd( ILD, bk_up->gIrrep( index ) );

         for ( int TwoSRU = TwoSLU - 1; TwoSRU <= TwoSLU + 1; TwoSRU += 2 ) {
            for ( int TwoSRD = TwoSLD - 1; TwoSRD <= TwoSLD + 1; TwoSRD += 2 ) {
               if ( ( TwoSRD >= 0 ) && ( TwoSRU >= 0 ) && ( abs( TwoSRD - TwoSRU ) < 2 ) ) {
                  dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
                  dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
                  if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
                     int fase         = ( ( ( ( TwoSLD + TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
                     dcomplex factor1 = fase * sqrt( ( TwoSRU + 1.0 ) / ( TwoSLD + 1.0 ) ) * ( TwoSRD + 1 ) * Wigner::wigner6j( TwoSLU, TwoSLD, 1, TwoSRD, TwoSRU, 1 );
                     dcomplex factor2 = ( TwoSRD + 1.0 ) / ( TwoSLD + 1.0 );

                     int dimRUxRD = dimRU * dimRD;
                     for ( int cnt = 0; cnt < dimRUxRD; cnt++ ) {
                        workmem[ cnt ] = 0.0;
                     }

                     for ( int loca = index + 1; loca < Prob->gL(); loca++ ) {
                        if ( Ltensors[ loca - index - 1 ]->get_irrep() == n_irrep ) {
                           dcomplex * BlockL = Ltensors[ loca - index - 1 ]->gStorage( NRU, TwoSRU, IRU,
                                                                                       NRD, TwoSRD, IRD );
                           dcomplex alpha = factor1 * Prob->gMxElement( site, index, loca, index );
                           if ( TwoSRU == TwoSLD ) {
                              alpha += factor2 * Prob->gMxElement( site, index, index, loca );
                           }
                           int inc = 1;
                           zaxpy_( &dimRUxRD, &alpha, BlockL, &inc, workmem, &inc );
                        }
                     }

                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; // set

                     dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
                     dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

                     // Tup * mem --> mem2
                     zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, workmem, &dimRU, &beta, workmem2, &dimLU );

                     beta = 1.0; // add mem2 * Tdo^T --> storage
                     zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermsAB( CTensorOperator * denA, CTensorOperator * denB, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   if ( moving_right ) {
      AddTermsABRight( denA, denB, denTup, denTdown, workmem, workmem2 );
   } else {
      AddTermsABLeft( denA, denB, denTup, denTdown, workmem, workmem2 );
   }
}

void CheMPS2::CTensorQ::AddTermsABRight( CTensorOperator * denA, CTensorOperator * denB, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   char cotrans = 'C';
   char notrans = 'N';

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NRU    = sector_nelec_up[ ikappa ];
      const int TwoSRU = sector_spin_up[ ikappa ];
      const int IRU    = sector_irrep_up[ ikappa ];

      const int NRD    = sector_nelec_up[ ikappa ] + 1;
      const int TwoSRD = sector_spin_down[ ikappa ];
      const int IRD    = Irreps::directProd( IRU, n_irrep );

      int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
      int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

      // case 1
      int NLU = NRU - 1;
      int ILU = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );

      int NLD    = NRU + 1;
      int TwoSLD = TwoSRD;
      int ILD    = IRD;

      for ( int TwoSLU = TwoSRU - 1; TwoSLU <= TwoSRU + 1; TwoSLU += 2 ) {
         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            int fase         = ( ( ( ( TwoSLU + TwoSRD + 2 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factorB = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) * Wigner::wigner6j( 1, 2, 1, TwoSRD, TwoSRU, TwoSLU );

            dcomplex alpha;
            dcomplex * mem;
            dcomplex * memTmp;

            if ( TwoSLU == TwoSRD ) {
               fase             = ( ( ( ( TwoSRD + 1 - TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               dcomplex factorA = fase * sqrt( 0.5 * ( TwoSRU + 1.0 ) / ( TwoSRD + 1.0 ) );

               dcomplex * BlockA = denA->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
               dcomplex * BlockB = denB->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

               memTmp = workmem;
               for ( int cnt = 0; cnt < dimLU * dimLD; cnt++ ) {
                  memTmp[ cnt ] = factorA * BlockA[ cnt ] + factorB * BlockB[ cnt ];
               }
               alpha = 1.0;
               mem   = memTmp;
            } else {
               alpha = factorB;
               mem   = denB->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
            }

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex beta = 0.0; // set
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, mem, &dimLU, &beta, workmem2, &dimRU );
            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }

      // case 2
      NLU        = NRU - 2;
      int TwoSLU = TwoSRU;
      ILU        = IRU;

      NLD = NRU;
      ILD = Irreps::directProd( IRD, bk_down->gIrrep( index - 1 ) );

      for ( TwoSLD = TwoSRD - 1; TwoSLD <= TwoSRD + 1; TwoSLD += 2 ) {
         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            int fase         = ( ( ( ( TwoSRU + TwoSRD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factorB = fase * sqrt( 3.0 * ( TwoSLD + 1 ) ) * Wigner::wigner6j( 1, 2, 1, TwoSRU, TwoSRD, TwoSLD );

            dcomplex alpha;
            dcomplex * mem;
            dcomplex * memTmp;

            if ( TwoSLD == TwoSRU ) {
               dcomplex factorA = -sqrt( 0.5 );

               dcomplex * BlockA = denA->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
               dcomplex * BlockB = denB->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );

               memTmp = workmem;
               for ( int cnt = 0; cnt < dimLU * dimLD; cnt++ ) {
                  memTmp[ cnt ] = factorA * BlockA[ cnt ] + factorB * BlockB[ cnt ];
               }
               alpha = 1.0;
               mem   = memTmp;

            } else {
               alpha = factorB;
               mem   = denB->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
            }

            dcomplex * BlockTup =
                denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo =
                denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex beta = 0.0; // set
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, mem, &dimLU, &beta, workmem2, &dimRU );
            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermsABLeft( CTensorOperator * denA, CTensorOperator * denB, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   char cotrans = 'C';
   char notrans = 'N';

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NLU    = sector_nelec_up[ ikappa ];
      const int TwoSLU = sector_spin_up[ ikappa ];
      const int ILU    = sector_irrep_up[ ikappa ];

      const int NLD    = NLU + 1;
      const int TwoSLD = sector_spin_down[ ikappa ];
      const int ILD    = Irreps::directProd( ILU, n_irrep );

      int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
      int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

      // case 1
      int NRU    = NLU;
      int TwoSRU = TwoSLU;
      int IRU    = ILU;

      int NRD = NLU + 2;
      int IRD = Irreps::directProd( ILD, bk_up->gIrrep( index ) );

      for ( int TwoSRD = TwoSLD - 1; TwoSRD <= TwoSLD + 1; TwoSRD += 2 ) {
         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            int fase = ( ( ( ( TwoSRD + TwoSLU + 2 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            const dcomplex factorB =
                fase * sqrt( 3.0 / ( TwoSLD + 1.0 ) ) * ( TwoSRD + 1 ) *
                Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSRD, TwoSLD );

            dcomplex alpha;
            dcomplex * mem;
            dcomplex * memTmp;

            if ( TwoSRD == TwoSLU ) {
               fase             = ( ( ( ( TwoSLU + 1 - TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               dcomplex factorA = fase * sqrt( 0.5 * ( TwoSLU + 1.0 ) / ( TwoSLD + 1.0 ) );

               dcomplex * BlockA = denA->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
               dcomplex * BlockB = denB->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

               memTmp = workmem;
               for ( int cnt = 0; cnt < dimRU * dimRD; cnt++ ) {
                  memTmp[ cnt ] = factorA * BlockA[ cnt ] + factorB * BlockB[ cnt ];
               }
               alpha = 1.0;
               mem   = memTmp;

            } else {
               alpha = factorB;
               mem   = denB->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
            }

            dcomplex * BlockTup =
                denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo =
                denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, mem, &dimRU, &beta, workmem2, &dimLU );

            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }

      // case 2
      NRU = NLU + 1;
      IRU = Irreps::directProd( ILU, bk_up->gIrrep( index ) );

      NRD        = NLU + 3;
      int TwoSRD = TwoSLD;
      IRD        = ILD;

      for ( TwoSRU = TwoSLU - 1; TwoSRU <= TwoSLU + 1; TwoSRU += 2 ) {
         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            int fase         = ( ( ( ( TwoSLU + TwoSLD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factorB = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) *
                               Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSLD, TwoSLU );

            dcomplex alpha;
            dcomplex * mem;
            dcomplex * memTmp;

            if ( TwoSRU == TwoSLD ) {
               dcomplex factorA = -sqrt( 0.5 );

               dcomplex * BlockA = denA->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
               dcomplex * BlockB = denB->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );

               memTmp = workmem;
               for ( int cnt = 0; cnt < dimRU * dimRD; cnt++ ) {
                  memTmp[ cnt ] = factorA * BlockA[ cnt ] + factorB * BlockB[ cnt ];
               }
               alpha = 1.0;
               mem   = memTmp;

            } else {
               alpha = factorB;
               mem   = denB->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
            }

            dcomplex * BlockTup =
                denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo =
                denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex beta = 0.0; // set
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, mem, &dimRU, &beta, workmem2, &dimLU );

            alpha = 1.0;
            beta  = 1.0; // add
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermsCD( CTensorOperator * denC, CTensorOperator * denD, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {
   if ( moving_right ) {
      AddTermsCDRight( denC, denD, denTup, denTdown, workmem, workmem2 );
   } else {
      AddTermsCDLeft( denC, denD, denTup, denTdown, workmem, workmem2 );
   }
}

void CheMPS2::CTensorQ::AddTermsCDRight( CTensorOperator * denC, CTensorOperator * denD, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   char cotrans = 'C';
   char notrans = 'N';

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NRU    = sector_nelec_up[ ikappa ];
      const int TwoSRU = sector_spin_up[ ikappa ];
      const int IRU    = sector_irrep_up[ ikappa ];

      const int NRD    = NRU + 1;
      const int TwoSRD = sector_spin_down[ ikappa ];
      const int IRD    = Irreps::directProd( IRU, n_irrep );

      int dimRU = bk_up->gCurrentDim( index, NRU, TwoSRU, IRU );
      int dimRD = bk_down->gCurrentDim( index, NRD, TwoSRD, IRD );

      // case 1
      int NLU    = NRU;
      int TwoSLU = TwoSRU;
      int ILU    = IRU;

      int NLD = NRU;
      int ILD = Irreps::directProd( IRD, bk_up->gIrrep( index - 1 ) );

      for ( int TwoSLD = TwoSRD - 1; TwoSLD <= TwoSRD + 1; TwoSLD += 2 ) {
         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );
         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            int dimLUxLD = dimLU * dimLD;

            // first set to D
            int fase        = ( ( ( ( TwoSRU + TwoSRD + 1 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factor = fase * sqrt( 3.0 * ( TwoSLD + 1 ) ) * Wigner::wigner6j( 1, 2, 1, TwoSRU, TwoSRD, TwoSLD );

            dcomplex * block = denD->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
            for ( int cnt = 0; cnt < dimLUxLD; cnt++ ) {
               workmem[ cnt ] = factor * block[ cnt ];
            }

            // add C
            if ( TwoSLD == TwoSRU ) {
               factor  = sqrt( 0.5 );
               block   = denC->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
               int inc = 1;
               zaxpy_( &dimLUxLD, &factor, block, &inc, workmem, &inc );
            }

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex alpha = 1.0;
            dcomplex beta  = 0.0;
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, workmem, &dimLU, &beta, workmem2, &dimRU );
            beta = 1.0;
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }

      // case 2
      NLU = NRU - 1;
      ILU = Irreps::directProd( IRU, bk_up->gIrrep( index - 1 ) );

      NLD        = NRU - 1;
      int TwoSLD = TwoSRD;
      ILD        = IRD;

      for ( int TwoSLU = TwoSRU - 1; TwoSLU <= TwoSRU + 1; TwoSLU += 2 ) {
         int dimLU = bk_up->gCurrentDim( index - 1, NLU, TwoSLU, ILU );
         int dimLD = bk_down->gCurrentDim( index - 1, NLD, TwoSLD, ILD );

         if ( ( dimLU > 0 ) && ( dimLD > 0 ) ) {
            int dimLUxLD = dimLU * dimLD;

            // first set to D
            int fase        = ( ( ( ( TwoSLU + TwoSRD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factor = fase * sqrt( 3.0 * ( TwoSRU + 1 ) ) *
                              Wigner::wigner6j( 1, 2, 1, TwoSRD, TwoSRU, TwoSLU );

            dcomplex * block =
                denD->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
            for ( int cnt = 0; cnt < dimLUxLD; cnt++ ) {
               workmem[ cnt ] = factor * block[ cnt ];
            }

            // add C
            if ( TwoSLU == TwoSRD ) {
               fase    = ( ( ( ( TwoSRD + 1 - TwoSRU ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               factor  = fase * sqrt( 0.5 * ( TwoSRU + 1.0 ) / ( TwoSRD + 1.0 ) );
               block   = denC->gStorage( NLU, TwoSLU, ILU, NLD, TwoSLD, ILD );
               int inc = 1;
               zaxpy_( &dimLUxLD, &factor, block, &inc, workmem, &inc );
            }

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex alpha = 1.0;
            dcomplex beta  = 0.0;
            zgemm_( &cotrans, &notrans, &dimRU, &dimLD, &dimLU, &alpha, BlockTup, &dimLU, workmem, &dimLU, &beta, workmem2, &dimRU );
            beta = 1.0;
            zgemm_( &notrans, &notrans, &dimRU, &dimRD, &dimLD, &alpha, workmem2, &dimRU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimRU );
         }
      }
   }
}

void CheMPS2::CTensorQ::AddTermsCDLeft( CTensorOperator * denC, CTensorOperator * denD, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 ) {

   char cotrans = 'C';
   char notrans = 'N';

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NLU    = sector_nelec_up[ ikappa ];
      const int TwoSLU = sector_spin_up[ ikappa ];
      const int ILU    = sector_irrep_up[ ikappa ];

      const int NLD    = NLU + 1;
      const int TwoSLD = sector_spin_down[ ikappa ];
      const int ILD    = Irreps::directProd( ILU, n_irrep );

      int dimLU = bk_up->gCurrentDim( index, NLU, TwoSLU, ILU );
      int dimLD = bk_down->gCurrentDim( index, NLD, TwoSLD, ILD );

      // case 1
      int NRU = NLU + 1;
      int IRU = Irreps::directProd( ILU, bk_up->gIrrep( index ) );

      int NRD    = NLU + 1;
      int TwoSRD = TwoSLD;
      int IRD    = ILD;

      for ( int TwoSRU = TwoSLU - 1; TwoSRU <= TwoSLU + 1; TwoSRU += 2 ) {
         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            int dimRUxRD = dimRU * dimRD;

            // first set to D
            int fase        = ( ( ( ( TwoSLU + TwoSRU + 3 ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factor = fase * sqrt( 3.0 / ( TwoSLD + 1.0 ) ) * ( TwoSRU + 1 ) *
                              Wigner::wigner6j( 1, 1, 2, TwoSRU, TwoSLD, TwoSLU );

            dcomplex * block =
                denD->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
            for ( int cnt = 0; cnt < dimRUxRD; cnt++ ) {
               workmem[ cnt ] = factor * block[ cnt ];
            }

            // add C
            if ( TwoSRU == TwoSLD ) {
               factor  = sqrt( 0.5 );
               block   = denC->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
               int inc = 1;
               zaxpy_( &dimRUxRD, &factor, block, &inc, workmem, &inc );
            }

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex alpha = 1.0;
            dcomplex beta  = 0.0;
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, workmem, &dimRU, &beta, workmem2, &dimLU );
            beta = 1.0;
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }

      // case 2
      NRU        = NLU + 2;
      int TwoSRU = TwoSLU;
      IRU        = ILU;

      NRD = NLU + 2;
      IRD = Irreps::directProd( ILD, bk_down->gIrrep( index ) );

      for ( TwoSRD = TwoSLD - 1; TwoSRD <= TwoSLD + 1; TwoSRD += 2 ) {
         int dimRU = bk_up->gCurrentDim( index + 1, NRU, TwoSRU, IRU );
         int dimRD = bk_down->gCurrentDim( index + 1, NRD, TwoSRD, IRD );
         if ( ( dimRU > 0 ) && ( dimRD > 0 ) ) {
            int dimRUxRD = dimRU * dimRD;

            // first set to D
            int fase        = ( ( ( TwoSRD + 1 ) % 2 ) != 0 ) ? -1 : 1;
            dcomplex factor = fase * sqrt( 3.0 * ( TwoSRD + 1.0 ) * ( TwoSLU + 1.0 ) / ( TwoSLD + 1.0 ) ) * Wigner::wigner6j( 1, 1, 2, TwoSLU, TwoSRD, TwoSLD );

            dcomplex * block = denD->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
            for ( int cnt = 0; cnt < dimRUxRD; cnt++ ) {
               workmem[ cnt ] = factor * block[ cnt ];
            }

            // add C
            if ( TwoSRD == TwoSLU ) {
               fase    = ( ( ( ( TwoSLU + 1 - TwoSLD ) / 2 ) % 2 ) != 0 ) ? -1 : 1;
               factor  = fase * sqrt( 0.5 * ( TwoSLU + 1.0 ) / ( TwoSLD + 1.0 ) );
               block   = denC->gStorage( NRU, TwoSRU, IRU, NRD, TwoSRD, IRD );
               int inc = 1;
               zaxpy_( &dimRUxRD, &factor, block, &inc, workmem, &inc );
            }

            dcomplex * BlockTup = denTup->gStorage( NLU, TwoSLU, ILU, NRU, TwoSRU, IRU );
            dcomplex * BlockTdo = denTdown->gStorage( NLD, TwoSLD, ILD, NRD, TwoSRD, IRD );

            dcomplex alpha = 1.0;
            dcomplex beta  = 0.0;
            zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimRU, &alpha, BlockTup, &dimLU, workmem, &dimRU, &beta, workmem2, &dimLU );

            beta = 1.0;
            zgemm_( &notrans, &cotrans, &dimLU, &dimLD, &dimRD, &alpha, workmem2, &dimLU, BlockTdo, &dimLD, &beta, storage + kappa2index[ ikappa ], &dimLU );
         }
      }
   }
}
