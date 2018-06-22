#include <iostream>

#include "COneDM.h"
#include "Special.h"
#include "Wigner.h"

CheMPS2::COneDM::COneDM( CTensorT ** mpsIn, const SyBookkeeper * denBKIn, const Problem * ProbIn )
    : L( denBKIn->gL() ), denBK( denBKIn ), Prob(ProbIn) {

   matrix      = new dcomplex[ L * L ];
   Ltensors    = new CTensorL **[ L ];
   F0tensors   = new CTensorF0 ***[ L ];
   F1tensors   = new CTensorF1 ***[ L ];
   Otensors    = new CTensorO *[ L ];
   isAllocated = new int[ L ];

   mps = new CTensorT *[ L ];
   for ( int index = 0; index < L; index++ ) {
      mps[ index ] = new CTensorT( mpsIn[ index ] );
   }

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      isAllocated[ cnt ] = 0;
   }

   for ( int cnt = 0; cnt < L; cnt++ ) {
      updateMovingRightSafe( cnt );
   }

   for ( int i = 0; i < L; i++ ) {
      for ( int j = 0; j <= i; j++ ) {
         matrix[ i + j * L ] = sqrt( 2 ) * F0tensors[ L - 1 ][ i - j ][ L - i - 1 ]->trace();
         matrix[ j + i * L ] = std::conj(matrix[ i + j * L ]);
      }
   }
}

CheMPS2::COneDM::~COneDM() {
   deleteAllBoundaryOperators();

   delete[] matrix;
   delete[] Ltensors;
   delete[] F0tensors;
   delete[] F1tensors;
   delete[] Otensors;   
   delete[] isAllocated;   

   for ( int site = 0; site < L; site++ ) {
      delete mps[ site ];
   }
   delete[] mps;
}

void CheMPS2::COneDM::gOEDMReDMRG( double * array ) {
   for ( int i = 0; i < L * L; i++ ) {
      array[ i ] = std::real( matrix[ i ] );
   }
}

void CheMPS2::COneDM::gOEDMImDMRG( double * array ) {
   for ( int i = 0; i < L * L; i++ ) {
      array[ i ] = std::imag( matrix[ i ] );
   }
}


void CheMPS2::COneDM::gOEDMReHamil( double * array ) {
   for ( int i = 0; i < L; i++ ) {
      for ( int j = 0; j < L; j++ ) {
         array[ i + L * j ] = std::real( matrix[ Prob->gf1( i ) + L * Prob->gf1( j ) ] );
      }
   }
}

void CheMPS2::COneDM::gOEDMImHamil( double * array ) {
   for ( int i = 0; i < L; i++ ) {
      for ( int j = 0; j < L; j++ ) {
         array[ i + L * j ] = std::imag( matrix[ Prob->gf1( i ) + L * Prob->gf1( j ) ] );
      }
   }
}


void CheMPS2::COneDM::updateMovingRightSafe( const int cnt ) {
   allocateTensors( cnt );
   isAllocated[ cnt ] = 1;
   updateMovingRight( cnt );
}

void CheMPS2::COneDM::updateMovingRight( const int index ) {

   const int dimL = denBK->gMaxDimAtBound( index );
   const int dimR = denBK->gMaxDimAtBound( index + 1 );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];

//Ltensors : all processes own all Ltensors
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == 0 ) {
               Ltensors[ index ][ cnt2 ]->create( mps[ index ], mps[ index ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( mps[ index ], mps[ index ], Otensors[ index - 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index - 1 ][ cnt2 - 1 ], mps[ index ], mps[ index ], workmem );
         }
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      const int k1          = index + 1;
      const int upperbound1 = ( k1 * ( k1 + 1 ) ) / 2;
      int result[ 2 ];
// After this parallel region, WAIT because F0,F1,S0,S1[ index ][ cnt2 ][ cnt3 == 0 ] is required for the complementary operators
#pragma omp for schedule( static )
      for ( int global = 0; global < upperbound1; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2 = index - result[ 1 ];
         const int cnt3 = result[ 0 ];
         if ( cnt3 == 0 ) { // Every MPI process owns the Operator[ index ][ cnt2 ][ cnt3 == 0 ]
            if ( cnt2 == 0 ) {
               if ( index == 0 ) {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mps[ index ], mps[ index ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mps[ index ], mps[ index ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mps[ index ], mps[ index ], Otensors[ index - 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mps[ index ], mps[ index ], Otensors[ index - 1 ], workmem );
               }
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mps[ index ], mps[ index ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mps[ index ], mps[ index ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mps[ index ], mps[ index ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mps[ index ], mps[ index ], workmem );
         }
      }

      delete[] workmem;
   }

   // Otensors
   if ( index == 0 ) {
      Otensors[ index ]->create( mps[ index ], mps[ index ] );
   } else {
      Otensors[ index ]->update_ownmem( mps[ index ], mps[ index ], Otensors[ index - 1 ] );
   }
}

void CheMPS2::COneDM::allocateTensors( const int index ) {

   // Ltensors : all processes own all Ltensors
   // To right: Ltens[cnt][cnt2] = operator on site cnt-cnt2; at boundary cnt+1
   Ltensors[ index ] = new CTensorL *[ index + 1 ];
   for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
      Ltensors[ index ][ cnt2 ] = new CTensorL( index + 1, denBK->gIrrep( index - cnt2 ), true, denBK, denBK );
   }

   //Two-operator tensors : certain processes own certain two-operator tensors
   //To right: F0tens[cnt][cnt2][cnt3] = operators on sites cnt-cnt3-cnt2 and cnt-cnt3; at boundary cnt+1
   F0tensors[ index ] = new CTensorF0 **[ index + 1 ];
   F1tensors[ index ] = new CTensorF1 **[ index + 1 ];
   for ( int cnt2 = 0; cnt2 < ( index + 1 ); cnt2++ ) {
      F0tensors[ index ][ cnt2 ] = new CTensorF0 *[ index - cnt2 + 1 ];
      F1tensors[ index ][ cnt2 ] = new CTensorF1 *[ index - cnt2 + 1 ];
      for ( int cnt3 = 0; cnt3 < ( index - cnt2 + 1 ); cnt3++ ) {
         const int Iprod                    = Irreps::directProd( denBK->gIrrep( index - cnt2 - cnt3 ), denBK->gIrrep( index - cnt3 ) );
         F0tensors[ index ][ cnt2 ][ cnt3 ] = new CTensorF0( index + 1, Iprod, true, denBK, denBK );
         F1tensors[ index ][ cnt2 ][ cnt3 ] = new CTensorF1( index + 1, Iprod, true, denBK, denBK );
      }
   }

   // Otensors :
   Otensors[ index ] = new CTensorO( index + 1, true, denBK, denBK );
}

void CheMPS2::COneDM::deleteAllBoundaryOperators() {

   for ( int cnt = 0; cnt < L; cnt++ ) {
      if ( isAllocated[ cnt ] == 1 ) { deleteTensors( cnt ); }
      isAllocated[ cnt ] = 0;
   }
}

void CheMPS2::COneDM::deleteTensors( const int index ) {

   const int Nbound = index + 1;
   const int Cbound = L - 1 - index;

   //Ltensors : all processes own all Ltensors
   for ( int cnt2 = 0; cnt2 < Nbound; cnt2++ ) {
      delete Ltensors[ index ][ cnt2 ];
   }
   delete[] Ltensors[ index ];

   //Two-operator tensors : certain processes own certain two-operator tensors
   for ( int cnt2 = 0; cnt2 < Nbound; cnt2++ ) {
      for ( int cnt3 = 0; cnt3 < Nbound - cnt2; cnt3++ ) {
         delete F0tensors[ index ][ cnt2 ][ cnt3 ];
         delete F1tensors[ index ][ cnt2 ][ cnt3 ];
      }
      delete[] F0tensors[ index ][ cnt2 ];
      delete[] F1tensors[ index ][ cnt2 ];
   }
   delete[] F0tensors[ index ];
   delete[] F1tensors[ index ];

   // Otensors
   delete Otensors[ index ];
}