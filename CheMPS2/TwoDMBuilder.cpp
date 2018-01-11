/*
   CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum chemistry
   Copyright (C) 2013-2017 Sebastian Wouters

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include "TwoDMBuilder.h"
#include "MPIchemps2.h"
#include "Special.h"
#include "TensorO.h"
#include "TensorT.h"

#include "Heff.h"
#include "Lapack.h"
#include "TwoDM.h"

#include <assert.h>
#include <iostream>
#include <math.h>
#include <omp.h>

using std::cout;
using std::endl;

CheMPS2::TwoDMBuilder::TwoDMBuilder( Problem * ProbIn, TensorT ** MpsIn, SyBookkeeper * bk_in ) {

   Prob  = ProbIn;
   L     = ProbIn->gL();
   denBK = bk_in;

   Ltensors    = new TensorL **[ L - 1 ];
   F0tensors   = new TensorF0 ***[ L - 1 ];
   F1tensors   = new TensorF1 ***[ L - 1 ];
   S0tensors   = new TensorS0 ***[ L - 1 ];
   S1tensors   = new TensorS1 ***[ L - 1 ];
   isAllocated = new int[ L - 1 ];

   MPS = new TensorT *[ L ];
   for ( int index = 0; index < L; index++ ) {
      MPS[ index ] = new TensorT( MpsIn[ index ] );
   }

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( MPS[ index ], MPS[ index + 1 ] );
   }
   left_normalize( MPS[ L - 1 ], NULL );

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      isAllocated[ cnt ] = 0;
   }
   deleteAllBoundaryOperators();
   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafeFirstTime( cnt );
   }

   the2DM = new TwoDM( denBK, Prob );

   for ( int siteindex = L - 1; siteindex >= 0; siteindex-- ) {

      the2DM->FillSite( MPS[ siteindex ], Ltensors, F0tensors, F1tensors, S0tensors, S1tensors );

      if ( siteindex > 0 ) {
         right_normalize( MPS[ siteindex - 1 ], MPS[ siteindex ] );
         updateMovingLeftSafe( siteindex - 1 );
      }
   }

   the2DM->correct_higher_multiplicities();

   std::cout << "   N(N-1)                     = " << denBK->gN() * ( denBK->gN() - 1 ) << std::endl;
   std::cout << "   Double trace of DMRG 2-RDM = " << the2DM->trace() << std::endl;
   std::cout << "   Econst + 0.5 * trace(2DM-A * Ham) = " << the2DM->energy() << std::endl;
}

CheMPS2::TwoDMBuilder::~TwoDMBuilder() {

   deleteAllBoundaryOperators();

   delete[] Ltensors;
   delete[] F0tensors;
   delete[] F1tensors;
   delete[] S0tensors;
   delete[] S1tensors;
   delete[] isAllocated;

   for ( int index = 0; index < L; index++ ) {
      delete MPS[ index ];
   }
   delete[] MPS;

   delete the2DM;
}

void CheMPS2::TwoDMBuilder::updateMovingLeftSafeFirstTime( const int cnt ) {

   if ( isAllocated[ cnt ] == 1 ) {
      deleteTensors( cnt, true );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, false );
      isAllocated[ cnt ] = 2;
   }
   updateMovingLeft( cnt );
}

void CheMPS2::TwoDMBuilder::updateMovingRightSafeFirstTime( const int cnt ) {

   if ( isAllocated[ cnt ] == 2 ) {
      deleteTensors( cnt, false );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, true );
      isAllocated[ cnt ] = 1;
   }
   updateMovingRight( cnt );
}

void CheMPS2::TwoDMBuilder::updateMovingRightSafe( const int cnt ) {

   if ( isAllocated[ cnt ] == 2 ) {
      deleteTensors( cnt, false );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, true );
      isAllocated[ cnt ] = 1;
   }
   updateMovingRight( cnt );
}

void CheMPS2::TwoDMBuilder::updateMovingLeftSafe( const int cnt ) {

   if ( isAllocated[ cnt ] == 1 ) {
      deleteTensors( cnt, true );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, false );
      isAllocated[ cnt ] = 2;
   }
   updateMovingLeft( cnt );
}

void CheMPS2::TwoDMBuilder::updateMovingLeft( const int index ) {

   const int dimL = denBK->gMaxDimAtBound( index + 1 );
   const int dimR = denBK->gMaxDimAtBound( index + 2 );

#pragma omp parallel
   {

      double * workmem = new double[ dimL * dimR ];

// Ltensors : all processes own all Ltensors
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( cnt2 == 0 ) {
            Ltensors[ index ][ cnt2 ]->create( MPS[ index + 1 ] );
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index + 1 ][ cnt2 - 1 ], MPS[ index + 1 ], MPS[ index + 1 ], workmem );
         }
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      const int k1          = L - 1 - index;
      const int upperbound1 = ( k1 * ( k1 + 1 ) ) / 2;
      int result[ 2 ];
      // After this parallel region, WAIT because F0,F1,S0,S1[ index ][ cnt2 ][ cnt3 == 0 ] is required for the complementary operators
#pragma omp for schedule( static )
      for ( int global = 0; global < upperbound1; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2 = k1 - 1 - result[ 1 ];
         const int cnt3 = result[ 0 ];
         if ( cnt3 == 0 ) { // Every MPI process owns the Operator[ index ][ cnt2 ][ cnt3==0 ]
            if ( cnt2 == 0 ) {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPS[ index + 1 ] );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPS[ index + 1 ] );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPS[ index + 1 ] );
               //S1[index][0] doesn't exist
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPS[ index + 1 ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPS[ index + 1 ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPS[ index + 1 ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPS[ index + 1 ], workmem );
            }
         } else {

            {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index + 1 ], MPS[ index + 1 ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index + 1 ], MPS[ index + 1 ], workmem );
            }

            {
               S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index + 1 ], MPS[ index + 1 ], workmem );
               if ( cnt2 > 0 ) { S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index + 1 ], MPS[ index + 1 ], workmem ); }
            }
         }
      }

      delete[] workmem;
   }
}

void CheMPS2::TwoDMBuilder::updateMovingRight( const int index ) {

   const int dimL = denBK->gMaxDimAtBound( index );
   const int dimR = denBK->gMaxDimAtBound( index + 1 );

#pragma omp parallel
   {
      double * workmem = new double[ dimL * dimR ];

//Ltensors : all processes own all Ltensors
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( cnt2 == 0 ) {
            Ltensors[ index ][ cnt2 ]->create( MPS[ index ] );
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index - 1 ][ cnt2 - 1 ], MPS[ index ], MPS[ index ], workmem );
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
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPS[ index ] );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPS[ index ] );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPS[ index ] );
               // S1[ index ][ 0 ][ cnt3 ] doesn't exist
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPS[ index ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPS[ index ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPS[ index ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPS[ index ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index ], MPS[ index ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index ], MPS[ index ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index ], MPS[ index ], workmem );
            if ( cnt2 > 0 ) { S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPS[ index ], MPS[ index ], workmem ); }
         }
      }

      delete[] workmem;
   }
}

void CheMPS2::TwoDMBuilder::allocateTensors( const int index, const bool movingRight ) {

   if ( movingRight ) {

      // Ltensors : all processes own all Ltensors
      // To right: Ltens[cnt][cnt2] = operator on site cnt-cnt2; at boundary cnt+1
      Ltensors[ index ] = new TensorL *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Ltensors[ index ][ cnt2 ] = new TensorL( index + 1, denBK->gIrrep( index - cnt2 ), movingRight, denBK, denBK );
      }

      //Two-operator tensors : certain processes own certain two-operator tensors
      //To right: F0tens[cnt][cnt2][cnt3] = operators on sites cnt-cnt3-cnt2 and cnt-cnt3; at boundary cnt+1
      F0tensors[ index ] = new TensorF0 **[ index + 1 ];
      F1tensors[ index ] = new TensorF1 **[ index + 1 ];
      S0tensors[ index ] = new TensorS0 **[ index + 1 ];
      S1tensors[ index ] = new TensorS1 **[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < ( index + 1 ); cnt2++ ) {
         F0tensors[ index ][ cnt2 ] = new TensorF0 *[ index - cnt2 + 1 ];
         F1tensors[ index ][ cnt2 ] = new TensorF1 *[ index - cnt2 + 1 ];
         S0tensors[ index ][ cnt2 ] = new TensorS0 *[ index - cnt2 + 1 ];
         if ( cnt2 > 0 ) { S1tensors[ index ][ cnt2 ] = new TensorS1 *[ index - cnt2 + 1 ]; }
         for ( int cnt3 = 0; cnt3 < ( index - cnt2 + 1 ); cnt3++ ) {
            const int Iprod                    = Irreps::directProd( denBK->gIrrep( index - cnt2 - cnt3 ), denBK->gIrrep( index - cnt3 ) );
            F0tensors[ index ][ cnt2 ][ cnt3 ] = new TensorF0( index + 1, Iprod, movingRight, denBK );
            F1tensors[ index ][ cnt2 ][ cnt3 ] = new TensorF1( index + 1, Iprod, movingRight, denBK );
            S0tensors[ index ][ cnt2 ][ cnt3 ] = new TensorS0( index + 1, Iprod, movingRight, denBK );
            if ( cnt2 > 0 ) { S1tensors[ index ][ cnt2 ][ cnt3 ] = new TensorS1( index + 1, Iprod, movingRight, denBK ); }
         }
      }

   } else {

      // Ltensors : all processes own all Ltensors
      // To left: Ltens[cnt][cnt2] = operator on site cnt+1+cnt2; at boundary cnt+1
      Ltensors[ index ] = new TensorL *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Ltensors[ index ][ cnt2 ] = new TensorL( index + 1, denBK->gIrrep( index + 1 + cnt2 ), movingRight, denBK, denBK );
      }

      //Two-operator tensors : certain processes own certain two-operator tensors
      //To left: F0tens[cnt][cnt2][cnt3] = operators on sites cnt+1+cnt3 and cnt+1+cnt3+cnt2; at boundary cnt+1
      F0tensors[ index ] = new TensorF0 **[ L - 1 - index ];
      F1tensors[ index ] = new TensorF1 **[ L - 1 - index ];
      S0tensors[ index ] = new TensorS0 **[ L - 1 - index ];
      S1tensors[ index ] = new TensorS1 **[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         F0tensors[ index ][ cnt2 ] = new TensorF0 *[ L - 1 - index - cnt2 ];
         F1tensors[ index ][ cnt2 ] = new TensorF1 *[ L - 1 - index - cnt2 ];
         S0tensors[ index ][ cnt2 ] = new TensorS0 *[ L - 1 - index - cnt2 ];
         if ( cnt2 > 0 ) { S1tensors[ index ][ cnt2 ] = new TensorS1 *[ L - 1 - index - cnt2 ]; }
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Iprod                    = Irreps::directProd( denBK->gIrrep( index + 1 + cnt3 ), denBK->gIrrep( index + 1 + cnt2 + cnt3 ) );
            F0tensors[ index ][ cnt2 ][ cnt3 ] = new TensorF0( index + 1, Iprod, movingRight, denBK );
            F1tensors[ index ][ cnt2 ][ cnt3 ] = new TensorF1( index + 1, Iprod, movingRight, denBK );
            S0tensors[ index ][ cnt2 ][ cnt3 ] = new TensorS0( index + 1, Iprod, movingRight, denBK );
            if ( cnt2 > 0 ) { S1tensors[ index ][ cnt2 ][ cnt3 ] = new TensorS1( index + 1, Iprod, movingRight, denBK ); }
         }
      }
   }
}

void CheMPS2::TwoDMBuilder::deleteAllBoundaryOperators() {

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      if ( isAllocated[ cnt ] == 1 ) { deleteTensors( cnt, true ); }
      if ( isAllocated[ cnt ] == 2 ) { deleteTensors( cnt, false ); }
      isAllocated[ cnt ] = 0;
   }
}

void CheMPS2::TwoDMBuilder::deleteTensors( const int index, const bool movingRight ) {

   const int Nbound = movingRight ? index + 1 : L - 1 - index;
   const int Cbound = movingRight ? L - 1 - index : index + 1;

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
         delete S0tensors[ index ][ cnt2 ][ cnt3 ];
         if ( cnt2 > 0 ) { delete S1tensors[ index ][ cnt2 ][ cnt3 ]; }
      }
      delete[] F0tensors[ index ][ cnt2 ];
      delete[] F1tensors[ index ][ cnt2 ];
      delete[] S0tensors[ index ][ cnt2 ];
      if ( cnt2 > 0 ) { delete[] S1tensors[ index ][ cnt2 ]; }
   }
   delete[] F0tensors[ index ];
   delete[] F1tensors[ index ];
   delete[] S0tensors[ index ];
   delete[] S1tensors[ index ];
}

void CheMPS2::TwoDMBuilder::left_normalize( TensorT * left_mps, TensorT * right_mps ) const {

#ifdef CHEPsi2_MPI_COMPILATION
   const bool am_i_master = ( MPIchemps2::mpi_rank() == MPI_CHEPsi2_MASTER );
#else
   const bool am_i_master = true;
#endif

   if ( am_i_master ) {
      const int siteindex        = left_mps->gIndex();
      const SyBookkeeper * theBK = left_mps->gBK();
      // (J,N,I) = (0,0,0) and (moving_right, prime_last, jw_phase) = (true, true, false)
      TensorOperator * temp = new TensorOperator( siteindex + 1, 0, 0, 0, true, true, false, theBK, theBK );
      left_mps->QR( temp );
      if ( right_mps != NULL ) { right_mps->LeftMultiply( temp ); }
      delete temp;
   }
#ifdef CHEPsi2_MPI_COMPILATION
   MPIchemps2::broadcast_tensor( left_mps, MPI_CHEPsi2_MASTER );
   if ( right_mps != NULL ) { MPIchemps2::broadcast_tensor( right_mps, MPI_CHEPsi2_MASTER ); }
#endif
}

void CheMPS2::TwoDMBuilder::right_normalize( TensorT * left_mps, TensorT * right_mps ) const {

#ifdef CHEPsi2_MPI_COMPILATION
   const bool am_i_master = ( MPIchemps2::mpi_rank() == MPI_CHEPsi2_MASTER );
#else
   const bool am_i_master = true;
#endif

   if ( am_i_master ) {
      const int siteindex        = right_mps->gIndex();
      const SyBookkeeper * theBK = right_mps->gBK();
      // (J,N,I) = (0,0,0) and (moving_right, prime_last, jw_phase) = (true, true, false)
      TensorOperator * temp = new TensorOperator( siteindex, 0, 0, 0, true, true, false, theBK, theBK );
      right_mps->LQ( temp );
      if ( left_mps != NULL ) { left_mps->RightMultiply( temp ); }
      delete temp;
   }
#ifdef CHEPsi2_MPI_COMPILATION
   MPIchemps2::broadcast_tensor( right_mps, MPI_CHEPsi2_MASTER );
   if ( left_mps != NULL ) { MPIchemps2::broadcast_tensor( left_mps, MPI_CHEPsi2_MASTER ); }
#endif
}