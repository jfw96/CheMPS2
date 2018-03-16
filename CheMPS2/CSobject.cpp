
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

// #include "Lapack.h"
#include "CSobject.h"
#include "CTensorT.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Special.h"
#include "SyBookkeeper.h"
#include "Wigner.h"

using std::max;
using std::min;

CheMPS2::CSobject::CSobject( const int index, CheMPS2::SyBookkeeper * denBK )
    : index( index ), denBK( denBK ), Ilocal1( denBK->gIrrep( index ) ), Ilocal2( denBK->gIrrep( index + 1 ) ) {
   nKappa = 0;

   for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( index, NL ); TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               for ( int N1 = 0; N1 <= 2; N1++ ) {
                  for ( int N2 = 0; N2 <= 2; N2++ ) {
                     const int NR      = NL + N1 + N2;
                     const int IM      = ( ( N1 == 1 ) ? Irreps::directProd( IL, Ilocal1 ) : IL );
                     const int IR      = ( ( N2 == 1 ) ? Irreps::directProd( IM, Ilocal2 ) : IM );
                     const int TwoJmin = ( N1 + N2 ) % 2;
                     const int TwoJmax = ( ( ( N1 == 1 ) && ( N2 == 1 ) ) ? 2 : TwoJmin );
                     for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {
                        for ( int TwoSR = TwoSL - TwoJ; TwoSR <= TwoSL + TwoJ; TwoSR += 2 ) {
                           if ( TwoSR >= 0 ) {
                              const int dimR = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );
                              if ( dimR > 0 ) {
                                 nKappa++;
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   sectorNL         = new int[ nKappa ];
   sectorTwoSL      = new int[ nKappa ];
   sectorIL         = new int[ nKappa ];
   sectorN1         = new int[ nKappa ];
   sectorN2         = new int[ nKappa ];
   sectorTwoJ       = new int[ nKappa ];
   sectorNR         = new int[ nKappa ];
   sectorTwoSR      = new int[ nKappa ];
   sectorIR         = new int[ nKappa ];
   kappa2index      = new int[ nKappa + 1 ];
   kappa2index[ 0 ] = 0;

   nKappa = 0;

   for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( index, NL ); TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               for ( int N1 = 0; N1 <= 2; N1++ ) {
                  for ( int N2 = 0; N2 <= 2; N2++ ) {
                     const int NR      = NL + N1 + N2;
                     const int IM      = ( ( N1 == 1 ) ? Irreps::directProd( IL, Ilocal1 ) : IL );
                     const int IR      = ( ( N2 == 1 ) ? Irreps::directProd( IM, Ilocal2 ) : IM );
                     const int TwoJmin = ( N1 + N2 ) % 2;
                     const int TwoJmax = ( ( ( N1 == 1 ) && ( N2 == 1 ) ) ? 2 : TwoJmin );
                     for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {
                        for ( int TwoSR = TwoSL - TwoJ; TwoSR <= TwoSL + TwoJ; TwoSR += 2 ) {
                           if ( TwoSR >= 0 ) {
                              const int dimR = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );
                              if ( dimR > 0 ) {
                                 sectorNL[ nKappa ]    = NL;
                                 sectorTwoSL[ nKappa ] = TwoSL;
                                 sectorIL[ nKappa ]    = IL;
                                 sectorN1[ nKappa ]    = N1;
                                 sectorN2[ nKappa ]    = N2;
                                 sectorTwoJ[ nKappa ]  = TwoJ;
                                 sectorNR[ nKappa ]    = NR;
                                 sectorTwoSR[ nKappa ] = TwoSR;
                                 sectorIR[ nKappa ]    = IR;
                                 nKappa++;
                                 kappa2index[ nKappa ] = kappa2index[ nKappa - 1 ] + dimL * dimR;
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   storage = new dcomplex[ kappa2index[ nKappa ] ];

   reorder = new int[ nKappa ];
   for ( int cnt = 0; cnt < nKappa; cnt++ ) {
      reorder[ cnt ] = cnt;
   }
   bool sorted = false;
   while ( sorted == false ) { // Bubble sort so that blocksize(reorder[i]) >= blocksize(reorder[i+1]), with blocksize(k) = kappa2index[k+1]-kappa2index[k]
      sorted = true;
      for ( int cnt = 0; cnt < nKappa - 1; cnt++ ) {
         const int index1 = reorder[ cnt ];
         const int index2 = reorder[ cnt + 1 ];
         const int size1  = kappa2index[ index1 + 1 ] - kappa2index[ index1 ];
         const int size2  = kappa2index[ index2 + 1 ] - kappa2index[ index2 ];
         if ( size1 < size2 ) {
            sorted             = false;
            reorder[ cnt ]     = index2;
            reorder[ cnt + 1 ] = index1;
         }
      }
   }
}

CheMPS2::CSobject::CSobject( const CSobject * cpy ) : index( cpy->gIndex() ), denBK( cpy->gBK_non_constant() ), Ilocal1( denBK->gIrrep( index ) ), Ilocal2( denBK->gIrrep( index + 1 ) ) {
   nKappa = cpy->gNKappa();

   sectorNL    = new int[ nKappa ];
   sectorTwoSL = new int[ nKappa ];
   sectorIL    = new int[ nKappa ];
   sectorN1    = new int[ nKappa ];
   sectorN2    = new int[ nKappa ];
   sectorTwoJ  = new int[ nKappa ];
   sectorNR    = new int[ nKappa ];
   sectorTwoSR = new int[ nKappa ];
   sectorIR    = new int[ nKappa ];
   kappa2index = new int[ nKappa + 1 ];

   std::copy( cpy->sectorNL, cpy->sectorNL + nKappa, sectorNL );
   std::copy( cpy->sectorTwoSL, cpy->sectorTwoSL + nKappa, sectorTwoSL );
   std::copy( cpy->sectorIL, cpy->sectorIL + nKappa, sectorIL );
   std::copy( cpy->sectorN1, cpy->sectorN1 + nKappa, sectorN1 );
   std::copy( cpy->sectorN2, cpy->sectorN2 + nKappa, sectorN2 );
   std::copy( cpy->sectorTwoJ, cpy->sectorTwoJ + nKappa, sectorTwoJ );
   std::copy( cpy->sectorNR, cpy->sectorNR + nKappa, sectorNR );
   std::copy( cpy->sectorTwoSR, cpy->sectorTwoSR + nKappa, sectorTwoSR );
   std::copy( cpy->sectorIR, cpy->sectorIR + nKappa, sectorIR );
   std::copy( cpy->kappa2index, cpy->kappa2index + nKappa + 1, kappa2index );
   storage = new dcomplex[ kappa2index[ nKappa ] ];
   std::copy( cpy->storage, cpy->storage + kappa2index[ nKappa ], storage );
   reorder = new int[ nKappa ];
   std::copy( cpy->reorder, cpy->reorder + nKappa, reorder );
}

CheMPS2::CSobject::~CSobject() {
   delete[] sectorNL;
   delete[] sectorTwoSL;
   delete[] sectorIL;
   delete[] sectorN1;
   delete[] sectorN2;
   delete[] sectorTwoJ;
   delete[] sectorNR;
   delete[] sectorTwoSR;
   delete[] sectorIR;
   delete[] kappa2index;
   delete[] storage;
   delete[] reorder;
}

void CheMPS2::CSobject::Clear() {
   for ( int i = 0; i < kappa2index[ nKappa ]; i++ ) {
      storage[ i ] = 0.0;
   }
}

int CheMPS2::CSobject::gNKappa() const { return nKappa; }

dcomplex * CheMPS2::CSobject::gStorage() { return storage; }

int CheMPS2::CSobject::gReorder( const int ikappa ) const {
   return reorder[ ikappa ];
}

int CheMPS2::CSobject::gKappa( const int NL, const int TwoSL, const int IL,
                               const int N1, const int N2, const int TwoJ,
                               const int NR, const int TwoSR, const int IR ) const {
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      if ( ( sectorNL[ ikappa ] == NL ) && ( sectorTwoSL[ ikappa ] == TwoSL ) &&
           ( sectorIL[ ikappa ] == IL ) && ( sectorN1[ ikappa ] == N1 ) &&
           ( sectorN2[ ikappa ] == N2 ) && ( sectorTwoJ[ ikappa ] == TwoJ ) &&
           ( sectorNR[ ikappa ] == NR ) && ( sectorTwoSR[ ikappa ] == TwoSR ) &&
           ( sectorIR[ ikappa ] == IR ) ) {
         return ikappa;
      }
   }

   return -1;
}

int CheMPS2::CSobject::gKappa2index( const int kappa ) const {
   return kappa2index[ kappa ];
}

dcomplex * CheMPS2::CSobject::gStorage( const int NL, const int TwoSL, const int IL,
                                        const int N1, const int N2, const int TwoJ,
                                        const int NR, const int TwoSR,
                                        const int IR ) {
   const int kappa = gKappa( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
   if ( kappa == -1 ) {
      return NULL;
   }
   return storage + kappa2index[ kappa ];
}

int CheMPS2::CSobject::gIndex() const { return index; }

int CheMPS2::CSobject::gNL( const int ikappa ) const { return sectorNL[ ikappa ]; }

int CheMPS2::CSobject::gTwoSL( const int ikappa ) const {
   return sectorTwoSL[ ikappa ];
}

int CheMPS2::CSobject::gIL( const int ikappa ) const { return sectorIL[ ikappa ]; }

int CheMPS2::CSobject::gN1( const int ikappa ) const { return sectorN1[ ikappa ]; }

int CheMPS2::CSobject::gN2( const int ikappa ) const { return sectorN2[ ikappa ]; }

int CheMPS2::CSobject::gTwoJ( const int ikappa ) const {
   return sectorTwoJ[ ikappa ];
}

int CheMPS2::CSobject::gNR( const int ikappa ) const { return sectorNR[ ikappa ]; }

int CheMPS2::CSobject::gTwoSR( const int ikappa ) const {
   return sectorTwoSR[ ikappa ];
}

int CheMPS2::CSobject::gIR( const int ikappa ) const { return sectorIR[ ikappa ]; }

void CheMPS2::CSobject::Join( CTensorT * Tleft, CTensorT * Tright ) {

   char cotrans = 'C';
   char notrans = 'N';

#pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NL    = sectorNL[ ikappa ];
      const int TwoSL = sectorTwoSL[ ikappa ];
      const int IL    = sectorIL[ ikappa ];

      const int NR    = sectorNR[ ikappa ];
      const int TwoSR = sectorTwoSR[ ikappa ];
      const int IR    = sectorIR[ ikappa ];

      const int TwoJ  = sectorTwoJ[ ikappa ];
      const int N1    = sectorN1[ ikappa ];
      const int N2    = sectorN2[ ikappa ];
      const int TwoS1 = ( ( N1 == 1 ) ? 1 : 0 );
      const int TwoS2 = ( ( N2 == 1 ) ? 1 : 0 );
      const int fase  = Special::phase( TwoSL + TwoSR + TwoS1 + TwoS2 );

      // Clear block
      int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );     // dimL > 0, checked at creation
      int dimR = denBK->gCurrentDim( index + 2, NR, TwoSR, IR ); // dimR > 0, checked at creation

      dcomplex * block_s = storage + kappa2index[ ikappa ];
      for ( int cnt = 0; cnt < dimL * dimR; cnt++ ) {
         block_s[ cnt ] = 0.0;
      }

      // Central symmetry sectors
      const int NM         = NL + N1;
      const int IM         = ( ( TwoS1 == 1 ) ? Irreps::directProd( IL, Ilocal1 ) : IL );
      const int TwoJMlower = max( abs( TwoSL - TwoS1 ), abs( TwoSR - TwoS2 ) );
      const int TwoJMupper = min( ( TwoSL + TwoS1 ), ( TwoSR + TwoS2 ) );
      for ( int TwoJM = TwoJMlower; TwoJM <= TwoJMupper; TwoJM += 2 ) {
         int dimM = denBK->gCurrentDim( index + 1, NM, TwoJM, IM );
         if ( dimM > 0 ) {
            dcomplex * block_left  = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
            dcomplex * block_right = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );
            dcomplex prefactor     = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) *
                                 Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
            dcomplex add = 1.0;
            zgemm_( &notrans, &notrans, &dimL, &dimR, &dimM, &prefactor, block_left,
                    &dimL, block_right, &dimM, &add, block_s, &dimL );
         }
      }
   }
}

void CheMPS2::CSobject::Join( CTensorO * Oleft, CTensorT * Tleft, CTensorT * Tright, CTensorO * Oright ) {

   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == denBK->gL() - 2 ) ? true : false;

   const int DIM_L = std::max( denBK->gMaxDimAtBound( index ), Tleft->gBK()->gMaxDimAtBound( index ) );
   const int DIM_M = std::max( denBK->gMaxDimAtBound( index + 1 ), Tleft->gBK()->gMaxDimAtBound( index + 1 ) );
   const int DIM_R = std::max( denBK->gMaxDimAtBound( index + 2 ), Tleft->gBK()->gMaxDimAtBound( index + 2 ) );

   char cotrans = 'C';
   char notrans = 'N';

#pragma omp parallel
   {
      dcomplex * tempA = new dcomplex[ DIM_L * DIM_M ];
      dcomplex * tempB = new dcomplex[ DIM_M * DIM_R ];
      dcomplex * tempC = new dcomplex[ DIM_L * DIM_R ];

#pragma omp for schedule( dynamic )
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         const int NL    = sectorNL[ ikappa ];
         const int TwoSL = sectorTwoSL[ ikappa ];
         const int IL    = sectorIL[ ikappa ];

         const int NR    = sectorNR[ ikappa ];
         const int TwoSR = sectorTwoSR[ ikappa ];
         const int IR    = sectorIR[ ikappa ];

         const int TwoJ       = sectorTwoJ[ ikappa ];
         const int N1         = sectorN1[ ikappa ];
         const int NM         = NL + N1;
         const int N2         = sectorN2[ ikappa ];
         const int TwoS1      = ( ( N1 == 1 ) ? 1 : 0 );
         const int TwoS2      = ( ( N2 == 1 ) ? 1 : 0 );
         const int TwoJMlower = max( abs( TwoSL - TwoS1 ), abs( TwoSR - TwoS2 ) );
         const int TwoJMupper = min( ( TwoSL + TwoS1 ), ( TwoSR + TwoS2 ) );
         const int IM         = ( ( TwoS1 == 1 ) ? Irreps::directProd( IL, Ilocal1 ) : IL );
         const int fase       = Special::phase( TwoSL + TwoSR + TwoS1 + TwoS2 );

         int dimLU = denBK->gCurrentDim( index, NL, TwoSL, IL );
         int dimRU = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );

         int dimLD = Tleft->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimRD = Tleft->gBK()->gCurrentDim( index + 2, NR, TwoSR, IR );

         dcomplex * block_s = storage + kappa2index[ ikappa ];
         for ( int cnt = 0; cnt < dimLU * dimRU; cnt++ ) {
            block_s[ cnt ] = 0.0;
         }

         for ( int TwoJM = TwoJMlower; TwoJM <= TwoJMupper; TwoJM += 2 ) {
            int dimM = Tleft->gBK()->gCurrentDim( index + 1, NM, TwoJM, IM );
            if ( dimLD > 0 && dimRD > 0 && dimM > 0 ) {

               if ( !atLeft && !atRight ) {

                  dcomplex * overlap_left  = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
                  dcomplex * block_left    = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
                  dcomplex * block_right   = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );
                  dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

                  dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
                  dcomplex add       = 1.0;
                  dcomplex noadd     = 0.0;
                  zgemm_( &notrans, &notrans, &dimLU, &dimM, &dimLD, &prefactor, overlap_left, &dimLU, block_left, &dimLD, &noadd, tempA, &dimLU );

                  prefactor = 1.0;
                  zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimM, &prefactor, tempA, &dimLU, block_right, &dimM, &noadd, tempC, &dimLU );

                  zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, tempC, &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
               }
               if ( !atLeft && atRight ) {

                  dcomplex * overlap_left = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
                  dcomplex * block_left   = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
                  dcomplex * block_right  = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );

                  dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
                  dcomplex add       = 1.0;
                  dcomplex noadd     = 0.0;
                  zgemm_( &notrans, &notrans, &dimLU, &dimM, &dimLD, &prefactor, overlap_left, &dimLU, block_left, &dimLD, &noadd, tempA, &dimLU );

                  prefactor = 1.0;
                  zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimM, &prefactor, tempA, &dimLU, block_right, &dimM, &add, block_s, &dimLU );
               }
               if ( atLeft && !atRight ) {
                  dcomplex * block_left    = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
                  dcomplex * block_right   = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );
                  dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

                  dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
                  dcomplex add       = 1.0;
                  dcomplex noadd     = 0.0;
                  zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimM, &prefactor, block_left, &dimLU, block_right, &dimM, &noadd, tempC, &dimLU );

                  prefactor = 1.0;
                  zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, tempC, &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
               }
            }
         }
      }
      delete[] tempA;
      delete[] tempB;
      delete[] tempC;
   }
}

void CheMPS2::CSobject::Join( CTensorO * Oleft, CSobject * innerS, CTensorO * Oright ) {

   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == denBK->gL() - 2 ) ? true : false;

   const int DIM_L = std::max( denBK->gMaxDimAtBound( index ),
                               innerS->gBK()->gMaxDimAtBound( index ) );
   const int DIM_R = std::max( denBK->gMaxDimAtBound( index + 2 ),
                               innerS->gBK()->gMaxDimAtBound( index + 2 ) );

   char cotrans = 'C';
   char notrans = 'N';

#pragma omp parallel
   {
      dcomplex * temp = new dcomplex[ DIM_L * DIM_R ];

#pragma omp for schedule( dynamic )
      for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
         const int NL    = sectorNL[ ikappa ];
         const int TwoSL = sectorTwoSL[ ikappa ];
         const int IL    = sectorIL[ ikappa ];

         const int NR    = sectorNR[ ikappa ];
         const int TwoSR = sectorTwoSR[ ikappa ];
         const int IR    = sectorIR[ ikappa ];

         const int TwoJ  = sectorTwoJ[ ikappa ];
         const int N1    = sectorN1[ ikappa ];
         const int N2    = sectorN2[ ikappa ];
         const int TwoS1 = ( ( N1 == 1 ) ? 1 : 0 );
         const int TwoS2 = ( ( N2 == 1 ) ? 1 : 0 );

         int dimLU = denBK->gCurrentDim( index, NL, TwoSL, IL );
         int dimRU = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );

         int dimLD = innerS->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimRD = innerS->gBK()->gCurrentDim( index + 2, NR, TwoSR, IR );

         if ( dimLD > 0 && dimRD > 0 ) {
            dcomplex * block_s = storage + kappa2index[ ikappa ];

            if ( !atLeft && !atRight ) {
               dcomplex * overlap_left  = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
               dcomplex * block_middle  = innerS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
               dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

               dcomplex prefactor = 1.0;
               dcomplex add       = 0.0;
               zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &prefactor, overlap_left,
                       &dimLU, block_middle, &dimLD, &add, temp, &dimLU );

               zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, temp,
                       &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
            }
            if ( !atLeft && atRight ) {
               dcomplex * overlap_left = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
               dcomplex * block_middle = innerS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );

               dcomplex prefactor = 1.0;
               dcomplex add       = 0.0;

               zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &prefactor, overlap_left,
                       &dimLU, block_middle, &dimLD, &add, block_s, &dimLU );
            }
            if ( atLeft && !atRight ) {
               dcomplex * block_middle  = innerS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
               dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

               dcomplex prefactor = 1.0;
               dcomplex add       = 0.0;

               zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &prefactor, block_middle,
                       &dimLU, overlap_right, &dimRU, &add, block_s, &dimLD );
            }
         }
      }
      delete[] temp;
   }
}

void CheMPS2::CSobject::Add( dcomplex alpha, CSobject * to_add ) {
   assert( index == to_add->gIndex() );
   assert( nKappa == to_add->gNKappa() );
   assert( kappa2index[ nKappa ] == to_add->gKappa2index( to_add->gNKappa() ) );

   int inc = 1;
   zaxpy_( kappa2index + nKappa, &alpha, to_add->gStorage(), &inc, storage, &inc );
}

void CheMPS2::CSobject::Multiply( dcomplex alpha ) {
   int inc = 1;
   zscal_( kappa2index + nKappa, &alpha, storage, &inc );
}

double CheMPS2::CSobject::Split( CTensorT * Tleft, CTensorT * Tright, const int virtualdimensionD,
                                 const double cut_off, const bool movingright, const bool change ) {
   // Get the number of central sectors
   int nCenterSectors = 0;
   for ( int NM = denBK->gNmin( index + 1 ); NM <= denBK->gNmax( index + 1 ); NM++ ) {
      for ( int TwoSM = denBK->gTwoSmin( index + 1, NM ); TwoSM <= denBK->gTwoSmax( index + 1, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < denBK->getNumberOfIrreps(); IM++ ) {
            const int dimM = denBK->gFCIdim( index + 1, NM, TwoSM, IM ); // FCIdim !! Whether possible hence.
            if ( dimM > 0 ) {
               nCenterSectors++;
            }
         }
      }
   }

   // Get the labels of the central sectors
   int * SplitSectNM    = new int[ nCenterSectors ];
   int * SplitSectTwoJM = new int[ nCenterSectors ];
   int * SplitSectIM    = new int[ nCenterSectors ];
   nCenterSectors       = 0;
   for ( int NM = denBK->gNmin( index + 1 ); NM <= denBK->gNmax( index + 1 ); NM++ ) {
      for ( int TwoSM = denBK->gTwoSmin( index + 1, NM ); TwoSM <= denBK->gTwoSmax( index + 1, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < denBK->getNumberOfIrreps(); IM++ ) {
            const int dimM = denBK->gFCIdim( index + 1, NM, TwoSM, IM ); // FCIdim !! Whether possible hence.
            if ( dimM > 0 ) {
               SplitSectNM[ nCenterSectors ]    = NM;
               SplitSectTwoJM[ nCenterSectors ] = TwoSM;
               SplitSectIM[ nCenterSectors ]    = IM;
               nCenterSectors++;
            }
         }
      }
   }

   // Only MPI_CHEMPS2_MASTER performs SVD --> Allocate memory
   double ** Lambdas = NULL;
   dcomplex ** Us    = NULL;
   dcomplex ** VTs   = NULL;
   int * CenterDims  = NULL;
   int * DimLtotal   = NULL;
   int * DimRtotal   = NULL;

   Lambdas    = new double *[ nCenterSectors ];
   Us         = new dcomplex *[ nCenterSectors ];
   VTs        = new dcomplex *[ nCenterSectors ];
   CenterDims = new int[ nCenterSectors ];
   DimLtotal  = new int[ nCenterSectors ];
   DimRtotal  = new int[ nCenterSectors ];

// PARALLEL
#pragma omp parallel for schedule( dynamic )
   for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
      // Determine left and right dimensions contributing to the center block
      // iCenter
      DimLtotal[ iCenter ] = 0;
      for ( int NL = SplitSectNM[ iCenter ] - 2; NL <= SplitSectNM[ iCenter ]; NL++ ) {
         const int TwoS1 = ( ( NL + 1 == SplitSectNM[ iCenter ] ) ? 1 : 0 );
         for ( int TwoSL = SplitSectTwoJM[ iCenter ] - TwoS1; TwoSL <= SplitSectTwoJM[ iCenter ] + TwoS1; TwoSL += 2 ) {
            if ( TwoSL >= 0 ) {
               const int IL   = ( ( TwoS1 == 1 ) ? Irreps::directProd( Ilocal1, SplitSectIM[ iCenter ] ) : SplitSectIM[ iCenter ] );
               const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
               if ( dimL > 0 ) {
                  DimLtotal[ iCenter ] += dimL;
               }
            }
         }
      }
      DimRtotal[ iCenter ] = 0;
      for ( int NR = SplitSectNM[ iCenter ]; NR <= SplitSectNM[ iCenter ] + 2; NR++ ) {
         const int TwoS2 = ( ( NR == SplitSectNM[ iCenter ] + 1 ) ? 1 : 0 );
         for ( int TwoSR = SplitSectTwoJM[ iCenter ] - TwoS2;
               TwoSR <= SplitSectTwoJM[ iCenter ] + TwoS2; TwoSR += 2 ) {
            if ( TwoSR >= 0 ) {
               const int IR   = ( ( TwoS2 == 1 ) ? Irreps::directProd( Ilocal2, SplitSectIM[ iCenter ] ) : SplitSectIM[ iCenter ] );
               const int dimR = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );
               if ( dimR > 0 ) {
                  DimRtotal[ iCenter ] += dimR;
               }
            }
         }
      }
      CenterDims[ iCenter ] = min( DimLtotal[ iCenter ], DimRtotal[ iCenter ] ); // CenterDims contains the min. amount

      // Allocate memory to copy the different parts of the S-object. Use
      // prefactor sqrt((2jR+1)/(2jM+1) * (2jM+1) * (2j+1)) W6J
      // (-1)^(jL+jR+s1+s2) and sum over j.
      if ( CenterDims[ iCenter ] > 0 ) {
         // Only if CenterDims[ iCenter ] exists should you allocate the
         // following three arrays
         Lambdas[ iCenter ] = new double[ CenterDims[ iCenter ] ];
         Us[ iCenter ]      = new dcomplex[ CenterDims[ iCenter ] * DimLtotal[ iCenter ] ];
         VTs[ iCenter ]     = new dcomplex[ CenterDims[ iCenter ] * DimRtotal[ iCenter ] ];

         const int memsize = DimLtotal[ iCenter ] * DimRtotal[ iCenter ];
         dcomplex * mem    = new dcomplex[ memsize ];
         for ( int cnt = 0; cnt < memsize; cnt++ ) {
            mem[ cnt ] = 0.0;
         }

         int dimLtotal2 = 0;
         for ( int NL = SplitSectNM[ iCenter ] - 2; NL <= SplitSectNM[ iCenter ]; NL++ ) {
            const int TwoS1 = ( ( NL + 1 == SplitSectNM[ iCenter ] ) ? 1 : 0 );
            for ( int TwoSL = SplitSectTwoJM[ iCenter ] - TwoS1; TwoSL <= SplitSectTwoJM[ iCenter ] + TwoS1; TwoSL += 2 ) {
               if ( TwoSL >= 0 ) {
                  const int IL   = ( ( TwoS1 == 1 ) ? Irreps::directProd( Ilocal1, SplitSectIM[ iCenter ] ) : SplitSectIM[ iCenter ] );
                  const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
                  if ( dimL > 0 ) {
                     int dimRtotal2 = 0;
                     for ( int NR = SplitSectNM[ iCenter ]; NR <= SplitSectNM[ iCenter ] + 2; NR++ ) {
                        const int TwoS2 = ( ( NR == SplitSectNM[ iCenter ] + 1 ) ? 1 : 0 );
                        for ( int TwoSR = SplitSectTwoJM[ iCenter ] - TwoS2; TwoSR <= SplitSectTwoJM[ iCenter ] + TwoS2; TwoSR += 2 ) {
                           if ( TwoSR >= 0 ) {
                              const int IR   = ( ( TwoS2 == 1 ) ? Irreps::directProd( Ilocal2, SplitSectIM[ iCenter ] ) : SplitSectIM[ iCenter ] );
                              const int dimR = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );
                              if ( dimR > 0 ) {
                                 // Loop over contributing TwoJ's
                                 const int fase    = Special::phase( TwoSL + TwoSR + TwoS1 + TwoS2 );
                                 const int TwoJmin = max( abs( TwoSR - TwoSL ), abs( TwoS2 - TwoS1 ) );
                                 const int TwoJmax = min( TwoS1 + TwoS2, TwoSL + TwoSR );
                                 for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {
                                    // Calc prefactor
                                    const dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoSR + 1 ) ) *
                                                               Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, SplitSectTwoJM[ iCenter ] );

                                    // Add them to mem --> += because several TwoJ
                                    dcomplex * Block = gStorage( NL, TwoSL, IL, SplitSectNM[ iCenter ] - NL, NR - SplitSectNM[ iCenter ], TwoJ, NR, TwoSR, IR );
                                    for ( int l = 0; l < dimL; l++ ) {
                                       for ( int r = 0; r < dimR; r++ ) {
                                          mem[ dimLtotal2 + l + DimLtotal[ iCenter ] * ( dimRtotal2 + r ) ] += prefactor * Block[ l + dimL * r ];
                                       }
                                    }
                                 }
                                 dimRtotal2 += dimR;
                              }
                           }
                        }
                     }
                     dimLtotal2 += dimL;
                  }
               }
            }
         }

         // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
         // central symmetry
         char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
         int lwork       = 3 * CenterDims[ iCenter ] + max( max( DimLtotal[ iCenter ], DimRtotal[ iCenter ] ), 4 * CenterDims[ iCenter ] * ( CenterDims[ iCenter ] + 1 ) );
         double * rwork  = new double[ 5 * CenterDims[ iCenter ] * CenterDims[ iCenter ] + 7 * CenterDims[ iCenter ] ];
         dcomplex * work = new dcomplex[ lwork ];
         int * iwork     = new int[ 8 * CenterDims[ iCenter ] ];
         int info;

         // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
         zgesdd_( &jobz, DimLtotal + iCenter, DimRtotal + iCenter, mem, DimLtotal + iCenter,
                  Lambdas[ iCenter ], Us[ iCenter ], DimLtotal + iCenter, VTs[ iCenter ],
                  CenterDims + iCenter, work, &lwork, rwork, iwork, &info );

         delete[] work;
         delete[] rwork;
         delete[] iwork;
         delete[] mem;
      }
   }

   double discardedWeight = 0.0; // Only if change==true; will the discardedWeight be meaningful and different from zero.
   int updateSectors      = 0;
   int * NewDims          = NULL;

   // If change: determine new virtual dimensions.
   if ( change ) {
      NewDims = new int[ nCenterSectors ];
      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
         NewDims[ iCenter ] = CenterDims[ iCenter ];
         totalDimSVD += NewDims[ iCenter ];
      }

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
         if ( NewDims[ iCenter ] > 0 ) {
            dcopy_( NewDims + iCenter, Lambdas[ iCenter ], &inc, values + totalDimSVD, &inc );
            totalDimSVD += NewDims[ iCenter ];
         }
      }

      // Sort them in decreasing order
         char ID = 'D';
         int info;
         dlasrt_( &ID, &totalDimSVD, values, &info ); // Quicksort

         // The D+1'th value becomes the lower bound Schmidt value. Every value smaller than or equal to the D+1'th value is thrown out (hence Dactual <= Ddesired).
         const double lowerBound = values[ virtualdimensionD ];
         for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ){
            for ( int cnt = 0; cnt < NewDims[ iCenter ]; cnt++ ){
               if ( Lambdas[ iCenter ][ cnt ] <= lowerBound ){ NewDims[ iCenter ] = cnt; }
            }
         }

         // Discarded weight
         double totalSum = 0.0;
         double discardedSum = 0.0;
         for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ){
            for ( int iLocal = 0; iLocal < CenterDims[ iCenter ]; iLocal++ ){
               double temp = ( SplitSectTwoJM[ iCenter ] + 1 ) * Lambdas[ iCenter ][ iLocal ] * Lambdas[ iCenter ][ iLocal ];
               totalSum += temp;
               if ( Lambdas[ iCenter ][ iLocal ] <= lowerBound ){ discardedSum += temp; }
            }
         }
         discardedWeight = discardedSum / totalSum;

         // Clean-up
         delete [] values;

      // Check if there is a sector which differs
      updateSectors = 0;
      for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
         const int MPSdim = denBK->gCurrentDim( index + 1, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
         if ( NewDims[ iCenter ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }

   if ( updateSectors == 1 ) {
      for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
         denBK->SetDim( index + 1, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ], NewDims[ iCenter ] );
      }
      Tleft->Reset();
      Tright->Reset();
   }

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

// Copy first dimM per central symmetry sector to the relevant parts
#pragma omp parallel for schedule( dynamic )
   for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
      const int dimM = denBK->gCurrentDim( index + 1, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
      if ( dimM > 0 ) {
         // U-part: copy
         int dimLtotal2 = 0;
         for ( int NL = SplitSectNM[ iCenter ] - 2; NL <= SplitSectNM[ iCenter ]; NL++ ) {
            const int TwoS1 = ( ( NL + 1 == SplitSectNM[ iCenter ] ) ? 1 : 0 );
            for ( int TwoSL = SplitSectTwoJM[ iCenter ] - TwoS1; TwoSL <= SplitSectTwoJM[ iCenter ] + TwoS1; TwoSL += 2 ) {
               if ( TwoSL >= 0 ) {
                  const int IL   = ( ( TwoS1 == 1 ) ? Irreps::directProd( Ilocal1, SplitSectIM[ iCenter ] ) : SplitSectIM[ iCenter ] );
                  const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
                  if ( dimL > 0 ) {
                     dcomplex * TleftBlock           = Tleft->gStorage( NL, TwoSL, IL, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
                     const int dimension_limit_right = min( dimM, CenterDims[ iCenter ] );
                     for ( int r = 0; r < dimension_limit_right; r++ ) {
                        const dcomplex factor = ( ( movingright ) ? 1.0 : Lambdas[ iCenter ][ r ] );
                        for ( int l = 0; l < dimL; l++ ) {
                           TleftBlock[ l + dimL * r ] = factor * Us[ iCenter ][ dimLtotal2 + l + DimLtotal[ iCenter ] * r ];
                        }
                     }
                     for ( int r = dimension_limit_right; r < dimM; r++ ) {
                        for ( int l = 0; l < dimL; l++ ) {
                           TleftBlock[ l + dimL * r ] = 0.0;
                        }
                     }
                     dimLtotal2 += dimL;
                  }
               }
            }
         }

         // VT-part: copy
         int dimRtotal2 = 0;
         for ( int NR = SplitSectNM[ iCenter ]; NR <= SplitSectNM[ iCenter ] + 2; NR++ ) {
            const int TwoS2 = ( ( NR == SplitSectNM[ iCenter ] + 1 ) ? 1 : 0 );
            for ( int TwoSR = SplitSectTwoJM[ iCenter ] - TwoS2; TwoSR <= SplitSectTwoJM[ iCenter ] + TwoS2; TwoSR += 2 ) {
               if ( TwoSR >= 0 ) {
                  const int IR   = ( ( TwoS2 == 1 ) ? Irreps::directProd( Ilocal2, SplitSectIM[ iCenter ] )
                                                  : SplitSectIM[ iCenter ] );
                  const int dimR = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );
                  if ( dimR > 0 ) {
                     dcomplex * TrightBlock         = Tright->gStorage( SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ], NR, TwoSR, IR );
                     const int dimension_limit_left = min( dimM, CenterDims[ iCenter ] );
                     const dcomplex factor_base     = sqrt( ( SplitSectTwoJM[ iCenter ] + 1.0 ) / ( TwoSR + 1 ) );
                     for ( int l = 0; l < dimension_limit_left; l++ ) {
                        const dcomplex factor = factor_base * ( ( movingright ) ? Lambdas[ iCenter ][ l ] : 1.0 );
                        for ( int r = 0; r < dimR; r++ ) {
                           TrightBlock[ l + dimM * r ] = factor * VTs[ iCenter ][ l + CenterDims[ iCenter ] * ( dimRtotal2 + r ) ];
                        }
                     }
                     for ( int r = 0; r < dimR; r++ ) {
                        for ( int l = dimension_limit_left; l < dimM; l++ ) {
                           TrightBlock[ l + dimM * r ] = 0.0;
                        }
                     }
                     dimRtotal2 += dimR;
                  }
               }
            }
         }
      }
   }

   // Clean up
   delete[] SplitSectNM;
   delete[] SplitSectTwoJM;
   delete[] SplitSectIM;
   for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
      if ( CenterDims[ iCenter ] > 0 ) {
         delete[] Us[ iCenter ];
         delete[] Lambdas[ iCenter ];
         delete[] VTs[ iCenter ];
      }
   }
   delete[] Us;
   delete[] Lambdas;
   delete[] VTs;
   delete[] CenterDims;
   delete[] DimLtotal;
   delete[] DimRtotal;

   return discardedWeight;
}

void CheMPS2::CSobject::prog2symm() {
#pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      int dim        = kappa2index[ ikappa + 1 ] - kappa2index[ ikappa ];
      dcomplex alpha = sqrt( sectorTwoSR[ ikappa ] + 1.0 );
      int inc        = 1;
      zscal_( &dim, &alpha, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CSobject::symm2prog() {
#pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      int dim        = kappa2index[ ikappa + 1 ] - kappa2index[ ikappa ];
      dcomplex alpha = 1.0 / sqrt( sectorTwoSR[ ikappa ] + 1.0 );
      int inc        = 1;
      zscal_( &dim, &alpha, storage + kappa2index[ ikappa ], &inc );
   }
}

void CheMPS2::CSobject::addNoise( const double NoiseLevel ) {
   for ( int cnt = 0; cnt < gKappa2index( gNKappa() ); cnt++ ) {
      const dcomplex RN = ( ( double ) rand() ) / RAND_MAX - 0.5;
      gStorage()[ cnt ] += RN * NoiseLevel;
   }
}

// void CheMPS2::CSobject::print() const {
//    std::cout << "############################################################"
//              << std::endl;
//    std::cout << "CSobject with " << nKappa << " symmetry blocks: " << std::endl;

//    for ( int ikappa = 0; ikappa < nKappa; ++ikappa ) {
//       const int NL    = gNL( ikappa );
//       const int TwoSL = gTwoSL( ikappa );
//       const int IL    = gIL( ikappa );

//       const int NR    = gNR( ikappa );
//       const int TwoSR = gTwoSR( ikappa );
//       const int IR    = gIR( ikappa );

//       const int N1   = gN1( ikappa );
//       const int N2   = gN2( ikappa );
//       const int TwoJ = gTwoJ( ikappa );

//       std::cout << "Block number " << ikappa << std::endl;

//       std::cout << "NL:    " << NL << std::endl;
//       std::cout << "TwoSL: " << TwoSL << std::endl;
//       std::cout << "IL:    " << IL << std::endl;
//       std::cout << "N1:    " << N1 << std::endl;
//       std::cout << "N2:    " << N2 << std::endl;
//       std::cout << "TwoJ:  " << TwoJ << std::endl;
//       std::cout << "NR:    " << NR << std::endl;
//       std::cout << "TwoSR: " << TwoSR << std::endl;
//       std::cout << "IR:    " << IR << std::endl;

//       for ( int i = kappa2index[ ikappa ]; i < kappa2index[ ikappa + 1 ]; ++i ) {
//          std::cout << storage[ i ] << std::endl;
//       }
//    }
//    std::cout << "############################################################"
//              << std::endl;
// }
