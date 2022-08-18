
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

// #include "Lapack.h"
#include "CDensityMatrix.h"
#include "CTensorT.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Special.h"
#include "SyBookkeeper.h"
#include "Wigner.h"

using std::max;
using std::min;

CheMPS2::CDensityMatrix::CDensityMatrix( const int index, CheMPS2::SyBookkeeper * denBK )
    : index( index ), denBK( denBK ), Ilocal1( denBK->gIrrep( index ) ) {
   nKappa = 0;

   for ( int NLU = denBK->gNmin( index ); NLU <= denBK->gNmax( index ); NLU++ ) {
      for ( int TwoSLU = denBK->gTwoSmin( index, NLU ); TwoSLU <= denBK->gTwoSmax( index, NLU ); TwoSLU += 2 ) {
         for ( int ILU = 0; ILU < denBK->getNumberOfIrreps(); ILU++ ) {
            const int dimLU = denBK->gCurrentDim( index, NLU, TwoSLU, ILU );
            if ( dimLU > 0 ) {
               for ( int NU = 0; NU <= 2; NU++ ) {
                  const int TwoSU   = ( NU == 1 ) ? 1 : 0;
                  const int IL      = ( ( NU == 1 ) ? Irreps::directProd( ILU, Ilocal1 ) : ILU );
                  const int TwoJmin = abs( TwoSLU - TwoSU );
                  const int TwoJmax = TwoSLU + TwoSU;
                  for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {
                     const int dimMU = denBK->gCurrentDim( index + 1, NLU + NU, TwoJ, IL );
                     if ( dimMU > 0 ) {
                        for ( int NLD = denBK->gNmin( index ); NLD <= denBK->gNmax( index ); NLD++ ) {
                           for ( int TwoSLD = denBK->gTwoSmin( index, NLD ); TwoSLD <= denBK->gTwoSmax( index, NLD ); TwoSLD += 2 ) {
                              for ( int ILD = 0; ILD < denBK->getNumberOfIrreps(); ILD++ ) {
                                 const int dimLD = denBK->gCurrentDim( index, NLD, TwoSLD, ILD );
                                 if ( dimLD > 0 ) {
                                    for ( int ND = 0; ND <= 2; ND++ ) {
                                       const int TwoSD    = ( ND == 1 ) ? 1 : 0;
                                       const int ILT      = ( ( ND == 1 ) ? Irreps::directProd( ILD, Ilocal1 ) : ILD );
                                       const int TwoJTmin = abs( TwoSLD - TwoSD );
                                       const int TwoJTmax = TwoSLD + TwoSD;
                                       for ( int TwoJT = TwoJTmin; TwoJT <= TwoJTmax; TwoJT += 2 ) {
                                          const int dimMD = denBK->gCurrentDim( index + 1, NLD + ND, TwoJT, ILT );
                                          if ( dimMD > 0 ) {
                                             if ( TwoJT == TwoJ && NLU + NU == NLD + ND && IL == ILT ) {
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
               }
            }
         }
      }
   }
   sectorNLU        = new int[ nKappa ];
   sectorTwoSLU     = new int[ nKappa ];
   sectorILU        = new int[ nKappa ];
   sectorTwoJ       = new int[ nKappa ];
   sectorNU         = new int[ nKappa ];
   sectorND         = new int[ nKappa ];
   sectorNLD        = new int[ nKappa ];
   sectorTwoJT      = new int[ nKappa ];
   sectorND         = new int[ nKappa ];
   sectorTwoSLD     = new int[ nKappa ];
   sectorILD        = new int[ nKappa ];
   kappa2index      = new int[ nKappa + 1 ];
   kappa2index[ 0 ] = 0;

   nKappa = 0;

   for ( int NLU = denBK->gNmin( index ); NLU <= denBK->gNmax( index ); NLU++ ) {
      for ( int TwoSLU = denBK->gTwoSmin( index, NLU ); TwoSLU <= denBK->gTwoSmax( index, NLU ); TwoSLU += 2 ) {
         for ( int ILU = 0; ILU < denBK->getNumberOfIrreps(); ILU++ ) {
            const int dimLU = denBK->gCurrentDim( index, NLU, TwoSLU, ILU );
            if ( dimLU > 0 ) {
               for ( int NU = 0; NU <= 2; NU++ ) {
                  const int TwoSU   = ( NU == 1 ) ? 1 : 0;
                  const int IL      = ( ( NU == 1 ) ? Irreps::directProd( ILU, Ilocal1 ) : ILU );
                  const int TwoJmin = abs( TwoSLU - TwoSU );
                  const int TwoJmax = TwoSLU + TwoSU;
                  for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {
                     const int dimM = denBK->gCurrentDim( index + 1, NLU + NU, TwoJ, IL );
                     if ( dimM > 0 ) {
                        for ( int NLD = denBK->gNmin( index ); NLD <= denBK->gNmax( index ); NLD++ ) {
                           for ( int TwoSLD = denBK->gTwoSmin( index, NLD ); TwoSLD <= denBK->gTwoSmax( index, NLD ); TwoSLD += 2 ) {
                              for ( int ILD = 0; ILD < denBK->getNumberOfIrreps(); ILD++ ) {
                                 const int dimLD = denBK->gCurrentDim( index, NLD, TwoSLD, ILD );
                                 if ( dimLD > 0 ) {
                                    for ( int ND = 0; ND <= 2; ND++ ) {
                                       const int TwoSD    = ( ND == 1 ) ? 1 : 0;
                                       const int ILT      = ( ( ND == 1 ) ? Irreps::directProd( ILD, Ilocal1 ) : ILD );
                                       const int TwoJTmin = abs( TwoSLD - TwoSD );
                                       const int TwoJTmax = TwoSLD + TwoSD;
                                       for ( int TwoJT = TwoJTmin; TwoJT <= TwoJTmax; TwoJT += 2 ) {
                                          const int dimMD = denBK->gCurrentDim( index + 1, NLD + ND, TwoJT, ILT );
                                          if ( dimMD > 0 ) {
                                             if ( TwoJT == TwoJ && NLU + NU == NLD + ND && IL == ILT ) {
                                                sectorNLU[ nKappa ]    = NLU;
                                                sectorTwoSLU[ nKappa ] = TwoSLU;
                                                sectorILU[ nKappa ]    = ILU;
                                                sectorTwoJ[ nKappa ]   = TwoJ;
                                                sectorNU[ nKappa ]     = NU;
                                                sectorND[ nKappa ]     = ND;
                                                sectorTwoJT[ nKappa ]  = TwoJT;
                                                sectorNLD[ nKappa ]    = NLD;
                                                sectorTwoSLD[ nKappa ] = TwoSLD;
                                                sectorILD[ nKappa ]    = ILD;
                                                nKappa++;
                                                kappa2index[ nKappa ] = kappa2index[ nKappa - 1 ] + dimLU * dimLD;
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

// CheMPS2::CSobject::CSobject( const CSobject * cpy ) : index( cpy->gIndex() ), denBK( cpy->gBK_non_constant() ), Ilocal1( denBK->gIrrep( index ) ), Ilocal2( denBK->gIrrep( index + 1 ) ) {
//    nKappa = cpy->gNKappa();

//    sectorNL    = new int[ nKappa ];
//    sectorTwoSL = new int[ nKappa ];
//    sectorIL    = new int[ nKappa ];
//    sectorN1    = new int[ nKappa ];
//    sectorN2    = new int[ nKappa ];
//    sectorTwoJ  = new int[ nKappa ];
//    sectorNR    = new int[ nKappa ];
//    sectorTwoSR = new int[ nKappa ];
//    sectorIR    = new int[ nKappa ];
//    kappa2index = new int[ nKappa + 1 ];

//    std::copy( cpy->sectorNL, cpy->sectorNL + nKappa, sectorNL );
//    std::copy( cpy->sectorTwoSL, cpy->sectorTwoSL + nKappa, sectorTwoSL );
//    std::copy( cpy->sectorIL, cpy->sectorIL + nKappa, sectorIL );
//    std::copy( cpy->sectorN1, cpy->sectorN1 + nKappa, sectorN1 );
//    std::copy( cpy->sectorN2, cpy->sectorN2 + nKappa, sectorN2 );
//    std::copy( cpy->sectorTwoJ, cpy->sectorTwoJ + nKappa, sectorTwoJ );
//    std::copy( cpy->sectorNR, cpy->sectorNR + nKappa, sectorNR );
//    std::copy( cpy->sectorTwoSR, cpy->sectorTwoSR + nKappa, sectorTwoSR );
//    std::copy( cpy->sectorIR, cpy->sectorIR + nKappa, sectorIR );
//    std::copy( cpy->kappa2index, cpy->kappa2index + nKappa + 1, kappa2index );
//    storage = new dcomplex[ kappa2index[ nKappa ] ];
//    std::copy( cpy->storage, cpy->storage + kappa2index[ nKappa ], storage );
//    reorder = new int[ nKappa ];
//    std::copy( cpy->reorder, cpy->reorder + nKappa, reorder );
// }

CheMPS2::CDensityMatrix::~CDensityMatrix() {
   delete[] sectorNU;
   delete[] sectorTwoSLU;
   delete[] sectorILU;
   delete[] sectorTwoJ;
   delete[] sectorNU;
   delete[] sectorND;
   delete[] sectorTwoJ;
   delete[] sectorTwoJT;
   delete[] sectorNLD;
   delete[] sectorTwoSLD;
   delete[] sectorILD;
   delete[] kappa2index;
   delete[] storage;
   delete[] reorder;
}

// void CheMPS2::CSobject::Clear() {
//    for ( int i = 0; i < kappa2index[ nKappa ]; i++ ) {
//       storage[ i ] = 0.0;
//    }
// }

// int CheMPS2::CSobject::gNKappa() const { return nKappa; }

int CheMPS2::CDensityMatrix::gKappa( const int NLU, const int TwoSLU, const int ILU,
                                     const int TwoJ, const int NU,
                                     const int ND, const int TwoJT,
                                     const int NLD, const int TwoSLD, const int ILD ) {
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      if ( ( sectorNLU[ ikappa ] == NLU ) && ( sectorTwoSLU[ ikappa ] == TwoSLU ) && ( sectorILU[ ikappa ] == ILU ) &&
           ( sectorTwoJ[ ikappa ] == TwoJ ) && ( sectorNU[ ikappa ] == NU ) &&
           ( sectorND[ ikappa ] == ND ) && ( sectorTwoJT[ ikappa ] == TwoJT ) &&
           ( sectorNLD[ ikappa ] == NLD ) && ( sectorTwoSLD[ ikappa ] == TwoSLD ) && ( sectorILD[ ikappa ] == ILD ) ) {
         return ikappa;
      }
   }

   return -1;
}
// dcomplex * CheMPS2::CSobject::gStorage() { return storage; }

// int CheMPS2::CSobject::gReorder( const int ikappa ) const {
//    return reorder[ ikappa ];
// }

// int CheMPS2::CSobject::gKappa( const int NL, const int TwoSL, const int IL,
//                                const int N1, const int N2, const int TwoJ,
//                                const int NR, const int TwoSR, const int IR ) const {
//    for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
//       if ( ( sectorNL[ ikappa ] == NL ) && ( sectorTwoSL[ ikappa ] == TwoSL ) &&
//            ( sectorIL[ ikappa ] == IL ) && ( sectorN1[ ikappa ] == N1 ) &&
//            ( sectorN2[ ikappa ] == N2 ) && ( sectorTwoJ[ ikappa ] == TwoJ ) &&
//            ( sectorNR[ ikappa ] == NR ) && ( sectorTwoSR[ ikappa ] == TwoSR ) &&
//            ( sectorIR[ ikappa ] == IR ) ) {
//          return ikappa;
//       }
//    }

//    return -1;
// }

// int CheMPS2::CSobject::gKappa2index( const int kappa ) const {
//    return kappa2index[ kappa ];
// }

dcomplex * CheMPS2::CDensityMatrix::gStorage( const int NLU, const int TwoSLU, const int ILU,
                                              const int TwoJ, const int NU,
                                              const int ND, const int TwoJT,
                                              const int NLD, const int TwoSLD, const int ILD ) {
   const int kappa = gKappa( NLU, TwoSLU, ILU, TwoJ, NU, ND, TwoJT, NLD, TwoSLD, ILD );
   if ( kappa == -1 ) {
      return NULL;
   }
   return storage + kappa2index[ kappa ];
}

// int CheMPS2::CSobject::gIndex() const { return index; }

// int CheMPS2::CSobject::gNL( const int ikappa ) const { return sectorNL[ ikappa ]; }

// int CheMPS2::CSobject::gTwoSL( const int ikappa ) const {
//    return sectorTwoSL[ ikappa ];
// }

// int CheMPS2::CSobject::gIL( const int ikappa ) const { return sectorIL[ ikappa ]; }

// int CheMPS2::CSobject::gN1( const int ikappa ) const { return sectorN1[ ikappa ]; }

// int CheMPS2::CSobject::gN2( const int ikappa ) const { return sectorN2[ ikappa ]; }

// int CheMPS2::CSobject::gTwoJ( const int ikappa ) const {
//    return sectorTwoJ[ ikappa ];
// }

// int CheMPS2::CSobject::gNR( const int ikappa ) const { return sectorNR[ ikappa ]; }

// int CheMPS2::CSobject::gTwoSR( const int ikappa ) const {
//    return sectorTwoSR[ ikappa ];
// }

// int CheMPS2::CSobject::gIR( const int ikappa ) const { return sectorIR[ ikappa ]; }

void CheMPS2::CDensityMatrix::Make( CTensorT * tensor ) {

   char cotrans = 'C';
   char notrans = 'N';

   // #pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      const int NLU    = sectorNLU[ ikappa ];
      const int TwoSLU = sectorTwoSLU[ ikappa ];
      const int ILU    = sectorILU[ ikappa ];

      const int NU    = sectorNU[ ikappa ];
      const int TwoSU = ( NU == 1 ) ? 1 : 0;
      const int IU    = ( ( NU == 1 ) ? Irreps::directProd( ILU, Ilocal1 ) : ILU );
      const int TwoJ  = sectorTwoJ[ ikappa ];

      const int NLD    = sectorNLD[ ikappa ];
      const int TwoSLD = sectorTwoSLD[ ikappa ];
      const int ILD    = sectorILD[ ikappa ];

      const int ND    = sectorND[ ikappa ];
      const int TwoSD = ( ND == 1 ) ? 1 : 0;
      const int ID    = ( ( ND == 1 ) ? Irreps::directProd( ILD, Ilocal1 ) : ILD );
      const int TwoJT = sectorTwoJT[ ikappa ];

      // Clear block
      int dimU = denBK->gCurrentDim( index, NLU, TwoSLU, ILU ); // dimL > 0, checked at creation
      int dimD = denBK->gCurrentDim( index, NLD, TwoSLD, ILD ); // dimR > 0, checked at creation

      dcomplex * block_s = storage + kappa2index[ ikappa ];
      for ( int cnt = 0; cnt < dimU * dimD; cnt++ ) {
         block_s[ cnt ] = 0.0;
      }

      int dimM = denBK->gCurrentDim( index + 1, NLU + NU, TwoJ, IU ); // dimL > 0, checked at creation
      if ( dimM > 0 ) {
         dcomplex * block_upper = tensor->gStorage( NLU, TwoSLU, ILU, NLU + NU, TwoJ, IU );
         dcomplex * block_lower = tensor->gStorage( NLD, TwoSLD, ILD, NLD + ND, TwoJT, ID );
         dcomplex prefactor     = 1.0;
         dcomplex add           = 1.0;
         zgemm_( &notrans, &cotrans, &dimU, &dimD, &dimM, &prefactor, block_upper, &dimU, block_lower, &dimD, &add, block_s, &dimU );
      }
   }
}

// void CheMPS2::CSobject::Join( CTensorO * Oleft, CTensorT * Tleft, CTensorT * Tright, CTensorO * Oright ) {

//    const bool atLeft  = ( index == 0 ) ? true : false;
//    const bool atRight = ( index == denBK->gL() - 2 ) ? true : false;

//    const int DIM_L = std::max( denBK->gMaxDimAtBound( index ), Tleft->gBK()->gMaxDimAtBound( index ) );
//    const int DIM_M = std::max( denBK->gMaxDimAtBound( index + 1 ), Tleft->gBK()->gMaxDimAtBound( index + 1 ) );
//    const int DIM_R = std::max( denBK->gMaxDimAtBound( index + 2 ), Tleft->gBK()->gMaxDimAtBound( index + 2 ) );

//    char cotrans = 'C';
//    char notrans = 'N';

// #pragma omp parallel
//    {
//       dcomplex * tempA = new dcomplex[ DIM_L * DIM_M ];
//       dcomplex * tempB = new dcomplex[ DIM_M * DIM_R ];
//       dcomplex * tempC = new dcomplex[ DIM_L * DIM_R ];

// #pragma omp for schedule( dynamic )
//       for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
//          const int NL    = sectorNL[ ikappa ];
//          const int TwoSL = sectorTwoSL[ ikappa ];
//          const int IL    = sectorIL[ ikappa ];

//          const int NR    = sectorNR[ ikappa ];
//          const int TwoSR = sectorTwoSR[ ikappa ];
//          const int IR    = sectorIR[ ikappa ];

//          const int TwoJ       = sectorTwoJ[ ikappa ];
//          const int N1         = sectorN1[ ikappa ];
//          const int NM         = NL + N1;
//          const int N2         = sectorN2[ ikappa ];
//          const int TwoS1      = ( ( N1 == 1 ) ? 1 : 0 );
//          const int TwoS2      = ( ( N2 == 1 ) ? 1 : 0 );
//          const int TwoJMlower = max( abs( TwoSL - TwoS1 ), abs( TwoSR - TwoS2 ) );
//          const int TwoJMupper = min( ( TwoSL + TwoS1 ), ( TwoSR + TwoS2 ) );
//          const int IM         = ( ( TwoS1 == 1 ) ? Irreps::directProd( IL, Ilocal1 ) : IL );
//          const int fase       = Special::phase( TwoSL + TwoSR + TwoS1 + TwoS2 );

//          int dimLU = denBK->gCurrentDim( index, NL, TwoSL, IL );
//          int dimRU = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );

//          int dimLD = Tleft->gBK()->gCurrentDim( index, NL, TwoSL, IL );
//          int dimRD = Tleft->gBK()->gCurrentDim( index + 2, NR, TwoSR, IR );

//          dcomplex * block_s = storage + kappa2index[ ikappa ];
//          for ( int cnt = 0; cnt < dimLU * dimRU; cnt++ ) {
//             block_s[ cnt ] = 0.0;
//          }

//          for ( int TwoJM = TwoJMlower; TwoJM <= TwoJMupper; TwoJM += 2 ) {
//             int dimM = Tleft->gBK()->gCurrentDim( index + 1, NM, TwoJM, IM );
//             if ( dimLD > 0 && dimRD > 0 && dimM > 0 ) {

//                if ( !atLeft && !atRight ) {

//                   dcomplex * overlap_left  = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
//                   dcomplex * block_left    = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
//                   dcomplex * block_right   = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );
//                   dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

//                   dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
//                   dcomplex add       = 1.0;
//                   dcomplex noadd     = 0.0;
//                   zgemm_( &notrans, &notrans, &dimLU, &dimM, &dimLD, &prefactor, overlap_left, &dimLU, block_left, &dimLD, &noadd, tempA, &dimLU );

//                   prefactor = 1.0;
//                   zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimM, &prefactor, tempA, &dimLU, block_right, &dimM, &noadd, tempC, &dimLU );

//                   zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, tempC, &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
//                }
//                if ( !atLeft && atRight ) {

//                   dcomplex * overlap_left = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
//                   dcomplex * block_left   = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
//                   dcomplex * block_right  = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );

//                   dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
//                   dcomplex add       = 1.0;
//                   dcomplex noadd     = 0.0;
//                   zgemm_( &notrans, &notrans, &dimLU, &dimM, &dimLD, &prefactor, overlap_left, &dimLU, block_left, &dimLD, &noadd, tempA, &dimLU );

//                   prefactor = 1.0;
//                   zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimM, &prefactor, tempA, &dimLU, block_right, &dimM, &add, block_s, &dimLU );
//                }
//                if ( atLeft && !atRight ) {
//                   dcomplex * block_left    = Tleft->gStorage( NL, TwoSL, IL, NM, TwoJM, IM );
//                   dcomplex * block_right   = Tright->gStorage( NM, TwoJM, IM, NR, TwoSR, IR );
//                   dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

//                   dcomplex prefactor = fase * sqrt( 1.0 * ( TwoJ + 1 ) * ( TwoJM + 1 ) ) * Wigner::wigner6j( TwoSL, TwoSR, TwoJ, TwoS2, TwoS1, TwoJM );
//                   dcomplex add       = 1.0;
//                   dcomplex noadd     = 0.0;
//                   zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimM, &prefactor, block_left, &dimLU, block_right, &dimM, &noadd, tempC, &dimLU );

//                   prefactor = 1.0;
//                   zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, tempC, &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
//                }
//             }
//          }
//       }
//       delete[] tempA;
//       delete[] tempB;
//       delete[] tempC;
//    }
// }

// void CheMPS2::CSobject::Join( CTensorO * Oleft, CSobject * innerS, CTensorO * Oright ) {

//    const bool atLeft  = ( index == 0 ) ? true : false;
//    const bool atRight = ( index == denBK->gL() - 2 ) ? true : false;

//    const int DIM_L = std::max( denBK->gMaxDimAtBound( index ),
//                                innerS->gBK()->gMaxDimAtBound( index ) );
//    const int DIM_R = std::max( denBK->gMaxDimAtBound( index + 2 ),
//                                innerS->gBK()->gMaxDimAtBound( index + 2 ) );

//    char cotrans = 'C';
//    char notrans = 'N';

// #pragma omp parallel
//    {
//       dcomplex * temp = new dcomplex[ DIM_L * DIM_R ];

// #pragma omp for schedule( dynamic )
//       for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
//          const int NL    = sectorNL[ ikappa ];
//          const int TwoSL = sectorTwoSL[ ikappa ];
//          const int IL    = sectorIL[ ikappa ];

//          const int NR    = sectorNR[ ikappa ];
//          const int TwoSR = sectorTwoSR[ ikappa ];
//          const int IR    = sectorIR[ ikappa ];

//          const int TwoJ  = sectorTwoJ[ ikappa ];
//          const int N1    = sectorN1[ ikappa ];
//          const int N2    = sectorN2[ ikappa ];
//          const int TwoS1 = ( ( N1 == 1 ) ? 1 : 0 );
//          const int TwoS2 = ( ( N2 == 1 ) ? 1 : 0 );

//          int dimLU = denBK->gCurrentDim( index, NL, TwoSL, IL );
//          int dimRU = denBK->gCurrentDim( index + 2, NR, TwoSR, IR );

//          int dimLD = innerS->gBK()->gCurrentDim( index, NL, TwoSL, IL );
//          int dimRD = innerS->gBK()->gCurrentDim( index + 2, NR, TwoSR, IR );

//          if ( dimLD > 0 && dimRD > 0 ) {
//             dcomplex * block_s = storage + kappa2index[ ikappa ];

//             if ( !atLeft && !atRight ) {
//                dcomplex * overlap_left  = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
//                dcomplex * block_middle  = innerS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
//                dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

//                dcomplex prefactor = 1.0;
//                dcomplex add       = 0.0;
//                zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &prefactor, overlap_left,
//                        &dimLU, block_middle, &dimLD, &add, temp, &dimLU );

//                zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, temp,
//                        &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
//             }
//             if ( !atLeft && atRight ) {
//                dcomplex * overlap_left = Oleft->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
//                dcomplex * block_middle = innerS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );

//                dcomplex prefactor = 1.0;
//                dcomplex add       = 0.0;

//                zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &prefactor, overlap_left,
//                        &dimLU, block_middle, &dimLD, &add, block_s, &dimLU );
//             }
//             if ( atLeft && !atRight ) {
//                dcomplex * block_middle  = innerS->gStorage( NL, TwoSL, IL, N1, N2, TwoJ, NR, TwoSR, IR );
//                dcomplex * overlap_right = Oright->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

//                dcomplex prefactor = 1.0;
//                dcomplex add       = 0.0;

//                zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &prefactor, block_middle,
//                        &dimLU, overlap_right, &dimRU, &add, block_s, &dimLD );
//             }
//          }
//       }
//       delete[] temp;
//    }
// }

// void CheMPS2::CSobject::Add( dcomplex alpha, CSobject * to_add ) {
//    assert( index == to_add->gIndex() );
//    assert( nKappa == to_add->gNKappa() );
//    assert( kappa2index[ nKappa ] == to_add->gKappa2index( to_add->gNKappa() ) );

//    int inc = 1;
//    zaxpy_( kappa2index + nKappa, &alpha, to_add->gStorage(), &inc, storage, &inc );
// }

// void CheMPS2::CSobject::Multiply( dcomplex alpha ) {
//    int inc = 1;
//    zscal_( kappa2index + nKappa, &alpha, storage, &inc );
// }

double CheMPS2::CDensityMatrix::Split( CTensorT * tensor, const int virtualdimensionD,
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
   double ** Lambdas  = NULL;
   dcomplex ** Us     = NULL;
   dcomplex ** Swaped = NULL;
   int * CenterDims   = NULL;
   int * DimLtotal    = NULL;
   int * DimRtotal    = NULL;

   Lambdas    = new double *[ nCenterSectors ];
   Us         = new dcomplex *[ nCenterSectors ];
   Swaped     = new dcomplex *[ nCenterSectors ];
   CenterDims = new int[ nCenterSectors ];
   DimLtotal  = new int[ nCenterSectors ];
   DimRtotal  = new int[ nCenterSectors ];

   // PARALLEL
   // #pragma omp parallel for schedule( dynamic )
   for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {
      // Determine left and right dimensions contributing to the center block iCenter
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
      CenterDims[ iCenter ] = DimLtotal[ iCenter ]; // CenterDims contains the min. amount
      // Allocate memory to copy the different parts of the S-object. Use
      // prefactor sqrt((2jR+1)/(2jM+1) * (2jM+1) * (2j+1)) W6J
      // (-1)^(jL+jR+s1+s2) and sum over j.
      if ( CenterDims[ iCenter ] > 0 ) {
         // Only if CenterDims[ iCenter ] exists should you allocate the
         // following three arrays
         Lambdas[ iCenter ] = new double[ CenterDims[ iCenter ] ];
         Us[ iCenter ]      = new dcomplex[ CenterDims[ iCenter ] * DimLtotal[ iCenter ] ];
         Swaped[ iCenter ]  = new dcomplex[ CenterDims[ iCenter ] * DimLtotal[ iCenter ] ];

         const int memsize = DimLtotal[ iCenter ] * DimLtotal[ iCenter ];
         for ( int cnt = 0; cnt < memsize; cnt++ ) {
            Us[ iCenter ][ cnt ]     = 0.0;
            Swaped[ iCenter ][ cnt ] = 0.0;
         }

         int dimLtotalA = 0;
         for ( int NLU = denBK->gNmin( index ); NLU <= denBK->gNmax( index ); NLU++ ) {
            for ( int TwoSLU = denBK->gTwoSmin( index, NLU ); TwoSLU <= denBK->gTwoSmax( index, NLU ); TwoSLU += 2 ) {
               for ( int ILU = 0; ILU < denBK->getNumberOfIrreps(); ILU++ ) {
                  const int dimLU = denBK->gCurrentDim( index, NLU, TwoSLU, ILU );
                  if ( dimLU > 0 ) {
                     for ( int NU = 0; NU <= 2; NU++ ) {
                        const int TwoSU   = ( NU == 1 ) ? 1 : 0;
                        const int IL      = ( ( NU == 1 ) ? Irreps::directProd( ILU, Ilocal1 ) : ILU );
                        const int TwoJmin = abs( TwoSLU - TwoSU );
                        const int TwoJmax = TwoSLU + TwoSU;
                        for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {

                           if ( NLU + NU == SplitSectNM[ iCenter ] && TwoJ == SplitSectTwoJM[ iCenter ] && IL == SplitSectIM[ iCenter ] ) {
                              int dimLtotalB = 0;
                              for ( int NLD = denBK->gNmin( index ); NLD <= denBK->gNmax( index ); NLD++ ) {
                                 for ( int TwoSLD = denBK->gTwoSmin( index, NLD ); TwoSLD <= denBK->gTwoSmax( index, NLD ); TwoSLD += 2 ) {
                                    for ( int ILD = 0; ILD < denBK->getNumberOfIrreps(); ILD++ ) {
                                       const int dimLD = denBK->gCurrentDim( index, NLD, TwoSLD, ILD );
                                       if ( dimLD > 0 ) {
                                          for ( int ND = 0; ND <= 2; ND++ ) {
                                             const int TwoSD    = ( ND == 1 ) ? 1 : 0;
                                             const int ILT      = ( ( ND == 1 ) ? Irreps::directProd( ILD, Ilocal1 ) : ILD );
                                             const int TwoJTmin = abs( TwoSLD - TwoSD );
                                             const int TwoJTmax = TwoSLD + TwoSD;
                                             for ( int TwoJT = TwoJTmin; TwoJT <= TwoJTmax; TwoJT += 2 ) {
                                                if ( TwoJT == TwoJ && NLU + NU == NLD + ND && IL == ILT ) {
                                                   dcomplex * block = gStorage( NLU, TwoSLU, ILU, TwoJ, NU, ND, TwoJT, NLD, TwoSLD, ILD );
                                                   for ( int l = 0; l < dimLU; l++ ) {
                                                      for ( int r = 0; r < dimLD; r++ ) {
                                                         Us[ iCenter ][ dimLtotalA + l + DimLtotal[ iCenter ] * ( dimLtotalB + r ) ] += block[ l + dimLU * r ];
                                                      }
                                                   }
                                                   dimLtotalB += dimLD;
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                              dimLtotalA += dimLU;
                           }
                        }
                     }
                  }
               }
            }
         }

         // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
         // central symmetry
         char jobz       = 'V'; // M x min(M,N) in U and min(M,N) x N in VT
         char uplo       = 'U';
         double * rwork  = new double[ 3 * CenterDims[ iCenter ] - 2 ];
         dcomplex * work = new dcomplex[ 2 * CenterDims[ iCenter ] - 1 ];
         int lwork       = 2 * CenterDims[ iCenter ] - 1;

         int * iwork = new int[ 8 * CenterDims[ iCenter ] ];
         int info;
         zheev_( &jobz, &uplo, CenterDims + iCenter, Us[ iCenter ], CenterDims + iCenter, Lambdas[ iCenter ], work, &lwork, rwork, &info );
         for ( int i = 0; i < CenterDims[ iCenter ]; i++ ) {
            for ( int j = 0; j < CenterDims[ iCenter ]; j++ ) {
               Swaped[ iCenter ][ i + CenterDims[ iCenter ] * j ] = Us[ iCenter ][ i + CenterDims[ iCenter ] * ( CenterDims[ iCenter ] - 1 - j ) ];
            }
         }

         delete[] work;
         delete[] rwork;
         delete[] iwork;
      }
   }

   // // Copy first dimM per central symmetry sector to the relevant parts
   // #pragma omp parallel for schedule( dynamic )
   for ( int iCenter = 0; iCenter < nCenterSectors; iCenter++ ) {

      const int dimM = denBK->gCurrentDim( index + 1, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
      if ( dimM > 0 ) {
         int dimLtotalA = 0;
         int lmbcount   = 0;
         for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
            for ( int TwoSL = denBK->gTwoSmin( index, NL ); TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
               for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
                  const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
                  if ( dimL > 0 ) {
                     for ( int N = 0; N <= 2; N++ ) {
                        const int TwoS    = ( N == 1 ) ? 1 : 0;
                        const int I       = ( N == 1 ) ? Irreps::directProd( IL, Ilocal1 ) : IL;
                        const int TwoJmin = abs( TwoSL - TwoS );
                        const int TwoJmax = TwoSL + TwoS;
                        for ( int TwoJ = TwoJmin; TwoJ <= TwoJmax; TwoJ += 2 ) {
                           if ( TwoJ == SplitSectTwoJM[ iCenter ] && I == SplitSectIM[ iCenter ] && NL + N == SplitSectNM[ iCenter ] ) {
                              dcomplex * block = tensor->gStorage( NL, TwoSL, IL, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
                              for ( int r = 0; r < dimM; r++ ) {
                                 for ( int l = 0; l < dimL; l++ ) {
                                    block[ l + dimL * r ] = Swaped[ iCenter ][ dimLtotalA + l + CenterDims[ iCenter ] * r ];
                                 }
                              }
                              dimLtotalA += dimL;
                              lmbcount++;
                           }
                        }
                     }
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
         delete[] Swaped[ iCenter ];
      }
   }
   delete[] Us;
   delete[] Lambdas;
   delete[] Swaped;
   delete[] CenterDims;
   delete[] DimLtotal;
   delete[] DimRtotal;

   return discardedWeight;
}

// void CheMPS2::CSobject::prog2symm() {
// #pragma omp parallel for schedule( dynamic )
//    for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
//       int dim        = kappa2index[ ikappa + 1 ] - kappa2index[ ikappa ];
//       dcomplex alpha = sqrt( sectorTwoSR[ ikappa ] + 1.0 );
//       int inc        = 1;
//       zscal_( &dim, &alpha, storage + kappa2index[ ikappa ], &inc );
//    }
// }

// void CheMPS2::CSobject::symm2prog() {
// #pragma omp parallel for schedule( dynamic )
//    for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
//       int dim        = kappa2index[ ikappa + 1 ] - kappa2index[ ikappa ];
//       dcomplex alpha = 1.0 / sqrt( sectorTwoSR[ ikappa ] + 1.0 );
//       int inc        = 1;
//       zscal_( &dim, &alpha, storage + kappa2index[ ikappa ], &inc );
//    }
// }

// void CheMPS2::CSobject::addNoise( const double NoiseLevel ) {
//    for ( int cnt = 0; cnt < gKappa2index( gNKappa() ); cnt++ ) {
//       const dcomplex RN = ( ( double ) rand() ) / RAND_MAX - 0.5;
//       gStorage()[ cnt ] += RN * NoiseLevel;
//    }
// }

void CheMPS2::CDensityMatrix::print() const {
   std::cout << "############################################################" << std::endl;
   std::cout << "CSobject with " << nKappa << " symmetry blocks: " << std::endl;

   for ( int ikappa = 0; ikappa < nKappa; ++ikappa ) {
      const int NLU    = sectorNLU[ ikappa ];
      const int TwoSLU = sectorTwoSLU[ ikappa ];
      const int ILU    = sectorILU[ ikappa ];
      const int TwoJ   = sectorTwoJ[ ikappa ];

      const int NU = sectorNU[ ikappa ];
      const int ND = sectorND[ ikappa ];

      const int TwoJT  = sectorTwoJT[ ikappa ];
      const int NLD    = sectorNLD[ ikappa ];
      const int TwoSLD = sectorTwoSLD[ ikappa ];
      const int ILD    = sectorILD[ ikappa ];

      std::cout << "Block number " << ikappa << std::endl;

      std::cout << "NLU:    " << NLU << std::endl;
      std::cout << "TwoSLU: " << TwoSLU << std::endl;
      std::cout << "ILU:    " << ILU << std::endl;
      std::cout << "NU:     " << NU << std::endl;
      std::cout << "TwoJ:   " << TwoJ << std::endl;
      std::cout << "    --- " << std::endl;
      std::cout << "NLD:    " << NLD << std::endl;
      std::cout << "TwoSLD: " << TwoSLD << std::endl;
      std::cout << "ILD:    " << ILD << std::endl;
      std::cout << "ND:     " << ND << std::endl;
      std::cout << "TwoJT:  " << TwoJT << std::endl;

      for ( int i = kappa2index[ ikappa ]; i < kappa2index[ ikappa + 1 ]; ++i ) {
         std::cout << storage[ i ] << std::endl;
      }
   }
   std::cout << "############################################################" << std::endl;
}
