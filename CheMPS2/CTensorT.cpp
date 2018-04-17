
// #include <math.h>
// #include <stdlib.h> /*rand*/
#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CTensorO.h"
#include "CTensorT.h"
#include "Lapack.h"
#include "Logger.h"
#include "Special.h"
#include "Wigner.h"

using std::min;

CheMPS2::CTensorT::CTensorT( const int site_index,
                             const CheMPS2::SyBookkeeper * denBK )
    : CTensor() {
   this->index = site_index;
   this->denBK = denBK;

   AllocateAllArrays();
}

CheMPS2::CTensorT::CTensorT( CTensorT * cpy ) : CTensor() {

   this->index = cpy->gIndex(); //left boundary = index ; right boundary = index+1
   this->denBK = cpy->gBK();

   AllocateAllArrays();

   int size = kappa2index[ nKappa ];
   std::copy( &cpy->gStorage()[ 0 ], &cpy->gStorage()[ size ], &this->storage[ 0 ] );
}

void CheMPS2::CTensorT::AllocateAllArrays() {
   nKappa = 0;
   for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( index, NL ); TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               for ( int NR = NL; NR <= NL + 2; NR++ ) {
                  const int TwoJ = ( ( NR == NL + 1 ) ? 1 : 0 );
                  for ( int TwoSR = TwoSL - TwoJ; TwoSR <= TwoSL + TwoJ; TwoSR += 2 ) {
                     if ( TwoSR >= 0 ) {
                        int IR         = ( ( NR == NL + 1 ) ? Irreps::directProd( IL, denBK->gIrrep( index ) ) : IL );
                        const int dimR = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );
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

   sectorNL         = new int[ nKappa ];
   sectorNR         = new int[ nKappa ];
   sectorIL         = new int[ nKappa ];
   sectorIR         = new int[ nKappa ];
   sectorTwoSL      = new int[ nKappa ];
   sectorTwoSR      = new int[ nKappa ];
   kappa2index      = new int[ nKappa + 1 ];
   kappa2index[ 0 ] = 0;

   nKappa = 0;
   for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( index, NL );
            TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            const int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               for ( int NR = NL; NR <= NL + 2; NR++ ) {
                  const int TwoJ = ( ( NR == NL + 1 ) ? 1 : 0 );
                  for ( int TwoSR = TwoSL - TwoJ; TwoSR <= TwoSL + TwoJ; TwoSR += 2 ) {
                     if ( TwoSR >= 0 ) {
                        int IR         = ( ( NR == NL + 1 ) ? Irreps::directProd( IL, denBK->gIrrep( index ) ) : IL );
                        const int dimR = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );
                        if ( dimR > 0 ) {
                           sectorNL[ nKappa ]        = NL;
                           sectorNR[ nKappa ]        = NR;
                           sectorIL[ nKappa ]        = IL;
                           sectorIR[ nKappa ]        = IR;
                           sectorTwoSL[ nKappa ]     = TwoSL;
                           sectorTwoSR[ nKappa ]     = TwoSR;
                           kappa2index[ nKappa + 1 ] = kappa2index[ nKappa ] + dimL * dimR;
                           nKappa++;
                        }
                     }
                  }
               }
            }
         }
      }
   }

   storage = new dcomplex[ kappa2index[ nKappa ] ];
}

CheMPS2::CTensorT::~CTensorT() { DeleteAllArrays(); }

void CheMPS2::CTensorT::DeleteAllArrays() {
   delete[] sectorNL;
   delete[] sectorNR;
   delete[] sectorIL;
   delete[] sectorIR;
   delete[] sectorTwoSL;
   delete[] sectorTwoSR;
   delete[] kappa2index;
   delete[] storage;
}

void CheMPS2::CTensorT::Reset() {
   DeleteAllArrays();
   AllocateAllArrays();
}

int CheMPS2::CTensorT::gNKappa() const { return nKappa; }

dcomplex * CheMPS2::CTensorT::gStorage() { return storage; }

int CheMPS2::CTensorT::gKappa( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 ) const {
   for ( int cnt = 0; cnt < nKappa; cnt++ ) {
      if ( ( sectorNL[ cnt ] == N1 ) && ( sectorNR[ cnt ] == N2 ) && ( sectorIL[ cnt ] == I1 ) && ( sectorIR[ cnt ] == I2 ) && ( sectorTwoSL[ cnt ] == TwoS1 ) && ( sectorTwoSR[ cnt ] == TwoS2 ) ) {
         return cnt;
      }
   }

   return -1;
}

int CheMPS2::CTensorT::gKappa2index( const int kappa ) const {
   return kappa2index[ kappa ];
}

dcomplex * CheMPS2::CTensorT::gStorage( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 ) {
   int kappa = gKappa( N1, TwoS1, I1, N2, TwoS2, I2 );
   if ( kappa == -1 ) {
      return NULL;
   }
   return storage + kappa2index[ kappa ];
}

int CheMPS2::CTensorT::gIndex() const { return index; }

const CheMPS2::SyBookkeeper * CheMPS2::CTensorT::gBK() const { return denBK; }

void CheMPS2::CTensorT::sBK( const CheMPS2::SyBookkeeper * newBK ) { denBK = newBK; }

void CheMPS2::CTensorT::random() {
   for ( int cnt = 0; cnt < kappa2index[ nKappa ]; cnt++ ) {
      storage[ cnt ] =
          dcomplex( ( ( double ) rand() ) / RAND_MAX, ( ( double ) rand() ) / RAND_MAX );
   }
}

// void CheMPS2::CTensorT::random() {
//    for ( int cnt = 0; cnt < kappa2index[ nKappa ]; cnt++ ) {
//       storage[ cnt ] = ( double ) rand() / RAND_MAX;
//    }
// }

void CheMPS2::CTensorT::Clear() {
   for ( int cnt = 0; cnt < kappa2index[ nKappa ]; cnt++ ) {
      storage[ cnt ] = 0.0;
   }
}

void CheMPS2::CTensorT::number_operator( dcomplex alpha, dcomplex beta ) {
#pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      int size         = kappa2index[ ikappa + 1 ] - kappa2index[ ikappa ];
      dcomplex * array = storage + kappa2index[ ikappa ];
      dcomplex factor  = beta + alpha * static_cast< double >( sectorNR[ ikappa ] - sectorNL[ ikappa ] );
      int inc1         = 1;
      zscal_( &size, &factor, array, &inc1 );
   }
}

void CheMPS2::CTensorT::scale( dcomplex alpha ) {
   int size = kappa2index[ nKappa ];
   int inc1 = 1;
   zscal_( &size, &alpha, storage, &inc1 );
}

void CheMPS2::CTensorT::add( CTensorT * toAdd ) {
   assert( index == toAdd->gIndex() );

   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {

      const int NL    = sectorNL[ ikappa ];
      const int TwoSL = sectorTwoSL[ ikappa ];
      const int IL    = sectorIL[ ikappa ];

      const int NR    = sectorNR[ ikappa ];
      const int TwoSR = sectorTwoSR[ ikappa ];
      const int IR    = sectorIR[ ikappa ];

      const int N    = NR - NL;
      const int TwoS = ( N == 1 ) ? 1 : 0;
      const int I    = denBK->gIrrep( index );

      int memNonExpKappa = toAdd->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );
      if ( memNonExpKappa != -1 ) {

         int dimLEXP = denBK->gCurrentDim( index, NL, TwoSL, IL );
         int dimREXP = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );

         int dimLNON = toAdd->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimRNON = toAdd->gBK()->gCurrentDim( index + 1, NR, TwoSR, IR );

         // If nonExpanded has the block it has to have the same size
         assert( dimLEXP == dimLNON );
         assert( dimREXP == dimRNON );

         int dim      = dimLEXP * dimREXP;
         int inc      = 1;
         dcomplex one = 1.0;

         dcomplex * BlockExp = storage + kappa2index[ ikappa ];
         dcomplex * BlockNon = toAdd->gStorage() + toAdd->gKappa2index( memNonExpKappa );

         zaxpy_( &dim, &one, BlockNon, &inc, BlockExp, &inc );
      }
   }
}

void CheMPS2::CTensorT::QR( CTensor * Rstorage ) {
// Left normalization occurs in T-convention: no pre or after multiplication
// Work per right symmetry sector

// PARALLEL
#pragma omp parallel for schedule( dynamic )
   for ( int NR = denBK->gNmin( index + 1 ); NR <= denBK->gNmax( index + 1 ); NR++ ) {
      for ( int TwoSR = denBK->gTwoSmin( index + 1, NR ); TwoSR <= denBK->gTwoSmax( index + 1, NR ); TwoSR += 2 ) {
         for ( int IR = 0; IR < denBK->getNumberOfIrreps(); IR++ ) {

            int dimR = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );

            if ( dimR > 0 ) {
               // Find out the total left dimension
               int dimLtotal = 0;
               for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
                  if ( ( NR == sectorNR[ ikappa ] ) && ( TwoSR == sectorTwoSR[ ikappa ] ) && ( IR == sectorIR[ ikappa ] ) ) {
                     dimLtotal += denBK->gCurrentDim( index, sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ] );
                  }
               }

               if ( dimLtotal > 0 ) {

                  dcomplex * mem = new dcomplex[ dimLtotal * dimR ];
                  //Copy the relevant parts from storage to mem
                  int dimLtotal2 = 0;
                  for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
                     if ( ( NR == sectorNR[ ikappa ] ) && ( TwoSR == sectorTwoSR[ ikappa ] ) && ( IR == sectorIR[ ikappa ] ) ) {
                        int dimL = denBK->gCurrentDim( index, sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ] );
                        if ( dimL > 0 ) {
                           for ( int l = 0; l < dimL; l++ ) {
                              for ( int r = 0; r < dimR; r++ ) {
                                 mem[ dimLtotal2 + l + dimLtotal * r ] = storage[ kappa2index[ ikappa ] + l + dimL * r ];
                              }
                           }
                           dimLtotal2 += dimL;
                        }
                     }
                  }

                  //QR mem --> m = dimLtotal ; n = dimR
                  int info;
                  int minofdims   = min( dimR, dimLtotal );
                  dcomplex * tau  = new dcomplex[ minofdims ];
                  dcomplex * work = new dcomplex[ dimR ];
                  zgeqrf_( &dimLtotal, &dimR, mem, &dimLtotal, tau, work, &dimR, &info );

                  //Copy R to Rstorage
                  dcomplex * wheretoput = Rstorage->gStorage( NR, TwoSR, IR, NR, TwoSR, IR ); //dimR x dimR

                  for ( int irow = 0; irow < minofdims; irow++ ) {
                     for ( int icol = 0; icol < irow; icol++ ) {
                        wheretoput[ irow + dimR * icol ] = 0.0;
                     }
                     for ( int icol = irow; icol < dimR; icol++ ) {
                        wheretoput[ irow + dimR * icol ] = mem[ irow + dimLtotal * icol ];
                     }
                  }
                  for ( int irow = minofdims; irow < dimR; irow++ ) {
                     for ( int icol = 0; icol < dimR; icol++ ) {
                        wheretoput[ irow + dimR * icol ] = 0.0;
                     }
                  }

                  //Construct Q
                  zungqr_( &dimLtotal, &minofdims, &minofdims, mem, &dimLtotal, tau, work, &dimR, &info );
                  if ( dimLtotal < dimR ) { //if number of cols larger than number of rows, rest of cols zero.
                     for ( int irow = 0; irow < dimLtotal; irow++ ) {
                        for ( int icol = dimLtotal; icol < dimR; icol++ ) {
                           mem[ irow + dimLtotal * icol ] = 0.0;
                        }
                     }
                  }

                  //Copy from mem to storage
                  dimLtotal2 = 0;
                  for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
                     if ( ( NR == sectorNR[ ikappa ] ) && ( TwoSR == sectorTwoSR[ ikappa ] ) && ( IR == sectorIR[ ikappa ] ) ) {
                        int dimL = denBK->gCurrentDim( index, sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ] );
                        if ( dimL > 0 ) {
                           for ( int l = 0; l < dimL; l++ ) {
                              for ( int r = 0; r < dimR; r++ ) {
                                 storage[ kappa2index[ ikappa ] + l + dimL * r ] = mem[ dimLtotal2 + l + dimLtotal * r ];
                              }
                           }
                           dimLtotal2 += dimL;
                        }
                     }
                  }

                  // Clear the memory
                  delete[] work;
                  delete[] tau;
                  delete[] mem;
               }
            }
         }
      }
   }
}

void CheMPS2::CTensorT::LQ( CTensor * Lstorage ) {
// Right normalization occurs in U-convention: pre-multiplication with
// sqrt{2jR+1/2jL+1} and after multiplication with sqrt{2jL+1/2jR+1}
// Work per left symmetry sector

// PARALLEL
#pragma omp parallel for schedule( dynamic )
   for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( index, NL ); TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {

            int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               // Find out the total right dimension
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
                  if ( ( NL == sectorNL[ ikappa ] ) && ( TwoSL == sectorTwoSL[ ikappa ] ) && ( IL == sectorIL[ ikappa ] ) ) {
                     dimRtotal += denBK->gCurrentDim( index + 1, sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ] );
                  }
               }

               if ( dimRtotal > 0 ) { // Due to the initial truncation, it is possible that dimRtotal is temporarily smaller than dimL ...

                  dcomplex * mem = new dcomplex[ dimRtotal * dimL ];
                  // Copy the relevant parts from storage to mem & multiply with factor !!
                  int dimRtotal2 = 0;
                  for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
                     if ( ( NL == sectorNL[ ikappa ] ) && ( TwoSL == sectorTwoSL[ ikappa ] ) && ( IL == sectorIL[ ikappa ] ) ) {
                        int dimR = denBK->gCurrentDim( index + 1, sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ] );
                        if ( dimR > 0 ) {
                           double factor = sqrt( ( sectorTwoSR[ ikappa ] + 1.0 ) / ( TwoSL + 1.0 ) );
                           for ( int l = 0; l < dimL; l++ ) {
                              for ( int r = 0; r < dimR; r++ ) {
                                 mem[ l + dimL * ( dimRtotal2 + r ) ] = factor * storage[ kappa2index[ ikappa ] + l + dimL * r ];
                              }
                           }
                           dimRtotal2 += dimR;
                        }
                     }
                  }

                  // LQ mem --> m = dimL ; n = dimRtotal
                  int info;
                  int minofdims   = min( dimL, dimRtotal );
                  dcomplex * tau  = new dcomplex[ minofdims ];
                  dcomplex * work = new dcomplex[ dimL ];
                  zgelqf_( &dimL, &dimRtotal, mem, &dimL, tau, work, &dimL, &info );

                  // Copy L to Lstorage
                  dcomplex * wheretoput = Lstorage->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );

                  for ( int irow = 0; irow < dimL; irow++ ) {
                     for ( int icol = 0; icol < min( irow + 1, dimRtotal ); icol++ ) { // icol can be max. irow and max. dimRtotal-1
                        wheretoput[ irow + dimL * icol ] = mem[ irow + dimL * icol ];
                     }
                     for ( int icol = min( irow + 1, dimRtotal ); icol < dimL; icol++ ) {
                        wheretoput[ irow + dimL * icol ] = 0.0;
                     }
                  }

                  // Construct Q
                  zunglq_( &minofdims, &dimRtotal, &minofdims, mem, &dimL, tau, work, &dimL, &info );
                  if ( dimRtotal < dimL ) { // if number of rows larger than number of cols, rest of rows zero.
                     for ( int irow = dimRtotal; irow < dimL; irow++ ) {
                        for ( int icol = 0; icol < dimRtotal; icol++ ) {
                           mem[ irow + dimL * icol ] = 0.0;
                        }
                     }
                  }

                  // Copy from mem to storage & multiply with factor !!
                  dimRtotal2 = 0;
                  for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
                     if ( ( NL == sectorNL[ ikappa ] ) && ( TwoSL == sectorTwoSL[ ikappa ] ) && ( IL == sectorIL[ ikappa ] ) ) {
                        int dimR = denBK->gCurrentDim( index + 1, sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ] );
                        if ( dimR > 0 ) {
                           double factor = sqrt( ( TwoSL + 1.0 ) / ( sectorTwoSR[ ikappa ] + 1.0 ) );
                           for ( int l = 0; l < dimL; l++ ) {
                              for ( int r = 0; r < dimR; r++ ) {
                                 storage[ kappa2index[ ikappa ] + l + dimL * r ] = factor * mem[ l + dimL * ( r + dimRtotal2 ) ];
                              }
                           }
                           dimRtotal2 += dimR;
                        }
                     }
                  }

                  // Clear the memory
                  delete[] work;
                  delete[] tau;
                  delete[] mem;
               }
            }
         }
      }
   }
}

void CheMPS2::CTensorT::LeftMultiply( CTensor * Mx, char * trans ) {
// PARALLEL
#pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      int dimL           = denBK->gCurrentDim( index, sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ] );
      int dimR           = denBK->gCurrentDim( index + 1, sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ] );
      dcomplex * MxBlock = Mx->gStorage( sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ],
                                         sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ] );
      char notrans       = 'N';
      dcomplex one       = 1.0;
      dcomplex zero      = 0.0;
      int dim            = dimL * dimR;
      dcomplex * mem     = new dcomplex[ dim ];
      zgemm_( trans, &notrans, &dimL, &dimR, &dimL, &one, MxBlock, &dimL, storage + kappa2index[ ikappa ], &dimL, &zero, mem, &dimL );
      int inc = 1;
      zcopy_( &dim, mem, &inc, storage + kappa2index[ ikappa ], &inc );
      delete[] mem;
   }
}

void CheMPS2::CTensorT::RightMultiply( CTensor * Mx, char * trans ) {
// PARALLEL
#pragma omp parallel for schedule( dynamic )
   for ( int ikappa = 0; ikappa < nKappa; ikappa++ ) {
      int dimL           = denBK->gCurrentDim( index, sectorNL[ ikappa ], sectorTwoSL[ ikappa ], sectorIL[ ikappa ] );
      int dimR           = denBK->gCurrentDim( index + 1, sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ] );
      dcomplex * MxBlock = Mx->gStorage( sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ],
                                         sectorNR[ ikappa ], sectorTwoSR[ ikappa ], sectorIR[ ikappa ] );

      char notrans   = 'N';
      char twtrans   = ( *trans == 'N' ) ? 'C' : 'N';
      dcomplex one   = 1.0;
      dcomplex zero  = 0.0;
      int dim        = dimL * dimR;
      dcomplex * mem = new dcomplex[ dim ];
      zgemm_( &notrans, &twtrans, &dimL, &dimR, &dimR, &one, storage + kappa2index[ ikappa ], &dimL, MxBlock, &dimR, &zero, mem, &dimL );

      int inc = 1;
      zcopy_( &dim, mem, &inc, storage + kappa2index[ ikappa ], &inc );
      delete[] mem;
   }
}

void CheMPS2::CTensorT::Join( CTensor * left, CTensorT * buddy, CTensor * right ) {

   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == denBK->gL() - 1 ) ? true : false;

   const int DIM_L = std::max( denBK->gMaxDimAtBound( index ), buddy->gBK()->gMaxDimAtBound( index ) );
   const int DIM_R = std::max( denBK->gMaxDimAtBound( index + 1 ), buddy->gBK()->gMaxDimAtBound( index + 1 ) );

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

         int dimLU = denBK->gCurrentDim( index, NL, TwoSL, IL );
         int dimRU = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );

         int dimLD = buddy->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimRD = buddy->gBK()->gCurrentDim( index + 1, NR, TwoSR, IR );

         if ( dimLD > 0 && dimRD > 0 ) {
            dcomplex * block_s = storage + kappa2index[ ikappa ];

            if ( !atLeft && !atRight ) {
               dcomplex * overlap_left  = left->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
               dcomplex * block_middle  = buddy->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
               dcomplex * overlap_right = right->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

               dcomplex prefactor = 1.0;
               dcomplex add       = 0.0;
               zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &prefactor, overlap_left,
                       &dimLU, block_middle, &dimLD, &add, temp, &dimLU );

               zgemm_( &notrans, &cotrans, &dimLU, &dimRU, &dimRD, &prefactor, temp,
                       &dimLU, overlap_right, &dimRU, &add, block_s, &dimLU );
            }
            if ( !atLeft && atRight ) {
               dcomplex * overlap_left = left->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
               dcomplex * block_middle = buddy->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               dcomplex prefactor = 1.0;
               dcomplex add       = 0.0;

               zgemm_( &notrans, &notrans, &dimLU, &dimRD, &dimLD, &prefactor, overlap_left,
                       &dimLU, block_middle, &dimLD, &add, block_s, &dimLU );
            }
            if ( atLeft && !atRight ) {
               dcomplex * block_middle  = buddy->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
               dcomplex * overlap_right = right->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );

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

//! Add CTensorT elements
void CheMPS2::CTensorT::zaxpy( dcomplex factor, CTensorT * y ) {
   int dim = kappa2index[ nKappa ];
   assert( dim == y->gKappa2index( y->gNKappa() ) );
   int inc = 1;
   zaxpy_( &dim, &factor, storage, &inc, y->gStorage(), &inc );
}

//! Add CTensorT elements
void CheMPS2::CTensorT::zcopy( CTensorT * y ) {
   int dim = kappa2index[ nKappa ];
   assert( dim == y->gKappa2index( y->gNKappa() ) );
   int inc = 1;
   zcopy_( &dim, storage, &inc, y->gStorage(), &inc );
}

bool CheMPS2::CTensorT::CheckLeftNormal() const {
   bool isLeftNormal = true;

   for ( int NR = denBK->gNmin( index + 1 ); NR <= denBK->gNmax( index + 1 ); NR++ ) {
      for ( int TwoSR = denBK->gTwoSmin( index + 1, NR ); TwoSR <= denBK->gTwoSmax( index + 1, NR ); TwoSR += 2 ) {
         for ( int IR = 0; IR < denBK->getNumberOfIrreps(); IR++ ) {

            int dimR = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );

            if ( dimR > 0 ) {

               dcomplex * result = new dcomplex[ dimR * dimR ];
               bool firsttime    = true;

               for ( int NL = NR - 2; NL <= NR; NL++ ) {
                  for ( int TwoSL = TwoSR - ( ( NR == NL + 1 ) ? 1 : 0 ); TwoSL < TwoSR + 2; TwoSL += 2 ) {
                     int IL   = ( NR == NL + 1 ) ? ( Irreps::directProd( denBK->gIrrep( index ), IR ) ) : IR;
                     int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );
                     if ( dimL > 0 ) {
                        dcomplex * Block = storage + kappa2index[ gKappa( NL, TwoSL, IL, NR, TwoSR, IR ) ];

                        char cotrans  = 'C';
                        char notrans  = 'N';
                        dcomplex one  = 1.0;
                        dcomplex beta = ( firsttime ) ? 0.0 : 1.0;
                        zgemm_( &cotrans, &notrans, &dimR, &dimR, &dimL, &one, Block, &dimL, Block, &dimL, &beta, result, &dimR );

                        firsttime = false;
                     }
                  }
               }

               for ( int cnt = 0; cnt < dimR; cnt++ ) {
                  result[ ( dimR + 1 ) * cnt ] -= 1.0;
               }

               char norm      = 'F'; // Frobenius norm
               char uplo      = 'U'; // Doesn't matter as result is fully filled
               double TwoNorm = zlansy_( &norm, &uplo, &dimR, result, &dimR, result );

               if ( TwoNorm > CheMPS2::TENSORT_orthoComparison )
                  isLeftNormal = false;
               delete[] result;
            }
         }
      }
   }

   return isLeftNormal;
}

bool CheMPS2::CTensorT::CheckRightNormal() const {
   bool isRightNormal = true;

   for ( int NL = denBK->gNmin( index ); NL <= denBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( index, NL ); TwoSL <= denBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {

            int dimL = denBK->gCurrentDim( index, NL, TwoSL, IL );

            if ( dimL > 0 ) {
               dcomplex * result = new dcomplex[ dimL * dimL ];
               bool firsttime    = true;
               for ( int NR = NL; NR <= NL + 2; NR++ ) {
                  for ( int TwoSR = TwoSL - ( ( NR == NL + 1 ) ? 1 : 0 ); TwoSR < TwoSL + 2; TwoSR += 2 ) {
                     int IR   = ( NR == NL + 1 ) ? ( Irreps::directProd( denBK->gIrrep( index ), IL ) ) : IL;
                     int dimR = denBK->gCurrentDim( index + 1, NR, TwoSR, IR );
                     if ( dimR > 0 ) {
                        dcomplex * Block = storage + kappa2index[ gKappa( NL, TwoSL, IL, NR, TwoSR, IR ) ];

                        char cotrans   = 'C';
                        char notrans   = 'N';
                        dcomplex alpha = ( TwoSR + 1.0 ) / ( TwoSL + 1.0 );
                        dcomplex beta  = ( firsttime ) ? 0.0 : 1.0;
                        zgemm_( &notrans, &cotrans, &dimL, &dimL, &dimR, &alpha, Block, &dimL, Block, &dimL, &beta, result, &dimL );

                        firsttime = false;
                     }
                  }
               }
               for ( int cnt = 0; cnt < dimL; cnt++ ) {
                  result[ ( dimL + 1 ) * cnt ] -= 1.0;
               }

               char norm      = 'F'; // Frobenius norm
               char uplo      = 'U'; // Doesn't matter as result is fully filled
               double TwoNorm = zlansy_( &norm, &uplo, &dimL, result, &dimL, result );

               if ( TwoNorm > CheMPS2::TENSORT_orthoComparison )
                  isRightNormal = false;
               delete[] result;
            }
         }
      }
   }

   return isRightNormal;
}

std::ostream & CheMPS2::operator<<( std::ostream & os, const CheMPS2::CTensorT & tns ) {
   os << CheMPS2::hashline;

   os << "CTensorT with " << tns.gNKappa() << " symmetry blocks: " << std::endl;
   for ( int ikappa = 0; ikappa < tns.gNKappa(); ++ikappa ) {
      const int NL    = tns.sectorNL[ ikappa ];
      const int TwoSL = tns.sectorTwoSL[ ikappa ];
      const int IL    = tns.sectorIL[ ikappa ];
      const int NR    = tns.sectorNR[ ikappa ];
      const int TwoSR = tns.sectorTwoSR[ ikappa ];
      const int IR    = tns.sectorIR[ ikappa ];

      os << "Block number " << ikappa << " with left size "
         << tns.denBK->gCurrentDim( tns.index, NL, TwoSL, IL ) << " and rightsize "
         << tns.denBK->gCurrentDim( tns.index + 1, NR, TwoSR, IR ) << std::endl;
      os << "NL:    " << NL << ", "
         << "NR:    " << NR << std::endl;
      os << "TwoSL: " << TwoSL << ", "
         << "TwoSR: " << TwoSR << std::endl;
      os << "IL:    " << IL << ", "
         << "IR:    " << IR << std::endl;

      for ( int i = tns.kappa2index[ ikappa ]; i < tns.kappa2index[ ikappa + 1 ]; ++i ) {
         os << tns.storage[ i ] << std::endl;
      }
   }
   os << CheMPS2::hashline;
   return os;
}

void recusion( CheMPS2::Problem * prob, CheMPS2::CTensorT ** mps,
               std::vector< int > alphas, std::vector< int > betas, int L,
               std::vector< std::vector< int > > & alphasOut,
               std::vector< std::vector< int > > & betasOut,
               std::vector< double > & coefsRealOut,
               std::vector< double > & coefsImagOut ) {

   int sumAlpha = 0;
   int sumBeta  = 0;
   for ( int i = 0; i < alphas.size(); i++ ) {
      sumAlpha += alphas[ i ];
   }
   for ( int i = 0; i < betas.size(); i++ ) {
      sumBeta += betas[ i ];
   }
   if ( sumAlpha + sumBeta > prob->gN() ) {
      return;
   }

   if ( alphas.size() == L && betas.size() == L ) {
      if ( sumAlpha + sumBeta == prob->gN() ) {
         // for ( int i = 0; i < L; i++ ) {
         //    std::cout << alphas[ i ] << " ";
         // }
         // for ( int i = 0; i < L; i++ ) {
         //    std::cout << betas[ i ] << " ";
         // }
         // dcomplex coef = getFCICoefficient( prob, mps, &alphas[ 0 ], &betas[ 0 ] );
         alphasOut.push_back( alphas );
         betasOut.push_back( betas );
         coefsRealOut.push_back( std::real( getFCICoefficient( prob, mps, &alphas[ 0 ], &betas[ 0 ] ) ) );
         coefsImagOut.push_back( std::imag( getFCICoefficient( prob, mps, &alphas[ 0 ], &betas[ 0 ] ) ) );
         // std::cout << std::real( coef ) << " " << std::imag( coef ) << std::endl;
      }
   } else if ( alphas.size() == L ) {
      std::vector< int > caseA = betas;
      caseA.push_back( 0 );
      recusion( prob, mps, alphas, caseA, L, alphasOut, betasOut, coefsRealOut, coefsImagOut );
      std::vector< int > caseB = betas;
      caseB.push_back( 1 );
      recusion( prob, mps, alphas, caseB, L, alphasOut, betasOut, coefsRealOut, coefsImagOut );
   } else {
      std::vector< int > caseA = alphas;
      caseA.push_back( 0 );
      recusion( prob, mps, caseA, betas, L, alphasOut, betasOut, coefsRealOut, coefsImagOut );
      std::vector< int > caseB = alphas;
      caseB.push_back( 1 );
      recusion( prob, mps, caseB, betas, L, alphasOut, betasOut, coefsRealOut, coefsImagOut );
   }
}

void CheMPS2::getFCITensor( Problem * prob, CTensorT ** mps,
                            std::vector< std::vector< int > > & alphasOut,
                            std::vector< std::vector< int > > & betasOut,
                            std::vector< double > & coefsRealOut,
                            std::vector< double > & coefsImagOut ) {

   std::vector< int > alphas;
   std::vector< int > betas;

   recusion( prob, mps, alphas, betas, prob->gL(), alphasOut, betasOut, coefsRealOut, coefsImagOut );
}

void CheMPS2::printFCITensor( Problem * prob, CTensorT ** mps ) {

   std::vector< int > alphas;
   std::vector< int > betas;
   std::vector< std::vector< int > > alphasOut;
   std::vector< std::vector< int > > betasOut;
   std::vector< double > coefsRealOut;
   std::vector< double > coefsImagOut;

   recusion( prob, mps, alphas, betas, prob->gL(), alphasOut, betasOut, coefsRealOut, coefsImagOut );

   for ( int coef = 0; coef < coefsRealOut.size(); coef++ ) {
      for ( int i = 0; i < prob->gL(); i++ ) {
         std::cout << alphasOut[ coef ][ i ] << " ";
      }
      for ( int i = 0; i < prob->gL(); i++ ) {
         std::cout << betasOut[ coef ][ i ] << " ";
      }
      std::cout << coefsRealOut[ coef ] << " " << coefsImagOut[ coef ] << std::endl;
   }
}

dcomplex CheMPS2::getFCICoefficient( Problem * prob, CTensorT ** mps, int * alpha, int * beta ) {
   const SyBookkeeper * denBK = mps[ 0 ]->gBK();
   int L                      = mps[ 0 ]->gBK()->gL();
   //DMRGcoeff = alpha/beta[Hamindex = Prob->gf2(DMRGindex)]

   //Check if it's possible
   {
      int nTot  = 0;
      int twoSz = 0;
      int iTot  = 0;
      for ( int DMRGindex = 0; DMRGindex < L; DMRGindex++ ) {
         const int HamIndex = DMRGindex;
         assert( ( alpha[ HamIndex ] == 0 ) || ( alpha[ HamIndex ] == 1 ) );
         assert( ( beta[ HamIndex ] == 0 ) || ( beta[ HamIndex ] == 1 ) );
         nTot += alpha[ HamIndex ] + beta[ HamIndex ];
         twoSz += alpha[ HamIndex ] - beta[ HamIndex ];
         if ( ( alpha[ HamIndex ] + beta[ HamIndex ] ) == 1 ) { iTot = Irreps::directProd( iTot, denBK->gIrrep( DMRGindex ) ); }
      }
      if ( prob->gN() != nTot ) {
         // std::cout << "DMRG::getFCIcoefficient : Ndesired = " << prob->gN() << " and Ntotal in alpha and beta strings = " << nTot << std::endl;
         return 0.0;
      }
      // 2Sz can be -Prob->2S() ; -Prob->2S()+2 ; -Prob->2S()+4 ; ... ; Prob->2S()
      if ( ( prob->gTwoS() < twoSz ) || ( twoSz < -prob->gTwoS() ) || ( ( prob->gTwoS() - twoSz ) % 2 != 0 ) ) {
         // std::cout << "DMRG::getFCIcoefficient : 2Sdesired = " << prob->gTwoS() << " and 2Sz in alpha and beta strings = " << twoSz << std::endl;
         return 0.0;
      }
      if ( prob->gIrrep() != iTot ) {
         // std::cout << "DMRG::getFCIcoefficient : Idesired = " << prob->gIrrep() << " and Irrep of alpha and beta strings = " << iTot << std::endl;
         return 0.0;
      }
   }

   dcomplex theCoeff = 2.0; // A FCI coefficient always lies in between -1.0 and 1.0
#ifdef CHEMPS2_MPI_COMPILATION
   if ( ( MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER ) || ( mpi_chemps2_master_only == false ) )
#endif
   {

      //Construct necessary arrays
      int Dmax = 1;
      for ( int DMRGindex = 1; DMRGindex < L; DMRGindex++ ) {
         const int DtotBound = denBK->gTotDimAtBound( DMRGindex );
         if ( DtotBound > Dmax ) { Dmax = DtotBound; }
      }
      dcomplex * arrayL = new dcomplex[ Dmax ];
      dcomplex * arrayR = new dcomplex[ Dmax ];
      int * twoSL       = new int[ L ];
      int * twoSR       = new int[ L ];
      int * jumpL       = new int[ L + 1 ];
      int * jumpR       = new int[ L + 1 ];

      //Start the iterator
      int num_SL          = 0;
      jumpL[ num_SL ]     = 0;
      int dimFirst        = 1;
      jumpL[ num_SL + 1 ] = jumpL[ num_SL ] + dimFirst;
      twoSL[ num_SL ]     = 0;
      num_SL++;
      arrayL[ 0 ] = 1.0;
      int NL      = 0;
      int IL      = 0;
      int twoSLz  = 0;

      for ( int DMRGindex = 0; DMRGindex < L; DMRGindex++ ) {

         //Clear the right array
         for ( int count = 0; count < Dmax; count++ ) {
            arrayR[ count ] = 0.0;
         }

         //The local occupation
         const int HamIndex = ( prob->gReorder() ) ? prob->gf2( DMRGindex ) : DMRGindex;
         const int Nlocal   = alpha[ HamIndex ] + beta[ HamIndex ];
         const int twoSzloc = alpha[ HamIndex ] - beta[ HamIndex ];

         //The right symmetry sectors
         const int NR     = NL + Nlocal;
         const int twoSRz = twoSLz + twoSzloc;
         const int IR     = ( ( Nlocal == 1 ) ? ( Irreps::directProd( IL, denBK->gIrrep( DMRGindex ) ) ) : IL );

         int num_SR       = 0;
         jumpR[ num_SR ]  = 0;
         const int spread = ( ( Nlocal == 1 ) ? 1 : 0 );
         for ( int cntSL = 0; cntSL < num_SL; cntSL++ ) {
            for ( int TwoSRattempt = twoSL[ cntSL ] - spread; TwoSRattempt <= twoSL[ cntSL ] + spread; TwoSRattempt += 2 ) {
               bool encountered = false;
               for ( int cntSR = 0; cntSR < num_SR; cntSR++ ) {
                  if ( twoSR[ cntSR ] == TwoSRattempt ) {
                     encountered = true;
                  }
               }
               if ( encountered == false ) {
                  const int dimR = denBK->gCurrentDim( DMRGindex + 1, NR, TwoSRattempt, IR );
                  if ( dimR > 0 ) {
                     jumpR[ num_SR + 1 ] = jumpR[ num_SR ] + dimR;
                     twoSR[ num_SR ]     = TwoSRattempt;
                     num_SR++;
                  }
               }
            }
         }
         assert( jumpR[ num_SR ] <= Dmax );

         for ( int cntSR = 0; cntSR < num_SR; cntSR++ ) {
            int TwoSRvalue = twoSR[ cntSR ];
            int dimR       = jumpR[ cntSR + 1 ] - jumpR[ cntSR ];
            for ( int TwoSLvalue = TwoSRvalue - spread; TwoSLvalue <= TwoSRvalue + spread; TwoSLvalue += 2 ) {

               int indexSL = -1;
               for ( int cntSL = 0; cntSL < num_SL; cntSL++ ) {
                  if ( twoSL[ cntSL ] == TwoSLvalue ) {
                     indexSL = cntSL;
                     cntSL   = num_SL; //exit loop
                  }
               }
               if ( indexSL != -1 ) {
                  int dimL           = jumpL[ indexSL + 1 ] - jumpL[ indexSL ];
                  dcomplex * Tblock  = mps[ DMRGindex ]->gStorage( NL, TwoSLvalue, IL, NR, TwoSRvalue, IR );
                  dcomplex prefactor = sqrt( TwoSRvalue + 1 ) * Wigner::wigner3j( TwoSLvalue, spread, TwoSRvalue, twoSLz, twoSzloc, -twoSRz ) * Special::phase( -TwoSLvalue + spread - twoSRz );
                  dcomplex add2array = 1.0;
                  char notrans       = 'N';
                  zgemm_( &notrans, &notrans, &dimFirst, &dimR, &dimL, &prefactor, arrayL + jumpL[ indexSL ], &dimFirst, Tblock, &dimL, &add2array, arrayR + jumpR[ cntSR ], &dimFirst );
               }
            }
         }

         //Swap L <--> R
         {
            dcomplex * temp = arrayR;
            arrayR          = arrayL;
            arrayL          = temp;
            int * temp2     = twoSR;
            twoSR           = twoSL;
            twoSL           = temp2;
            temp2           = jumpR;
            jumpR           = jumpL;
            jumpL           = temp2;
            num_SL          = num_SR;
            NL              = NR;
            IL              = IR;
            twoSLz          = twoSRz;
         }
      }

      theCoeff = arrayL[ 0 ];

      // assert( num_SL == 1 );
      assert( jumpL[ 1 ] == 1 );
      assert( twoSL[ 0 ] == prob->gTwoS() );
      assert( NL == prob->gN() );
      // assert( IL == prob->gIrrep() );

      delete[] arrayL;
      delete[] arrayR;
      delete[] twoSL;
      delete[] twoSR;
      delete[] jumpL;
      delete[] jumpR;
   }

#ifdef CHEMPS2_MPI_COMPILATION
   if ( mpi_chemps2_master_only ) { MPIchemps2::broadcast_array_double( &theCoeff, 1, MPI_CHEMPS2_MASTER ); }
#endif
   return theCoeff;
}

dcomplex CheMPS2::overlap( CTensorT ** mpsA, CTensorT ** mpsB ) {
   const int L = mpsA[ 0 ]->gBK()->gL();

   CTensorO * overlapOld;
   overlapOld = new CTensorO( L - 1, false, mpsA[ L - 1 ]->gBK(), mpsB[ L - 1 ]->gBK() );
   overlapOld->create( mpsA[ L - 1 ], mpsB[ L - 1 ] );

   CTensorO * overlapNext;
   for ( int i = L - 2; i >= 0; i-- ) {
      overlapNext = new CTensorO( i, false, mpsA[ i ]->gBK(), mpsB[ i ]->gBK() );
      overlapNext->update_ownmem( mpsA[ i ], mpsB[ i ], overlapOld );
      delete overlapOld;
      overlapOld = overlapNext;
   }
   assert( overlapOld->gNKappa() == 1 );
   dcomplex result = overlapOld->trace();
   delete overlapOld;
   return result;
}

double CheMPS2::norm( CTensorT ** mps ) {
   return std::real( sqrt( overlap( mps, mps ) ) );
}

void CheMPS2::left_normalize( CTensorT * left_mps, CTensorT * right_mps ) {

#ifdef CHEPsi2_MPI_COMPILATION
   const bool am_i_master = ( MPIchemps2::mpi_rank() == MPI_CHEPsi2_MASTER );
#else
   const bool am_i_master = true;
#endif

   if ( am_i_master ) {
      const int siteindex        = left_mps->gIndex();
      const SyBookkeeper * theBK = left_mps->gBK();
      // (J,N,I) = (0,0,0) and (moving_right, prime_last, jw_phase) = (true, true, false)
      CTensorOperator * temp = new CTensorOperator( siteindex + 1, 0, 0, 0, true, true, false, theBK, theBK );
      left_mps->QR( temp );
      char notrans = 'N';
      if ( right_mps != NULL ) { right_mps->LeftMultiply( temp, &notrans ); }
      delete temp;
   }
#ifdef CHEPsi2_MPI_COMPILATION
   MPIchemps2::broadcast_tensor( left_mps, MPI_CHEPsi2_MASTER );
   if ( right_mps != NULL ) { MPIchemps2::broadcast_tensor( right_mps, MPI_CHEPsi2_MASTER ); }
#endif
}

void CheMPS2::right_normalize( CTensorT * left_mps, CTensorT * right_mps ) {

#ifdef CHEPsi2_MPI_COMPILATION
   const bool am_i_master = ( MPIchemps2::mpi_rank() == MPI_CHEPsi2_MASTER );
#else
   const bool am_i_master = true;
#endif

   if ( am_i_master ) {
      const int siteindex        = right_mps->gIndex();
      const SyBookkeeper * theBK = right_mps->gBK();
      // (J,N,I) = (0,0,0) and (moving_right, prime_last, jw_phase) = (true, true, false)
      CTensorOperator * temp = new CTensorOperator( siteindex, 0, 0, 0, true, true, false, theBK, theBK );
      right_mps->LQ( temp );
      char cotrans = 'C';
      if ( left_mps != NULL ) { left_mps->RightMultiply( temp, &cotrans ); }
      delete temp;
   }
#ifdef CHEPsi2_MPI_COMPILATION
   MPIchemps2::broadcast_tensor( right_mps, MPI_CHEPsi2_MASTER );
   if ( left_mps != NULL ) { MPIchemps2::broadcast_tensor( left_mps, MPI_CHEPsi2_MASTER ); }
#endif
}

void CheMPS2::decomposeMovingLeft( bool change, int virtualdimensionD, double cut_off,
                                   CTensorT * expandedLeft, SyBookkeeper * expandedLeftBK,
                                   CTensorT * expandedRight, SyBookkeeper * expandedRightBK,
                                   CTensorT * newLeft, SyBookkeeper * newLeftBK,
                                   CTensorT * newRight, SyBookkeeper * newRightBK ) {
   assert( expandedLeftBK == expandedRightBK );
   assert( newLeftBK == newRightBK );
   assert( expandedLeft->gIndex() == newLeft->gIndex() );
   assert( expandedRight->gIndex() == newRight->gIndex() );
   assert( expandedLeft->gIndex() + 1 == expandedRight->gIndex() );
   const int index = expandedRight->gIndex();

   int nMiddleSectors = 0;
   for ( int NM = expandedRightBK->gNmin( index ); NM <= expandedRightBK->gNmax( index ); NM++ ) {
      for ( int TwoSM = expandedRightBK->gTwoSmin( index, NM ); TwoSM <= expandedRightBK->gTwoSmax( index, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < expandedRightBK->getNumberOfIrreps(); IM++ ) {
            int dimM = expandedRightBK->gCurrentDim( index, NM, TwoSM, IM );
            if ( dimM > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NM == expandedRight->gNL( ikappa ) ) && ( TwoSM == expandedRight->gTwoSL( ikappa ) ) && ( IM == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  nMiddleSectors++;
               }
            }
         }
      }
   }

   int * SplitSectNM    = new int[ nMiddleSectors ];
   int * SplitSectTwoJM = new int[ nMiddleSectors ];
   int * SplitSectIM    = new int[ nMiddleSectors ];
   int * DimLs          = new int[ nMiddleSectors ];
   int * DimMs          = new int[ nMiddleSectors ];
   int * DimRs          = new int[ nMiddleSectors ];

   nMiddleSectors = 0;
   for ( int NM = expandedRightBK->gNmin( index ); NM <= expandedRightBK->gNmax( index ); NM++ ) {
      for ( int TwoSM = expandedRightBK->gTwoSmin( index, NM ); TwoSM <= expandedRightBK->gTwoSmax( index, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < expandedRightBK->getNumberOfIrreps(); IM++ ) {
            int dimM = expandedRightBK->gCurrentDim( index, NM, TwoSM, IM );
            if ( dimM > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NM == expandedRight->gNL( ikappa ) ) && ( TwoSM == expandedRight->gTwoSL( ikappa ) ) && ( IM == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  SplitSectNM[ nMiddleSectors ]    = NM;
                  SplitSectTwoJM[ nMiddleSectors ] = TwoSM;
                  SplitSectIM[ nMiddleSectors ]    = IM;
                  DimLs[ nMiddleSectors ]          = dimM;
                  DimMs[ nMiddleSectors ]          = std::min( dimM, dimRtotal );
                  DimRs[ nMiddleSectors ]          = dimRtotal;
                  nMiddleSectors++;
               }
            }
         }
      }
   }

   double ** Lambdas = NULL;
   dcomplex ** Us    = NULL;
   dcomplex ** VTs   = NULL;

   Lambdas = new double *[ nMiddleSectors ];
   Us      = new dcomplex *[ nMiddleSectors ];
   VTs     = new dcomplex *[ nMiddleSectors ];

   for ( int iMiddleSector = 0; iMiddleSector < nMiddleSectors; iMiddleSector++ ) {

      Lambdas[ iMiddleSector ] = new double[ DimMs[ iMiddleSector ] ];
      Us[ iMiddleSector ]      = new dcomplex[ DimLs[ iMiddleSector ] * DimMs[ iMiddleSector ] ];
      VTs[ iMiddleSector ]     = new dcomplex[ DimMs[ iMiddleSector ] * DimRs[ iMiddleSector ] ];

      // Copy the relevant parts from storage to mem & multiply with factor !!
      dcomplex * mem = new dcomplex[ DimRs[ iMiddleSector ] * DimLs[ iMiddleSector ] ];
      int dimRtotal2 = 0;
      for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
         if ( ( SplitSectNM[ iMiddleSector ] == expandedRight->gNL( ikappa ) ) && ( SplitSectTwoJM[ iMiddleSector ] == expandedRight->gTwoSL( ikappa ) ) && ( SplitSectIM[ iMiddleSector ] == expandedRight->gIL( ikappa ) ) ) {
            int dimR = expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
            if ( dimR > 0 ) {
               double factor = sqrt( ( expandedRight->gTwoSR( ikappa ) + 1.0 ) / ( SplitSectTwoJM[ iMiddleSector ] + 1.0 ) );
               for ( int l = 0; l < DimLs[ iMiddleSector ]; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     dcomplex * storage                                     = expandedRight->gStorage() + expandedRight->gKappa2index( ikappa );
                     mem[ l + DimLs[ iMiddleSector ] * ( dimRtotal2 + r ) ] = factor * storage[ l + DimLs[ iMiddleSector ] * r ];
                  }
               }
               dimRtotal2 += dimR;
            }
         }
      }

      // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
      // central symmetry
      char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
      int lwork       = DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + 3 * DimMs[ iMiddleSector ];
      int lrwork      = std::max( 5 * DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + 5 * DimMs[ iMiddleSector ], 2 * std::max( DimLs[ iMiddleSector ], DimRs[ iMiddleSector ] ) * DimMs[ iMiddleSector ] + 2 * DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + DimMs[ iMiddleSector ] );
      dcomplex * work = new dcomplex[ lwork ];
      double * rwork  = new double[ lrwork ];
      int * iwork     = new int[ 8 * DimMs[ iMiddleSector ] ];
      int info;

      // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
      zgesdd_( &jobz, DimLs + iMiddleSector, DimRs + iMiddleSector, mem, DimLs + iMiddleSector,
               Lambdas[ iMiddleSector ], Us[ iMiddleSector ], DimLs + iMiddleSector, VTs[ iMiddleSector ],
               DimMs + iMiddleSector, work, &lwork, rwork, iwork, &info );

      delete[] work;
      delete[] rwork;
      delete[] iwork;
      delete[] mem;
   }

   double discardedWeight = 0.0; // Only if change==true; will the discardedWeight be meaningful and different from zero.
   int updateSectors      = 0;
   int * NewDims          = NULL;

   // If change: determine new virtual dimensions.
   if ( change ) {
      NewDims = new int[ nMiddleSectors ];

      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         NewDims[ iSector ] = DimMs[ iSector ];
         totalDimSVD += NewDims[ iSector ];
      }

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         if ( NewDims[ iSector ] > 0 ) {
            dcopy_( NewDims + iSector, Lambdas[ iSector ], &inc, values + totalDimSVD, &inc );
            totalDimSVD += NewDims[ iSector ];
         }
      }

      // Sort them in decreasing order
      char ID = 'D';
      int info;
      dlasrt_( &ID, &totalDimSVD, values, &info ); // Quicksort

      int maxD = 0;
      while ( maxD < totalDimSVD && maxD < virtualdimensionD && cut_off < values[ maxD ] ) {
         maxD++;
      }

      // int maxD = virtualdimensionD;
      // If larger then the required virtualdimensionD, new virtual dimensions
      // will be set in NewDims.
      if ( totalDimSVD > maxD ) {

         // The D+1'th value becomes the lower bound Schmidt value. Every value
         // smaller than or equal to the D+1'th value is thrown out (hence Dactual // <= Ddesired).
         const double lowerBound = values[ maxD ];
         for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
            for ( int cnt = 0; cnt < NewDims[ iSector ]; cnt++ ) {
               if ( Lambdas[ iSector ][ cnt ] <= lowerBound ) {
                  NewDims[ iSector ] = cnt;
               }
            }
         }

         // Discarded weight
         double totalSum     = 0.0;
         double discardedSum = 0.0;
         for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
            for ( int iLocal = 0; iLocal < DimMs[ iSector ]; iLocal++ ) {
               double temp = ( expandedRight->gTwoSL( iSector ) + 1 ) * Lambdas[ iSector ][ iLocal ] * Lambdas[ iSector ][ iLocal ];
               totalSum += temp;
               if ( Lambdas[ iSector ][ iLocal ] <= lowerBound ) {
                  discardedSum += temp;
               }
            }
         }
         discardedWeight = discardedSum / totalSum;
      }
      // Clean-up
      delete[] values;

      // Check if there is a sector which differs
      updateSectors = 0;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         const int MPSdim = expandedRightBK->gCurrentDim( index, SplitSectNM[ iSector ], SplitSectTwoJM[ iSector ], SplitSectIM[ iSector ] );
         if ( NewDims[ iSector ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }

   if ( updateSectors == 1 ) {
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         newLeftBK->SetDim( index, SplitSectNM[ iSector ], SplitSectTwoJM[ iSector ], SplitSectIM[ iSector ], NewDims[ iSector ] );
      }
      newRight->Reset();
      newLeft->Reset();
   }

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

   newRight->Clear();
   newLeft->Clear();

   // Copy first dimM per central symmetry sector to the relevant parts
   for ( int iCenter = 0; iCenter < nMiddleSectors; iCenter++ ) {

      int dimLtotal2 = 0;
      for ( int ikappa = 0; ikappa < newLeft->gNKappa(); ikappa++ ) {

         const int NL    = newLeft->gNL( ikappa );
         const int TwoSL = newLeft->gTwoSL( ikappa );
         const int IL    = newLeft->gIL( ikappa );

         const int NR    = newLeft->gNR( ikappa );
         const int TwoSR = newLeft->gTwoSR( ikappa );
         const int IR    = newLeft->gIR( ikappa );

         if ( ( SplitSectNM[ iCenter ] == NR ) && ( SplitSectTwoJM[ iCenter ] == TwoSR ) && ( SplitSectIM[ iCenter ] == IR ) ) {
            const int dimL = newLeftBK->gCurrentDim( index - 1, NL, TwoSL, IL );
            const int dimM = expandedLeftBK->gCurrentDim( index, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
            if ( dimL > 0 ) {
               dcomplex * TleftOld = expandedLeft->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
               dcomplex * TleftNew = newLeft->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               const int dimension_limit_right = std::min( dimM, DimMs[ iCenter ] );
               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = 0; r < dimension_limit_right; r++ ) {
                     TleftNew[ l + dimL * r ] = 0.0;
                     for ( int m = 0; m < dimM; m++ ) {
                        TleftNew[ l + dimL * r ] += TleftOld[ l + dimL * m ] * Us[ iCenter ][ m + DimMs[ iCenter ] * r ] * Lambdas[ iCenter ][ r ];
                     }
                  }
               }
               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = dimension_limit_right; r < dimM; r++ ) {
                     TleftNew[ l + dimL * r ] = 0.0;
                  }
               }
               dimLtotal2 += dimL;
            }
         }
      }

      // Copy from mem to storage & multiply with factor !!
      int dimRtotal2 = 0;
      for ( int ikappa = 0; ikappa < newRight->gNKappa(); ikappa++ ) {

         const int NL    = newRight->gNL( ikappa );
         const int TwoSL = newRight->gTwoSL( ikappa );
         const int IL    = newRight->gIL( ikappa );

         const int NR    = newRight->gNR( ikappa );
         const int TwoSR = newRight->gTwoSR( ikappa );
         const int IR    = newRight->gIR( ikappa );

         if ( ( SplitSectNM[ iCenter ] == NL ) && ( SplitSectTwoJM[ iCenter ] == TwoSL ) && ( SplitSectIM[ iCenter ] == IL ) ) {
            int dimR = newRightBK->gCurrentDim( index + 1, NR, TwoSR, IR );
            if ( dimR > 0 ) {
               double factor      = sqrt( ( SplitSectTwoJM[ iCenter ] + 1.0 ) / ( TwoSR + 1.0 ) );
               dcomplex * storage = newRight->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               for ( int l = 0; l < DimMs[ iCenter ]; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     storage[ l + DimLs[ iCenter ] * r ] = factor * VTs[ iCenter ][ l + DimMs[ iCenter ] * ( r + dimRtotal2 ) ];
                  }
               }
               dimRtotal2 += dimR;
            }
         }
      }
   }
}

void CheMPS2::decomposeMovingRight( bool change, int virtualdimensionD, double cut_off,
                                    CTensorT * expandedLeft, SyBookkeeper * expandedLeftBK,
                                    CTensorT * expandedRight, SyBookkeeper * expandedRightBK,
                                    CTensorT * newLeft, SyBookkeeper * newLeftBK,
                                    CTensorT * newRight, SyBookkeeper * newRightBK ) {
   assert( expandedLeftBK == expandedRightBK );
   assert( newLeftBK == newRightBK );
   assert( expandedLeft->gIndex() == newLeft->gIndex() );
   assert( expandedRight->gIndex() == newRight->gIndex() );
   assert( expandedLeft->gIndex() + 1 == expandedRight->gIndex() );
   const int index = expandedLeft->gIndex();

   int nMiddleSectors = 0;
   for ( int NM = expandedLeftBK->gNmin( index + 1 ); NM <= expandedLeftBK->gNmax( index + 1 ); NM++ ) {
      for ( int TwoSM = expandedLeftBK->gTwoSmin( index + 1, NM ); TwoSM <= expandedLeftBK->gTwoSmax( index + 1, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < expandedLeftBK->getNumberOfIrreps(); IM++ ) {
            int dimM = expandedLeftBK->gCurrentDim( index + 1, NM, TwoSM, IM );
            if ( dimM > 0 ) {
               int dimLtotal = 0;
               for ( int ikappa = 0; ikappa < expandedLeft->gNKappa(); ikappa++ ) {
                  if ( ( NM == expandedLeft->gNR( ikappa ) ) && ( TwoSM == expandedLeft->gTwoSR( ikappa ) ) && ( IM == expandedLeft->gIR( ikappa ) ) ) {
                     dimLtotal += expandedLeftBK->gCurrentDim( index, expandedLeft->gNL( ikappa ), expandedLeft->gTwoSL( ikappa ), expandedRight->gIL( ikappa ) );
                  }
               }
               if ( dimLtotal > 0 ) {
                  nMiddleSectors++;
               }
            }
         }
      }
   }

   int * SplitSectNM    = new int[ nMiddleSectors ];
   int * SplitSectTwoJM = new int[ nMiddleSectors ];
   int * SplitSectIM    = new int[ nMiddleSectors ];
   int * DimLs          = new int[ nMiddleSectors ];
   int * DimMs          = new int[ nMiddleSectors ];
   int * DimRs          = new int[ nMiddleSectors ];

   nMiddleSectors = 0;
   for ( int NM = expandedLeftBK->gNmin( index + 1 ); NM <= expandedLeftBK->gNmax( index + 1 ); NM++ ) {
      for ( int TwoSM = expandedLeftBK->gTwoSmin( index + 1, NM ); TwoSM <= expandedLeftBK->gTwoSmax( index + 1, NM ); TwoSM += 2 ) {
         for ( int IM = 0; IM < expandedLeftBK->getNumberOfIrreps(); IM++ ) {
            int dimM = expandedLeftBK->gCurrentDim( index + 1, NM, TwoSM, IM );
            if ( dimM > 0 ) {
               int dimLtotal = 0;
               for ( int ikappa = 0; ikappa < expandedLeft->gNKappa(); ikappa++ ) {
                  if ( ( NM == expandedLeft->gNR( ikappa ) ) && ( TwoSM == expandedLeft->gTwoSR( ikappa ) ) && ( IM == expandedLeft->gIR( ikappa ) ) ) {
                     dimLtotal += expandedLeftBK->gCurrentDim( index, expandedLeft->gNL( ikappa ), expandedLeft->gTwoSL( ikappa ), expandedLeft->gIL( ikappa ) );
                  }
               }
               if ( dimLtotal > 0 ) {
                  SplitSectNM[ nMiddleSectors ]    = NM;
                  SplitSectTwoJM[ nMiddleSectors ] = TwoSM;
                  SplitSectIM[ nMiddleSectors ]    = IM;
                  DimLs[ nMiddleSectors ]          = dimLtotal;
                  DimMs[ nMiddleSectors ]          = std::min( dimM, dimLtotal );
                  DimRs[ nMiddleSectors ]          = dimM;
                  nMiddleSectors++;
               }
            }
         }
      }
   }
   double ** Lambdas = NULL;
   dcomplex ** Us    = NULL;
   dcomplex ** VTs   = NULL;

   Lambdas = new double *[ nMiddleSectors ];
   Us      = new dcomplex *[ nMiddleSectors ];
   VTs     = new dcomplex *[ nMiddleSectors ];

   for ( int iMiddleSector = 0; iMiddleSector < nMiddleSectors; iMiddleSector++ ) {

      Lambdas[ iMiddleSector ] = new double[ DimMs[ iMiddleSector ] ];
      Us[ iMiddleSector ]      = new dcomplex[ DimLs[ iMiddleSector ] * DimMs[ iMiddleSector ] ];
      VTs[ iMiddleSector ]     = new dcomplex[ DimMs[ iMiddleSector ] * DimRs[ iMiddleSector ] ];

      // Copy the relevant parts from storage to mem & multiply with factor !!
      dcomplex * mem = new dcomplex[ DimLs[ iMiddleSector ] * DimRs[ iMiddleSector ] ];
      int dimLtotal2 = 0;
      for ( int ikappa = 0; ikappa < expandedLeft->gNKappa(); ikappa++ ) {
         if ( ( SplitSectNM[ iMiddleSector ] == expandedLeft->gNR( ikappa ) ) && ( SplitSectTwoJM[ iMiddleSector ] == expandedLeft->gTwoSR( ikappa ) ) && ( SplitSectIM[ iMiddleSector ] == expandedLeft->gIR( ikappa ) ) ) {
            int dimL = expandedLeftBK->gCurrentDim( index, expandedLeft->gNL( ikappa ), expandedLeft->gTwoSL( ikappa ), expandedLeft->gIL( ikappa ) );
            if ( dimL > 0 ) {
               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = 0; r < DimRs[ iMiddleSector ]; r++ ) {
                     dcomplex * storage                                 = expandedLeft->gStorage() + expandedLeft->gKappa2index( ikappa );
                     mem[ dimLtotal2 + l + DimLs[ iMiddleSector ] * r ] = storage[ l + dimL * r ];
                  }
               }
               dimLtotal2 += dimL;
            }
         }
      }
      // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
      // central symmetry
      char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
      int lwork       = DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + 3 * DimMs[ iMiddleSector ];
      int lrwork      = std::max( 5 * DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + 5 * DimMs[ iMiddleSector ], 2 * std::max( DimLs[ iMiddleSector ], DimRs[ iMiddleSector ] ) * DimMs[ iMiddleSector ] + 2 * DimMs[ iMiddleSector ] * DimMs[ iMiddleSector ] + DimMs[ iMiddleSector ] );
      dcomplex * work = new dcomplex[ lwork ];
      double * rwork  = new double[ lrwork ];
      int * iwork     = new int[ 8 * DimMs[ iMiddleSector ] ];
      int info;

      // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
      zgesdd_( &jobz, DimLs + iMiddleSector, DimRs + iMiddleSector, mem, DimLs + iMiddleSector,
               Lambdas[ iMiddleSector ], Us[ iMiddleSector ], DimLs + iMiddleSector, VTs[ iMiddleSector ],
               DimMs + iMiddleSector, work, &lwork, rwork, iwork, &info );

      delete[] work;
      delete[] rwork;
      delete[] iwork;
      delete[] mem;
   }

   double discardedWeight = 0.0; // Only if change==true; will the discardedWeight be meaningful and different from zero.
   int updateSectors      = 0;
   int * NewDims          = NULL;

   // If change: determine new virtual dimensions.
   if ( change ) {
      NewDims = new int[ nMiddleSectors ];

      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         NewDims[ iSector ] = DimMs[ iSector ];
         totalDimSVD += NewDims[ iSector ];
      }

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         if ( NewDims[ iSector ] > 0 ) {
            dcopy_( NewDims + iSector, Lambdas[ iSector ], &inc, values + totalDimSVD, &inc );
            totalDimSVD += NewDims[ iSector ];
         }
      }

      // Sort them in decreasing order
      char ID = 'D';
      int info;
      dlasrt_( &ID, &totalDimSVD, values, &info ); // Quicksort

      int maxD = 0;
      while ( maxD < totalDimSVD && maxD < virtualdimensionD && cut_off < values[ maxD ] ) {
         maxD++;
      }

      // int maxD = virtualdimensionD;
      // If larger then the required virtualdimensionD, new virtual dimensions
      // will be set in NewDims.
      if ( totalDimSVD > maxD ) {

         // The D+1'th value becomes the lower bound Schmidt value. Every value
         // smaller than or equal to the D+1'th value is thrown out (hence Dactual // <= Ddesired).
         const double lowerBound = values[ maxD ];
         for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
            for ( int cnt = 0; cnt < NewDims[ iSector ]; cnt++ ) {
               if ( Lambdas[ iSector ][ cnt ] <= lowerBound ) {
                  NewDims[ iSector ] = cnt;
               }
            }
         }

         // Discarded weight
         double totalSum     = 0.0;
         double discardedSum = 0.0;
         for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
            for ( int iLocal = 0; iLocal < DimMs[ iSector ]; iLocal++ ) {
               double temp = ( expandedLeft->gTwoSL( iSector ) + 1 ) * Lambdas[ iSector ][ iLocal ] * Lambdas[ iSector ][ iLocal ];
               totalSum += temp;
               if ( Lambdas[ iSector ][ iLocal ] <= lowerBound ) {
                  discardedSum += temp;
               }
            }
         }
         discardedWeight = discardedSum / totalSum;
      }
      // Clean-up
      delete[] values;

      // Check if there is a sector which differs
      updateSectors = 0;
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         const int MPSdim = expandedLeftBK->gCurrentDim( index + 1, SplitSectNM[ iSector ], SplitSectTwoJM[ iSector ], SplitSectIM[ iSector ] );
         if ( NewDims[ iSector ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }

   if ( updateSectors == 1 ) {
      for ( int iSector = 0; iSector < nMiddleSectors; iSector++ ) {
         newLeftBK->SetDim( index + 1, SplitSectNM[ iSector ], SplitSectTwoJM[ iSector ], SplitSectIM[ iSector ], NewDims[ iSector ] );
      }
      newLeft->Reset();
      newRight->Reset();
   }

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

   newLeft->Clear();
   newRight->Clear();

   // Copy first dimM per central symmetry sector to the relevant parts
   for ( int iCenter = 0; iCenter < nMiddleSectors; iCenter++ ) {

      // Copy from mem to storage & multiply with factor !!
      int dimLtotal2 = 0;
      for ( int ikappa = 0; ikappa < newLeft->gNKappa(); ikappa++ ) {

         const int NL    = newLeft->gNL( ikappa );
         const int TwoSL = newLeft->gTwoSL( ikappa );
         const int IL    = newLeft->gIL( ikappa );

         const int NR    = newLeft->gNR( ikappa );
         const int TwoSR = newLeft->gTwoSR( ikappa );
         const int IR    = newLeft->gIR( ikappa );

         if ( ( SplitSectNM[ iCenter ] == NR ) && ( SplitSectTwoJM[ iCenter ] == TwoSR ) && ( SplitSectIM[ iCenter ] == IR ) ) {
            int dimL = newLeftBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               dcomplex * storage = newLeft->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = 0; r < DimMs[ iCenter ]; r++ ) {
                     storage[ l + dimL * r ] = Us[ iCenter ][ dimLtotal2 + l + DimLs[ iCenter ] * r ];
                  }
               }
               dimLtotal2 += dimL;
            }
         }
      }

      int dimRtotal2 = 0;
      for ( int ikappa = 0; ikappa < newRight->gNKappa(); ikappa++ ) {

         const int NL    = newRight->gNL( ikappa );
         const int TwoSL = newRight->gTwoSL( ikappa );
         const int IL    = newRight->gIL( ikappa );

         const int NR    = newRight->gNR( ikappa );
         const int TwoSR = newRight->gTwoSR( ikappa );
         const int IR    = newRight->gIR( ikappa );

         if ( ( SplitSectNM[ iCenter ] == NL ) && ( SplitSectTwoJM[ iCenter ] == TwoSL ) && ( SplitSectIM[ iCenter ] == IL ) ) {
            const int dimM = expandedRightBK->gCurrentDim( index + 1, SplitSectNM[ iCenter ], SplitSectTwoJM[ iCenter ], SplitSectIM[ iCenter ] );
            const int dimR = newRightBK->gCurrentDim( index + 2, NR, TwoSR, IR );
            if ( dimR > 0 ) {
               dcomplex * TrightOld = expandedRight->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );
               dcomplex * TrightNew = newRight->gStorage( NL, TwoSL, IL, NR, TwoSR, IR );

               const int dimension_limit_left = std::min( dimM, DimMs[ iCenter ] );
               for ( int l = 0; l < dimension_limit_left; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     TrightNew[ l + DimMs[ iCenter ] * r ] = 0.0;
                     for ( int m = 0; m < dimM; m++ ) {
                        TrightNew[ l + dimM * r ] += Lambdas[ iCenter ][ l ] * VTs[ iCenter ][ l + DimMs[ iCenter ] * m ] * TrightOld[ m + dimM * r ];
                     }
                  }
               }
               for ( int l = dimension_limit_left; l < dimM; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     TrightNew[ l + dimM * r ] = 0.0;
                  }
               }
               dimRtotal2 += dimR;
            }
         }
      }
   }
}
