
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

void CheMPS2::CTensorT::addNoise( dcomplex NoiseLevel ) {
   for ( int cnt = 0; cnt < kappa2index[ nKappa ]; cnt++ ) {
      const dcomplex RN = ( ( double ) rand() ) / RAND_MAX - 0.5;
      storage[ cnt ] += RN * NoiseLevel;
   }   
}

void CheMPS2::CTensorT::add( CTensorT * toAdd, dcomplex factor ) {
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
         dcomplex one = factor;

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
                  int TwoS = ( ( NR == NL + 1 ) ? 1 : 0 );
                  for ( int TwoSR = TwoSL - TwoS; TwoSR <= TwoSL + TwoS; TwoSR += 2 ) {
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
               if ( TwoNorm > CheMPS2::TENSORT_orthoComparison ){
                  // std::cout << denBK << " " << index << std::endl;
                  // std::cout << NL << " " << TwoSL << " " << IL << " " << TwoNorm << " " << dimL << std::endl;
                  isRightNormal = false;
               }
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

   const int L = prob->gL();

   //DMRGcoeff = alpha/beta[Hamindex = Prob->gf2(DMRGindex)]

   //Check if it's possible
   {
      int nTot = 0;
      int twoSz = 0;
      int iTot = 0;
      for (int DMRGindex=0; DMRGindex<L; DMRGindex++){
         const int HamIndex = (prob->gReorder()) ? prob->gf2(DMRGindex) : DMRGindex;
         assert( ( alpha[HamIndex] == 0 ) || ( alpha[HamIndex] == 1 ) );
         assert( (  beta[HamIndex] == 0 ) || (  beta[HamIndex] == 1 ) );
         nTot  += alpha[HamIndex] + beta[HamIndex];
         twoSz += alpha[HamIndex] - beta[HamIndex];
         if ((alpha[HamIndex]+beta[HamIndex])==1){ iTot = Irreps::directProd(iTot,denBK->gIrrep(DMRGindex)); }
      }
      if ( prob->gN() != nTot ){
         // std::cout << "DMRG::getFCIcoefficient : Ndesired = " << prob->gN() << " and Ntotal in alpha and beta strings = " << nTot << std::endl;
         return 0.0;
      }
      // 2Sz can be -prob->2S() ; -prob->2S()+2 ; -prob->2S()+4 ; ... ; prob->2S()
      if ( ( prob->gTwoS() < twoSz ) || ( twoSz < -prob->gTwoS() ) || ( ( prob->gTwoS() - twoSz ) % 2 != 0 ) ){
         // std::cout << "DMRG::getFCIcoefficient : 2Sdesired = " << prob->gTwoS() << " and 2Sz in alpha and beta strings = " << twoSz << std::endl;
         return 0.0;
      }
      if ( prob->gIrrep() != iTot ){
         // std::cout << "DMRG::getFCIcoefficient : Idesired = " << prob->gIrrep() << " and Irrep of alpha and beta strings = " << iTot << std::endl;
         return 0.0;
      }
   }
   
   dcomplex theCoeff = 2.0; // A FCI coefficient always lies in between -1.0 and 1.0
   #ifdef CHEMPS2_MPI_COMPILATION
   if (( MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER ) || ( mpi_chemps2_master_only == false ))
   #endif
   {
   
      //Construct necessary arrays
      int Dmax = 1;
      for (int DMRGindex=1; DMRGindex<L; DMRGindex++){
         const int DtotBound = denBK->gTotDimAtBound(DMRGindex);
         if (DtotBound>Dmax){ Dmax = DtotBound; }
      }
      dcomplex * arrayL = new dcomplex[Dmax];
      dcomplex * arrayR = new dcomplex[Dmax];
      int * twoSL = new int[L];
      int * twoSR = new int[L];
      int * jumpL = new int[L+1];
      int * jumpR = new int[L+1];
      
      //Start the iterator
      int num_SL = 0;
      jumpL[num_SL] = 0;
      int dimFirst = 1;
      jumpL[num_SL+1] = jumpL[num_SL] + dimFirst;
      twoSL[num_SL] = 0;
      num_SL++;
      arrayL[0] = 1.0;
      int NL = 0;
      int IL = 0;
      int twoSLz = 0;
      
      for (int DMRGindex=0; DMRGindex<L; DMRGindex++){
      
         //Clear the right array
         for (int count = 0; count < Dmax; count++){ arrayR[count] = 0.0; }
         
         //The local occupation
         const int HamIndex = (prob->gReorder()) ? prob->gf2(DMRGindex) : DMRGindex;
         const int Nlocal   = alpha[HamIndex] + beta[HamIndex];
         const int twoSzloc = alpha[HamIndex] - beta[HamIndex];
         
         //The right symmetry sectors
         const int NR     = NL + Nlocal;
         const int twoSRz = twoSLz + twoSzloc;
         const int IR     = (( Nlocal == 1 ) ? (Irreps::directProd(IL,denBK->gIrrep(DMRGindex))) : IL);
         
         int num_SR = 0;
         jumpR[num_SR] = 0;
         const int spread = ( ( Nlocal == 1 ) ? 1 : 0 );
         for ( int cntSL = 0; cntSL < num_SL; cntSL++ ){
            for ( int TwoSRattempt = twoSL[cntSL] - spread; TwoSRattempt <= twoSL[cntSL] + spread; TwoSRattempt+=2 ){
               bool encountered = false;
               for ( int cntSR = 0; cntSR < num_SR; cntSR++ ){
                  if ( twoSR[cntSR] == TwoSRattempt ){
                     encountered = true;
                  }
               }
               if ( encountered == false ){
                  const int dimR = denBK->gCurrentDim(DMRGindex+1,NR,TwoSRattempt,IR);
                  if ( dimR > 0 ){
                     jumpR[num_SR+1] = jumpR[num_SR] + dimR;
                     twoSR[num_SR] = TwoSRattempt;
                     num_SR++;
                  }
               }
            }
         }
         assert( jumpR[num_SR] <= Dmax );
         
         if( num_SR == 0 ) {
            delete[] arrayL;
            delete[] arrayR;
            delete[] twoSL;

            delete[] twoSR;
            delete[] jumpL;
            delete[] jumpR;

            return 0.0;         
         }

         for ( int cntSR = 0; cntSR < num_SR; cntSR++ ){
            int TwoSRvalue = twoSR[ cntSR ];
            int dimR = jumpR[ cntSR+1 ] - jumpR[ cntSR ];
            for ( int TwoSLvalue = TwoSRvalue - spread; TwoSLvalue <= TwoSRvalue + spread; TwoSLvalue += 2 ){
            
               int indexSL = -1;
               for ( int cntSL = 0; cntSL < num_SL; cntSL++ ){
                  if ( twoSL[cntSL] == TwoSLvalue ){
                     indexSL = cntSL;
                     cntSL = num_SL; //exit loop
                  }
               }
               if ( indexSL != -1 ){
                  int dimL = jumpL[ indexSL+1 ] - jumpL[ indexSL ];
                  dcomplex * Tblock = mps[ DMRGindex ]->gStorage(NL,TwoSLvalue,IL,NR,TwoSRvalue,IR);
                  dcomplex prefactor = sqrt( TwoSRvalue + 1 )
                                   * Wigner::wigner3j(TwoSLvalue, spread, TwoSRvalue, twoSLz, twoSzloc, -twoSRz)
                                   * Special::phase( -TwoSLvalue + spread - twoSRz );
                  dcomplex add2array = 1.0;
                  char notrans = 'N';
                  zgemm_( &notrans, &notrans, &dimFirst, &dimR, &dimL, &prefactor, arrayL + jumpL[indexSL], &dimFirst, Tblock, &dimL, &add2array, arrayR + jumpR[cntSR], &dimFirst);
               }
            }
         }
         
         //Swap L <--> R
         {
            dcomplex * temp = arrayR;
            arrayR = arrayL;
            arrayL = temp;
            int * temp2 = twoSR;
            twoSR = twoSL;
            twoSL = temp2;
            temp2 = jumpR;
            jumpR = jumpL;
            jumpL = temp2;
            num_SL = num_SR;
            NL = NR;
            IL = IR;
            twoSLz = twoSRz;
         }
      }
      
      theCoeff = arrayL[0];

      assert(   num_SL == 1              );
      assert( jumpL[1] == 1              );
      assert( twoSL[0] == prob->gTwoS()  );
      assert(       NL == prob->gN()     );
      assert(       IL == prob->gIrrep() );
      
      delete [] arrayL;
      delete [] arrayR;
      delete [] twoSL;
      delete [] twoSR;
      delete [] jumpL;
      delete [] jumpR;
   
   }
   
   #ifdef CHEMPS2_MPI_COMPILATION
   if ( mpi_chemps2_master_only ){ MPIchemps2::broadcast_array_double( &theCoeff, 1, MPI_CHEMPS2_MASTER ); }
   #endif
   return theCoeff;


//    const SyBookkeeper * denBK = mps[ 0 ]->gBK();
//    int L                      = mps[ 0 ]->gBK()->gL();
//    //DMRGcoeff = alpha/beta[Hamindex = Prob->gf2(DMRGindex)]

//    //Check if it's possible
//    int nTot  = 0;
//    int twoSz = 0;
//    int iTot  = 0;

//    for ( int DMRGindex = 0; DMRGindex < L; DMRGindex++ ) {
//       const int HamIndex = DMRGindex;
//       assert( ( alpha[ HamIndex ] == 0 ) || ( alpha[ HamIndex ] == 1 ) );
//       assert( ( beta[ HamIndex ] == 0 ) || ( beta[ HamIndex ] == 1 ) );
//       nTot += alpha[ HamIndex ] + beta[ HamIndex ];
//       twoSz += alpha[ HamIndex ] - beta[ HamIndex ];
//       if ( ( alpha[ HamIndex ] + beta[ HamIndex ] ) == 1 ) { iTot = Irreps::directProd( iTot, denBK->gIrrep( DMRGindex ) ); }
//    }

//    if ( prob->gN() != nTot ) {
//       // std::cout << "DMRG::getFCIcoefficient : Ndesired = " << prob->gN() << " and Ntotal in alpha and beta strings = " << nTot << std::endl;
//       return 0.0;
//    }

//    // 2Sz can be -Prob->2S() ; -Prob->2S()+2 ; -Prob->2S()+4 ; ... ; Prob->2S()
//    if ( ( prob->gTwoS() < twoSz ) || ( twoSz < -prob->gTwoS() ) || ( ( prob->gTwoS() - twoSz ) % 2 != 0 ) ) {
//       // std::cout << "DMRG::getFCIcoefficient : 2Sdesired = " << prob->gTwoS() << " and 2Sz in alpha and beta strings = " << twoSz << std::endl;
//       return 0.0;
//    }

//    if ( prob->gIrrep() != iTot ) {
//       // std::cout << "DMRG::getFCIcoefficient : Idesired = " << prob->gIrrep() << " and Irrep of alpha and beta strings = " << iTot << std::endl;
//       return 0.0;
//    }

//    dcomplex theCoeff = 2.0; // A FCI coefficient always lies in between -1.0 and 1.0

//    //Construct necessary arrays
//    int Dmax = 1;
//    for ( int DMRGindex = 1; DMRGindex < L; DMRGindex++ ) {
//       const int DtotBound = denBK->gTotDimAtBound( DMRGindex );
//       if ( DtotBound > Dmax ) { Dmax = DtotBound; }
//    }

//    dcomplex * arrayL = new dcomplex[ Dmax ];
//    dcomplex * arrayR = new dcomplex[ Dmax ];
//    int * twoSL       = new int[ L ];
//    int * twoSR       = new int[ L ];
//    int * jumpL       = new int[ L + 1 ];
//    int * jumpR       = new int[ L + 1 ];

//    //Start the iterator
//    int num_SL          = 0;
//    jumpL[ num_SL ]     = 0;
//    int dimFirst        = 1;
//    jumpL[ num_SL + 1 ] = jumpL[ num_SL ] + dimFirst;
//    twoSL[ num_SL ]     = 0;
//    num_SL++;
//    arrayL[ 0 ] = 1.0;
//    int NL      = 0;
//    int IL      = 0;
//    int twoSLz  = 0;

//    for ( int DMRGindex = 0; DMRGindex < L; DMRGindex++ ) {
//       //Clear the right array
//       for ( int count = 0; count < Dmax; count++ ) { arrayR[ count ] = 0.0; }

//       //The local occupation
//       const int HamIndex = ( prob->gReorder() ) ? prob->gf2( DMRGindex ) : DMRGindex;
//       const int Nlocal   = alpha[ HamIndex ] + beta[ HamIndex ];
//       const int twoSzloc = alpha[ HamIndex ] - beta[ HamIndex ];

//       //The right symmetry sectors
//       const int NR     = NL + Nlocal;
//       const int twoSRz = twoSLz + twoSzloc;
//       const int IR     = ( ( Nlocal == 1 ) ? ( Irreps::directProd( IL, denBK->gIrrep( DMRGindex ) ) ) : IL );

//       // std::cout << "NL " << NL << " twoSLz " << twoSLz << " IL " << IL << " NR " << NR << " twoSRz " << twoSRz << " IR " << IR << std::endl;
//       // std::cout << *mps[ DMRGindex ] << std::endl;

//       // abort();

//       int num_SR       = 0;
//       jumpR[ num_SR ]  = 0;
//       const int spread = ( ( Nlocal == 1 ) ? 1 : 0 );
//       for ( int cntSL = 0; cntSL < num_SL; cntSL++ ) {
//          for ( int TwoSRattempt = twoSL[ cntSL ] - spread; TwoSRattempt <= twoSL[ cntSL ] + spread; TwoSRattempt += 2 ) {
//             bool encountered = false;
//             for ( int cntSR = 0; cntSR < num_SR; cntSR++ ) {
//                if ( twoSR[ cntSR ] == TwoSRattempt ) {
//                   encountered = true;
//                }
//             }
//             if ( encountered == false ) {
//                const int dimR = denBK->gCurrentDim( DMRGindex + 1, NR, TwoSRattempt, IR );
//                if ( dimR > 0 ) {
//                   jumpR[ num_SR + 1 ] = jumpR[ num_SR ] + dimR;
//                   twoSR[ num_SR ]     = TwoSRattempt;
//                   num_SR++;
//                }
//             }
//          }
//       }
//       // std::cout << num_SR << std::endl;
      
//       assert( jumpR[ num_SR ] <= Dmax );

//       if( num_SR == 0 ) {
//          delete[] arrayL;
//          delete[] arrayR;
//          delete[] twoSL;
//          delete[] twoSR;
//          delete[] jumpL;
//          delete[] jumpR;

//          return 0.0;         
//       }

//       for ( int cntSR = 0; cntSR < num_SR; cntSR++ ) {
//          int TwoSRvalue = twoSR[ cntSR ];
//          int dimR       = jumpR[ cntSR + 1 ] - jumpR[ cntSR ];
//          for ( int TwoSLvalue = TwoSRvalue - spread; TwoSLvalue <= TwoSRvalue + spread; TwoSLvalue += 2 ) {

//             int indexSL = -1;
//             for ( int cntSL = 0; cntSL < num_SL; cntSL++ ) {
//                if ( twoSL[ cntSL ] == TwoSLvalue ) {
//                   indexSL = cntSL;
//                   cntSL   = num_SL; //exit loop
//                }
//             }
//             if ( indexSL != -1 ) {
//                int dimL           = jumpL[ indexSL + 1 ] - jumpL[ indexSL ];
//                dcomplex * Tblock  = mps[ DMRGindex ]->gStorage( NL, TwoSLvalue, IL, NR, TwoSRvalue, IR );
//                dcomplex prefactor = sqrt( TwoSRvalue + 1 ) * Wigner::wigner3j( TwoSLvalue, spread, TwoSRvalue, twoSLz, twoSzloc, -twoSRz ) * Special::phase( -TwoSLvalue + spread - twoSRz );
//                dcomplex add2array = 1.0;
//                char notrans       = 'N';
//                zgemm_( &notrans, &notrans, &dimFirst, &dimR, &dimL, &prefactor, arrayL + jumpL[ indexSL ], &dimFirst, Tblock, &dimL, &add2array, arrayR + jumpR[ cntSR ], &dimFirst );
//             }
//          }
//       }
//       // abort();

//       //Swap L <--> R
//       {
//          dcomplex * temp = arrayR;
//          arrayR          = arrayL;
//          arrayL          = temp;
//          int * temp2     = twoSR;
//          twoSR           = twoSL;
//          twoSL           = temp2;
//          temp2           = jumpR;
//          jumpR           = jumpL;
//          jumpL           = temp2;
//          num_SL          = num_SR;
//          NL              = NR;
//          IL              = IR;
//          twoSLz          = twoSRz;
//       }
//    }

//    theCoeff = arrayL[ 0 ];

//    assert( num_SL == 1 );
//    assert( jumpL[ 1 ] == 1 );
//    assert( twoSL[ 0 ] == prob->gTwoS() );
//    assert( NL == prob->gN() );
//    assert( IL == prob->gIrrep() );

//    delete[] arrayL;
//    delete[] arrayR;
//    delete[] twoSL;
//    delete[] twoSR;
//    delete[] jumpL;
//    delete[] jumpR;

//    return theCoeff;
}

dcomplex CheMPS2::overlap( CTensorT ** mpsA, CTensorT ** mpsB ) {
   const int L = mpsA[ 0 ]->gBK()->gL();

   CTensorO * overlapOld;
   overlapOld = new CTensorO( 1, true, mpsA[ 0 ]->gBK(), mpsB[ 0 ]->gBK() );
   overlapOld->create( mpsA[ 0 ], mpsB[ 0 ] );

   CTensorO * overlapNext;

   for ( int i = 1; i < L; i++ ) {
      overlapNext = new CTensorO( i + 1, true, mpsA[ i ]->gBK(), mpsB[ i ]->gBK() );
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
   return std::real( std::sqrt( overlap( mps, mps ) ) );
}

void CheMPS2::normalize( const int L, CTensorT ** mps ) {
   const double normMPS = norm( mps );
   for( int idx = 0; idx < L; idx++ ){
      mps[ idx ]->number_operator( 0.0,  std::pow( normMPS, - 1.0 / L ) );
   }
}

void CheMPS2::scale( const dcomplex factor, const int L, CTensorT ** mps ) {
   for( int idx = 0; idx < L; idx++ ){
      mps[ idx ]->number_operator( 0.0,  std::pow( factor, 1.0 / L ) );
   }
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
                                   CTensorT * oldLeft, SyBookkeeper * oldLeftBK,
                                   CTensorT * oldRight, SyBookkeeper * oldRightBK,
                                   CTensorT * newLeft, SyBookkeeper * newLeftBK,
                                   CTensorT * newRight, SyBookkeeper * newRightBK ) {
   assert( oldLeftBK == oldRightBK );
   assert( newLeftBK == newRightBK );
   assert( oldLeft->gIndex() == newLeft->gIndex() );
   assert( oldRight->gIndex() == newRight->gIndex() );
   assert( oldLeft->gIndex() + 1 == oldRight->gIndex() );
   const int index = oldRight->gIndex();

   int nSectors = 0;
   for ( int NL = oldRightBK->gNmin( index ); NL <= oldRightBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = oldRightBK->gTwoSmin( index, NL ); TwoSL <= oldRightBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < oldRightBK->getNumberOfIrreps(); IL++ ) {
            int dimL = oldRightBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < oldRight->gNKappa(); ikappa++ ) {
                  if ( ( NL == oldRight->gNL( ikappa ) ) && ( TwoSL == oldRight->gTwoSL( ikappa ) ) && ( IL == oldRight->gIL( ikappa ) ) ) {
                     dimRtotal += oldRightBK->gCurrentDim( index + 1, oldRight->gNR( ikappa ), oldRight->gTwoSR( ikappa ), oldRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  nSectors++;
               }
            }
         }
      }
   }

   int * SplitSectNL    = new int[ nSectors ];
   int * SplitSectTwoJL = new int[ nSectors ];
   int * SplitSectIL    = new int[ nSectors ];
   int * DimLs          = new int[ nSectors ];
   int * DimMs          = new int[ nSectors ];
   int * DimRs          = new int[ nSectors ];

   nSectors = 0;
   for ( int NL = oldRightBK->gNmin( index ); NL <= oldRightBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = oldRightBK->gTwoSmin( index, NL ); TwoSL <= oldRightBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < oldRightBK->getNumberOfIrreps(); IL++ ) {
            int dimL = oldRightBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < oldRight->gNKappa(); ikappa++ ) {
                  if ( ( NL == oldRight->gNL( ikappa ) ) && ( TwoSL == oldRight->gTwoSL( ikappa ) ) && ( IL == oldRight->gIL( ikappa ) ) ) {
                     dimRtotal += oldRightBK->gCurrentDim( index + 1, oldRight->gNR( ikappa ), oldRight->gTwoSR( ikappa ), oldRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  SplitSectNL[ nSectors ]    = NL;
                  SplitSectTwoJL[ nSectors ] = TwoSL;
                  SplitSectIL[ nSectors ]    = IL;
                  DimLs[ nSectors ]          = dimL;
                  DimMs[ nSectors ]          = std::min( dimL, dimRtotal );
                  DimRs[ nSectors ]          = dimRtotal;
                  nSectors++;
               }
            }
         }
      }
   }

   double ** Lambdas = new double *[ nSectors ];
   dcomplex ** Us    = new dcomplex *[ nSectors ];
   dcomplex ** VTs   = new dcomplex *[ nSectors ];

   for ( int iSector = 0; iSector < nSectors; iSector++ ) {

      Us[ iSector ]      = new dcomplex[ DimLs[ iSector ] * DimMs[ iSector ] ];
      Lambdas[ iSector ] = new double[ DimMs[ iSector ] ];
      VTs[ iSector ]     = new dcomplex[ DimMs[ iSector ] * DimRs[ iSector ] ];

      // Copy the relevant parts from storage to mem & multiply with factor !!
      dcomplex * mem = new dcomplex[ DimLs[ iSector ] * DimRs[ iSector ] ];
      int dimRtotal2 = 0;
      for ( int ikappa = 0; ikappa < oldRight->gNKappa(); ikappa++ ) {
         if ( ( SplitSectNL[ iSector ] == oldRight->gNL( ikappa ) ) && ( SplitSectTwoJL[ iSector ] == oldRight->gTwoSL( ikappa ) ) && ( SplitSectIL[ iSector ] == oldRight->gIL( ikappa ) ) ) {
            int dimL = oldRightBK->gCurrentDim( index    , oldRight->gNL( ikappa ), oldRight->gTwoSL( ikappa ), oldRight->gIL( ikappa ) );
            int dimR = oldRightBK->gCurrentDim( index + 1, oldRight->gNR( ikappa ), oldRight->gTwoSR( ikappa ), oldRight->gIR( ikappa ) );

            assert( dimL == DimLs[ iSector ] );

            if ( dimR > 0 ) {
               double factor      = sqrt( ( oldRight->gTwoSR( ikappa ) + 1.0 ) / ( SplitSectTwoJL[ iSector ] + 1.0 ) );
               dcomplex * storage = oldRight->gStorage() + oldRight->gKappa2index( ikappa );
               for ( int l = 0; l < DimLs[ iSector ]; l++ ) {
                  for ( int r = 0; r < dimR; r++ ) {
                     mem[ l + DimLs[ iSector ] * ( dimRtotal2 + r ) ] = factor * storage[ l + dimL * r ];
                  }
               }
            }
            dimRtotal2 += dimR;
         }
      }

      // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
      // central symmetry
      char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
      int lwork       = DimMs[ iSector ] * DimMs[ iSector ] + 3 * DimMs[ iSector ];
      int lrwork      = std::max( 5 * DimMs[ iSector ] * DimMs[ iSector ] + 5 * DimMs[ iSector ], 2 * std::max( DimLs[ iSector ], DimRs[ iSector ] ) * DimMs[ iSector ] + 2 * DimMs[ iSector ] * DimMs[ iSector ] + DimMs[ iSector ] );
      dcomplex * work = new dcomplex[ lwork ];
      double * rwork  = new double[ lrwork ];
      int * iwork     = new int[ 8 * DimMs[ iSector ] ];
      int info;

      // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
      zgesdd_( &jobz, DimLs + iSector, DimRs + iSector, mem, DimLs + iSector,
               Lambdas[ iSector ], Us[ iSector ], DimLs + iSector, VTs[ iSector ],
               DimMs + iSector, work, &lwork, rwork, iwork, &info );

      assert( info == 0 );

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
      NewDims = new int[ nSectors ];

      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         NewDims[ iSector ] = DimMs[ iSector ];
         totalDimSVD += NewDims[ iSector ];
      }

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
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
         for ( int iSector = 0; iSector < nSectors; iSector++ ) {
            for ( int cnt = 0; cnt < NewDims[ iSector ]; cnt++ ) {
               if ( Lambdas[ iSector ][ cnt ] <= lowerBound ) {
                  NewDims[ iSector ] = cnt;
               }
            }
         }

         // Discarded weight
         double totalSum     = 0.0;
         double discardedSum = 0.0;
         for ( int iSector = 0; iSector < nSectors; iSector++ ) {
            for ( int iLocal = 0; iLocal < DimMs[ iSector ]; iLocal++ ) {
               double temp = ( oldRight->gTwoSL( iSector ) + 1 ) * Lambdas[ iSector ][ iLocal ] * Lambdas[ iSector ][ iLocal ];
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
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         const int MPSdim = newRightBK->gCurrentDim( index, SplitSectNL[ iSector ], SplitSectTwoJL[ iSector ], SplitSectIL[ iSector ] );
         if ( NewDims[ iSector ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }
   
   newRightBK->allToZeroAtLink( index );
   for ( int iSector = 0; iSector < nSectors; iSector++ ) {
      // std::cout << "Setting Dim " << SplitSectNL[ iSector ] << " " << SplitSectTwoJL[ iSector ] << " " << SplitSectIL[ iSector ] << " " << NewDims[ iSector ] << std::endl;
      // std::cout << newRightBK << " " << index << " " << newRight->gIndex() <<  std::endl;
      newRightBK->SetDim( index, SplitSectNL[ iSector ], SplitSectTwoJL[ iSector ], SplitSectIL[ iSector ], NewDims[ iSector ] );
   }
   newRight->Reset();
   newLeft->Reset();

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

   newRight->Clear();
   newLeft->Clear();
   // std::cout << *newRight << std::endl;
   // Copy first dimM per central symmetry sector to the relevant parts
   for ( int iCenter = 0; iCenter < nSectors; iCenter++ ) {

      const int dimMnew = newLeftBK->gCurrentDim( index, SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ], SplitSectIL[ iCenter ] );
      const int dimMold = oldRightBK->gCurrentDim( index, SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ], SplitSectIL[ iCenter ] );
      if ( dimMnew > 0 ) {
         // U-part: copy
         int dimLtotal2 = 0;
         for ( int NL = SplitSectNL[ iCenter ] - 2; NL <= SplitSectNL[ iCenter ]; NL++ ) {
            const int TwoS = ( ( NL + 1 == SplitSectNL[ iCenter ] ) ? 1 : 0 );
            for ( int TwoSL = SplitSectTwoJL[ iCenter ] - TwoS; TwoSL <= SplitSectTwoJL[ iCenter ] + TwoS; TwoSL += 2 ) {
               if ( TwoSL >= 0 ) {
                  const int IL   = ( ( TwoS == 1 ) ? Irreps::directProd( newLeftBK->gIrrep( index - 1 ), SplitSectIL[ iCenter ] ) : SplitSectIL[ iCenter ] );
                  const int dimLnew = newLeftBK->gCurrentDim( index - 1, NL, TwoSL, IL );
                  if ( dimLnew > 0 ) {
                     dcomplex * TleftOld = oldLeft->gStorage( NL, TwoSL, IL, SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ], SplitSectIL[ iCenter ] );
                     dcomplex * TleftNew = newLeft->gStorage( NL, TwoSL, IL, SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ], SplitSectIL[ iCenter ] );
                     const int dimLold = oldLeftBK->gCurrentDim( index - 1, NL, TwoSL, IL );

                     assert( dimLnew == dimLold );

                     const int dimension_limit_right = std::min( dimMnew, DimMs[ iCenter ] );
                     for ( int l = 0; l < dimLnew; l++ ) {
                        for ( int r = 0; r < dimension_limit_right; r++ ) {
                           TleftNew[ l + dimLnew * r ] = 0.0;
                           for ( int m = 0; m < dimMold; m++ ) {
                              TleftNew[ l + dimLnew * r ] += TleftOld[ l + dimLold * m ] * Us[ iCenter ][ m + dimMold * r ] * Lambdas[ iCenter ][ r ];
                           }
                        }
                     }
                     for ( int l = 0; l < dimLnew; l++ ) {
                        for ( int r = dimension_limit_right; r < dimMnew; r++ ) {
                           TleftNew[ l + dimLnew * r ] = 0.0;
                        }
                     }
                     dimLtotal2 += dimLnew;
                  }
               }
            }
         }

         // VT-part: copy
         int dimRtotal2 = 0;
         for ( int NR = SplitSectNL[ iCenter ]; NR <= SplitSectNL[ iCenter ] + 2; NR++ ) {
            const int TwoS2 = ( ( NR == SplitSectNL[ iCenter ] + 1 ) ? 1 : 0 );
            for ( int TwoSR = SplitSectTwoJL[ iCenter ] - TwoS2; TwoSR <= SplitSectTwoJL[ iCenter ] + TwoS2; TwoSR += 2 ) {
               if ( TwoSR >= 0 ) {
                  const int IR   = ( ( TwoS2 == 1 ) ? Irreps::directProd( SplitSectIL[ iCenter ], newRightBK->gIrrep( index ) ) : SplitSectIL[ iCenter ] );
                  const int dimR = newRightBK->gCurrentDim( index + 1, NR, TwoSR, IR );
                  if ( dimR > 0 ) {
                     dcomplex * TrightNew  = newRight->gStorage( SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ], SplitSectIL[ iCenter ], NR, TwoSR, IR );
                     const dcomplex factor = sqrt( ( SplitSectTwoJL[ iCenter ] + 1.0 ) / ( TwoSR + 1.0 ) );

                     const int dimension_limit_left = min( dimMnew, DimMs[ iCenter ] );
                     for ( int l = 0; l < dimension_limit_left; l++ ) {
                        for ( int r = 0; r < dimR; r++ ) {
                           TrightNew[ l + dimMnew * r ] = factor * VTs[ iCenter ][ l + DimMs[ iCenter ] * ( dimRtotal2 + r ) ];
                        }
                     }
                     for ( int l = dimension_limit_left; l < dimMnew; l++ ) {
                        for ( int r = 0; r < dimR; r++ ) {
                           TrightNew[ l + dimMnew * r ] = 0.0;
                        }
                     }
                  }
                  dimRtotal2 += dimR;
               }
            }
         }
      }
   }

   for ( int iSector = 0; iSector < nSectors; iSector++ ) {
      if ( DimMs[ iSector ] > 0 ) {
         delete[] Us[ iSector ];
         delete[] Lambdas[ iSector ];
         delete[] VTs[ iSector ];
      }
   }

   delete[] Lambdas;
   delete[] Us;
   delete[] VTs;

   delete[] SplitSectNL;
   delete[] SplitSectTwoJL;
   delete[] SplitSectIL;
   delete[] DimLs;
   delete[] DimMs;
   delete[] DimRs;

   // std::cout << *oldRight << std::endl;
   // std::cout << *newRight << std::endl;
   // std::cout << "oooo " << newRight->gIndex() << std::endl;
   assert( newRight->CheckRightNormal() );

}

void CheMPS2::decomposeMovingRight( bool change, int virtualdimensionD, double cut_off,
                                    CTensorT * oldLeft, SyBookkeeper * oldLeftBK,
                                    CTensorT * oldRight, SyBookkeeper * oldRightBK,
                                    CTensorT * newLeft, SyBookkeeper * newLeftBK,
                                    CTensorT * newRight, SyBookkeeper * newRightBK ) {
   assert( oldLeftBK == oldRightBK );
   assert( newLeftBK == newRightBK );
   assert( oldLeft->gIndex() == newLeft->gIndex() );
   assert( oldRight->gIndex() == newRight->gIndex() );
   assert( oldLeft->gIndex() + 1 == oldRight->gIndex() );
   const int index = oldLeft->gIndex();

   int nSectors = 0;
   for ( int NR = oldLeftBK->gNmin( index + 1 ); NR <= oldLeftBK->gNmax( index + 1 ); NR++ ) {
      for ( int TwoSR = oldLeftBK->gTwoSmin( index + 1, NR ); TwoSR <= oldLeftBK->gTwoSmax( index + 1, NR ); TwoSR += 2 ) {
         for ( int IR = 0; IR < oldLeftBK->getNumberOfIrreps(); IR++ ) {
            int dimR = oldLeftBK->gCurrentDim( index + 1, NR, TwoSR, IR );
            if ( dimR > 0 ) {
               int dimLtotal = 0;
               for ( int ikappa = 0; ikappa < oldLeft->gNKappa(); ikappa++ ) {
                  if ( ( NR == oldLeft->gNR( ikappa ) ) && ( TwoSR == oldLeft->gTwoSR( ikappa ) ) && ( IR == oldLeft->gIR( ikappa ) ) ) {
                     dimLtotal += oldLeftBK->gCurrentDim( index, oldLeft->gNL( ikappa ), oldLeft->gTwoSL( ikappa ), oldLeft->gIL( ikappa ) );
                  }
               }
               if ( dimLtotal > 0 ) {
                  nSectors++;
               }
            }
         }
      }
   }

   int * SplitSectNR    = new int[ nSectors ];
   int * SplitSectTwoJR = new int[ nSectors ];
   int * SplitSectIR    = new int[ nSectors ];
   int * DimLs          = new int[ nSectors ];
   int * DimMs          = new int[ nSectors ];
   int * DimRs          = new int[ nSectors ];

   nSectors = 0;
   for ( int NR = oldLeftBK->gNmin( index + 1 ); NR <= oldLeftBK->gNmax( index + 1 ); NR++ ) {
      for ( int TwoSR = oldLeftBK->gTwoSmin( index + 1, NR ); TwoSR <= oldLeftBK->gTwoSmax( index + 1, NR ); TwoSR += 2 ) {
         for ( int IR = 0; IR < oldLeftBK->getNumberOfIrreps(); IR++ ) {
            int dimR = oldLeftBK->gCurrentDim( index + 1, NR, TwoSR, IR );
            if ( dimR > 0 ) {
               int dimLtotal = 0;
               for ( int ikappa = 0; ikappa < oldLeft->gNKappa(); ikappa++ ) {
                  if ( ( NR == oldLeft->gNR( ikappa ) ) && ( TwoSR == oldLeft->gTwoSR( ikappa ) ) && ( IR == oldLeft->gIR( ikappa ) ) ) {
                     dimLtotal += oldLeftBK->gCurrentDim( index, oldLeft->gNL( ikappa ), oldLeft->gTwoSL( ikappa ), oldLeft->gIL( ikappa ) );
                  }
               }
               if ( dimLtotal > 0 ) {
                  SplitSectNR[ nSectors ]    = NR;
                  SplitSectTwoJR[ nSectors ] = TwoSR;
                  SplitSectIR[ nSectors ]    = IR;
                  DimLs[ nSectors ]          = dimLtotal;
                  DimMs[ nSectors ]          = std::min( dimLtotal, dimR );
                  DimRs[ nSectors ]          = dimR;
                  nSectors++;
               }
            }
         }
      }
   }
   double ** Lambdas = NULL;
   dcomplex ** Us    = NULL;
   dcomplex ** VTs   = NULL;

   Lambdas = new double *[ nSectors ];
   Us      = new dcomplex *[ nSectors ];
   VTs     = new dcomplex *[ nSectors ];

   for ( int iSector = 0; iSector < nSectors; iSector++ ) {

      Us[ iSector ]      = new dcomplex[ DimLs[ iSector ] * DimMs[ iSector ] ];
      Lambdas[ iSector ] = new double[ DimMs[ iSector ] ];
      VTs[ iSector ]     = new dcomplex[ DimMs[ iSector ] * DimRs[ iSector ] ];

      // Copy the relevant parts from storage to mem & multiply with factor !!
      dcomplex * mem = new dcomplex[ DimLs[ iSector ] * DimRs[ iSector ] ];
      int dimLtotal2 = 0;
      for ( int ikappa = 0; ikappa < oldLeft->gNKappa(); ikappa++ ) {
         if ( ( SplitSectNR[ iSector ] == oldLeft->gNR( ikappa ) ) && ( SplitSectTwoJR[ iSector ] == oldLeft->gTwoSR( ikappa ) ) && ( SplitSectIR[ iSector ] == oldLeft->gIR( ikappa ) ) ) {
            int dimL = oldLeftBK->gCurrentDim( index, oldLeft->gNL( ikappa ), oldLeft->gTwoSL( ikappa ), oldLeft->gIL( ikappa ) );
            if ( dimL > 0 ) {
               dcomplex * storage = oldLeft->gStorage() + oldLeft->gKappa2index( ikappa );
               for ( int l = 0; l < dimL; l++ ) {
                  for ( int r = 0; r < DimRs[ iSector ]; r++ ) {
                     mem[ dimLtotal2 + l + DimLs[ iSector ] * r ] = storage[ l + dimL * r ];
                  }
               }
               dimLtotal2 += dimL;
            }
         }
      }
      // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
      // central symmetry
      char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
      int lwork       = DimMs[ iSector ] * DimMs[ iSector ] + 3 * DimMs[ iSector ];
      int lrwork      = std::max( 5 * DimMs[ iSector ] * DimMs[ iSector ] + 5 * DimMs[ iSector ], 2 * std::max( DimLs[ iSector ], DimRs[ iSector ] ) * DimMs[ iSector ] + 2 * DimMs[ iSector ] * DimMs[ iSector ] + DimMs[ iSector ] );
      dcomplex * work = new dcomplex[ lwork ];
      double * rwork  = new double[ lrwork ];
      int * iwork     = new int[ 8 * DimMs[ iSector ] ];
      int info;

      // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
      zgesdd_( &jobz, DimLs + iSector, DimRs + iSector, mem, DimLs + iSector,
               Lambdas[ iSector ], Us[ iSector ], DimLs + iSector, VTs[ iSector ],
               DimMs + iSector, work, &lwork, rwork, iwork, &info );

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
      NewDims = new int[ nSectors ];

      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         NewDims[ iSector ] = DimMs[ iSector ];
         totalDimSVD += NewDims[ iSector ];
      }

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
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
      // std::cout << "totalDimSVD" << std::endl;
      // std::cout << totalDimSVD << std::endl;
      // std::cout << maxD << std::endl;
      // int maxD = virtualdimensionD;
      // If larger then the required virtualdimensionD, new virtual dimensions
      // will be set in NewDims.
      if ( totalDimSVD > maxD ) {

         // The D+1'th value becomes the lower bound Schmidt value. Every value
         // smaller than or equal to the D+1'th value is thrown out (hence Dactual // <= Ddesired).
         const double lowerBound = values[ maxD ];
         for ( int iSector = 0; iSector < nSectors; iSector++ ) {
            for ( int cnt = 0; cnt < NewDims[ iSector ]; cnt++ ) {
               if ( Lambdas[ iSector ][ cnt ] <= lowerBound ) {
                  NewDims[ iSector ] = cnt;
               }
            }
         }

         // Discarded weight
         double totalSum     = 0.0;
         double discardedSum = 0.0;
         for ( int iSector = 0; iSector < nSectors; iSector++ ) {
            for ( int iLocal = 0; iLocal < DimMs[ iSector ]; iLocal++ ) {
               double temp = ( oldLeft->gTwoSL( iSector ) + 1 ) * Lambdas[ iSector ][ iLocal ] * Lambdas[ iSector ][ iLocal ];
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
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         const int MPSdim = newLeftBK->gCurrentDim( index + 1, SplitSectNR[ iSector ], SplitSectTwoJR[ iSector ], SplitSectIR[ iSector ] );
         if ( NewDims[ iSector ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }

   // if ( updateSectors == 1 ) {
   newLeftBK->allToZeroAtLink( index + 1 );
   for ( int iSector = 0; iSector < nSectors; iSector++ ) {
      newLeftBK->SetDim( index + 1, SplitSectNR[ iSector ], SplitSectTwoJR[ iSector ], SplitSectIR[ iSector ], NewDims[ iSector ] );
   }
   newLeft->Reset();
   newRight->Reset();
   // }

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

   newLeft->Clear();
   newRight->Clear();

   // Copy first dimM per central symmetry sector to the relevant parts
   for ( int iCenter = 0; iCenter < nSectors; iCenter++ ) {

      const int dimMnew = newLeftBK->gCurrentDim( index + 1, SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ] );
      const int dimMold = oldRightBK->gCurrentDim( index + 1, SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ] );
      if ( dimMnew > 0 ) {
         int dimLtotal2 = 0;

         for ( int NL = SplitSectNR[ iCenter ] - 2; NL <= SplitSectNR[ iCenter ]; NL++ ) {
            const int TwoS = ( ( NL + 1 == SplitSectNR[ iCenter ] ) ? 1 : 0 );

            for ( int TwoSL = SplitSectTwoJR[ iCenter ] - TwoS; TwoSL <= SplitSectTwoJR[ iCenter ] + TwoS; TwoSL += 2 ) {
               if ( TwoSL >= 0 ) {
                  const int IL   = ( ( TwoS == 1 ) ? Irreps::directProd( newLeftBK->gIrrep( index ), SplitSectIR[ iCenter ] ) : SplitSectIR[ iCenter ] );
                  const int dimL = newLeftBK->gCurrentDim( index, NL, TwoSL, IL );

                  if ( dimL > 0 ) {
                     dcomplex * TleftBlock           = newLeft->gStorage( NL, TwoSL, IL, SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ] );

                     const int dimension_limit_right = min( dimMnew, DimMs[ iCenter ] );
                     for ( int r = 0; r < dimension_limit_right; r++ ) {
                        for ( int l = 0; l < dimL; l++ ) {
                           TleftBlock[ l + dimL * r ] = Us[ iCenter ][ dimLtotal2 + l + DimLs[ iCenter ] * r ];
                        }
                     }
                     for ( int r = dimension_limit_right; r < dimMnew; r++ ) {
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
         // int dimRtotal2 = 0;
         for ( int NR = SplitSectNR[ iCenter ]; NR <= SplitSectNR[ iCenter ] + 2; NR++ ) {
            const int TwoS = ( ( NR == SplitSectNR[ iCenter ] + 1 ) ? 1 : 0 );

            for ( int TwoSR = SplitSectTwoJR[ iCenter ] - TwoS; TwoSR <= SplitSectTwoJR[ iCenter ] + TwoS; TwoSR += 2 ) {
               if ( TwoSR >= 0 ) {
                  const int IR   = ( ( TwoS == 1 ) ? Irreps::directProd( newLeftBK->gIrrep( index + 1 ), SplitSectIR[ iCenter ] ) : SplitSectIR[ iCenter ] );
                  const int dimR = newRightBK->gCurrentDim( index + 2, NR, TwoSR, IR );

                  if ( dimR > 0 ) {
                     dcomplex * Trightnew = newRight->gStorage( SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ], NR, TwoSR, IR );
                     dcomplex * Trightold = oldRight->gStorage( SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ], NR, TwoSR, IR );

                     const int dimMnew = newRight->gBK()->gCurrentDim( index + 1, SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ] );
                     const int dimMold = oldRight->gBK()->gCurrentDim( index + 1, SplitSectNR[ iCenter ], SplitSectTwoJR[ iCenter ], SplitSectIR[ iCenter ] );

                     const int dimension_limit_left = min( dimMnew, DimMs[ iCenter ] );
                     for ( int l = 0; l < dimension_limit_left; l++ ) {
                        for ( int r = 0; r < dimR; r++ ) {
                           Trightnew[ l + dimMnew * r ] = 0.0;
                           for ( int m = 0; m < dimMold; m++ ) {
                              Trightnew[ l + dimMnew * r ] += Lambdas[ iCenter ][ l ] * VTs[ iCenter ][ l + DimMs[ iCenter ] * m ] * Trightold[ m + dimMold * r ];
                           }
                        }
                     }
                     for ( int l = dimension_limit_left; l < dimMnew; l++ ) {
                        for ( int r = 0; r < dimR; r++ ) {
                           Trightnew[ l + dimR * r ] = 0.0;
                        }
                     }
                  }
               }
            }
         }
      }
   }

   for ( int iSector = 0; iSector < nSectors; iSector++ ) {
      if ( DimMs[ iSector ] > 0 ) {
         delete[] Us[ iSector ];
         delete[] Lambdas[ iSector ];
         delete[] VTs[ iSector ];
      }
   }

   delete[] Lambdas;
   delete[] Us;
   delete[] VTs;

   delete[] SplitSectNR;
   delete[] SplitSectTwoJR;
   delete[] SplitSectIR;
   delete[] DimLs;
   delete[] DimMs;
   delete[] DimRs;

   assert( newLeft->CheckLeftNormal() );
}
