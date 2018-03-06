
// #include <math.h>
// #include <stdlib.h> /*rand*/
#include <algorithm>
#include <assert.h>
#include <iostream>

#include "CTensorO.h"
#include "CTensorT.h"
#include "Lapack.h"
#include "Logger.h"

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
   dcomplex result = overlapOld->gStorage()[ 0 ];
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
