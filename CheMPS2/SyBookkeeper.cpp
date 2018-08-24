/*
   CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum chemistry
   Copyright (C) 2013-2018 Sebastian Wouters

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

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "Irreps.h"
#include "Options.h"
#include "SyBookkeeper.h"

CheMPS2::SyBookkeeper::SyBookkeeper( const Problem * Prob, const int D ) {

   this->Prob = Prob;
   Irreps temp( Prob->gSy() );
   this->num_irreps = temp.getNumberOfIrreps();

   // Allocate the arrays
   allocate_arrays();

   // Fill FCIdim
   fillFCIdim();

   // Copy FCIdim to CURdim
   CopyDim( FCIdim, CURdim );

   // Scale the CURdim
   ScaleCURdim( D, 1, gL() - 1 );

   assert( IsPossible() );
}

CheMPS2::SyBookkeeper::SyBookkeeper( const SyBookkeeper & tocopy ) {

   this->Prob = tocopy.gProb();
   Irreps temp( Prob->gSy() );
   this->num_irreps = temp.getNumberOfIrreps();

   // Allocate the arrays
   allocate_arrays();

   // Fill FCIdim
   fillFCIdim();

   // Copy the CURdim
   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
               CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = tocopy.gCurrentDim( boundary, N, TwoS, irrep );
            }
         }
      }
   }
}

CheMPS2::SyBookkeeper::SyBookkeeper( const Problem * Prob, const int * occupation ) {

   this->Prob = Prob;
   Irreps temp( Prob->gSy() );
   this->num_irreps = temp.getNumberOfIrreps();

   assert(Prob->gTwoS() <= 1);

   // Allocate the arrays
   allocate_arrays();

   // Fill FCIdim
   fillFCIdim();

   // Copy FCIdim to CURdim
   CopyDim( FCIdim, CURdim );

   int Nelec = 0;
   for ( int index = 0; index < gL(); index++ ) {
      int ni = occupation[ Prob->gReorder() ? Prob->gf2( index ) : index ];
      for ( int NL = gNmin( index ); NL <= gNmax( index ); NL++ ) {
         for ( int TwoSL = gTwoSmin( index, NL ); TwoSL <= gTwoSmax( index, NL ); TwoSL += 2 ) {
            for ( int IL = 0; IL < getNumberOfIrreps(); IL++ ) {
               const int dimL = gFCIdim( index, NL, TwoSL, IL );
               bool needBlock = false;
               if ( dimL > 0 ) {
                  for ( int NR = NL; NR <= NL + 2; NR++ ) {
                     const int TwoJ = ( ( NR == NL + 1 ) ? 1 : 0 );
                     for ( int TwoSR = TwoSL - TwoJ; TwoSR <= TwoSL + TwoJ; TwoSR += 2 ) {
                        if ( TwoSR >= 0 ) {
                           int IR         = ( ( NR == NL + 1 ) ? Irreps::directProd( IL, gIrrep( index ) ) : IL );
                           const int dimR = gFCIdim( index + 1, NR, TwoSR, IR );
                           if ( ( dimR > 0 ) && ( NL == Nelec ) && ( NR == Nelec + ni ) && ( TwoSL == ( NL % 2 ) ) && ( TwoSR == ( NR % 2 ) ) ) {
                              needBlock = true;
                           }
                        }
                     }
                  }
               }
               if ( needBlock ) {
                  SetDim( index, NL, TwoSL, IL, 1 );
               } else {
                  SetDim( index, NL, TwoSL, IL, 0 );
               }
            }
         }
      }
      Nelec += ni;
   }

   assert( IsPossible() );
}

CheMPS2::SyBookkeeper::SyBookkeeper( const int site, SyBookkeeper * orig ) {

   this->Prob = orig->gProb();
   Irreps temp( Prob->gSy() );
   this->num_irreps = temp.getNumberOfIrreps();

   // Allocate the arrays
   allocate_arrays();

   // Fill FCIdim
   fillFCIdim();

   // Copy the CURdim
   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
               if ( boundary != site ) {
                  CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = orig->gCurrentDim( boundary, N, TwoS, irrep );
               } else {
                  if ( orig->gCurrentDim( boundary, N, TwoS, irrep ) > 0 ) {
                     CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = orig->gCurrentDim( boundary, N, TwoS, irrep );
                  } else {
                     CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = orig->gFCIdim( boundary, N, TwoS, irrep );
                  }
               }
            }
         }
      }
   }
}

void CheMPS2::SyBookkeeper::allocate_arrays() {

   // Set the min and max particle number and spin
   Nmin    = new int[ gL() + 1 ];
   Nmax    = new int[ gL() + 1 ];
   TwoSmin = new int *[ gL() + 1 ];
   TwoSmax = new int *[ gL() + 1 ];
   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      Nmin[ boundary ]    = Prob->gNmin( boundary );
      Nmax[ boundary ]    = Prob->gNmax( boundary );
      TwoSmin[ boundary ] = new int[ Nmax[ boundary ] - Nmin[ boundary ] + 1 ];
      TwoSmax[ boundary ] = new int[ Nmax[ boundary ] - Nmin[ boundary ] + 1 ];
      for ( int N = Nmin[ boundary ]; N <= Nmax[ boundary ]; N++ ) {
         const int temporary                         = gL() - boundary - abs( gN() - N - gL() + boundary );
         TwoSmin[ boundary ][ N - Nmin[ boundary ] ] = std::max( N % 2, gTwoS() - temporary );
         TwoSmax[ boundary ][ N - Nmin[ boundary ] ] = std::min( boundary - abs( boundary - N ), gTwoS() + temporary );
      }
   }

   // FCIdim & CURdim memory allocation
   FCIdim = new int ***[ gL() + 1 ];
   CURdim = new int ***[ gL() + 1 ];
   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      FCIdim[ boundary ] = new int **[ gNmax( boundary ) - gNmin( boundary ) + 1 ];
      CURdim[ boundary ] = new int **[ gNmax( boundary ) - gNmin( boundary ) + 1 ];
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         FCIdim[ boundary ][ N - gNmin( boundary ) ] = new int *[ ( gTwoSmax( boundary, N ) - gTwoSmin( boundary, N ) ) / 2 + 1 ];
         CURdim[ boundary ][ N - gNmin( boundary ) ] = new int *[ ( gTwoSmax( boundary, N ) - gTwoSmin( boundary, N ) ) / 2 + 1 ];
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            FCIdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ] = new int[ num_irreps ];
            CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ] = new int[ num_irreps ];
         }
      }
   }
}

CheMPS2::SyBookkeeper::~SyBookkeeper() {

   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            delete[] FCIdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ];
            delete[] CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ];
         }
         delete[] FCIdim[ boundary ][ N - gNmin( boundary ) ];
         delete[] CURdim[ boundary ][ N - gNmin( boundary ) ];
      }
      delete[] FCIdim[ boundary ];
      delete[] CURdim[ boundary ];
   }
   delete[] FCIdim;
   delete[] CURdim;

   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      delete[] TwoSmin[ boundary ];
      delete[] TwoSmax[ boundary ];
   }
   delete[] TwoSmin;
   delete[] TwoSmax;
   delete[] Nmin;
   delete[] Nmax;
}

const CheMPS2::Problem * CheMPS2::SyBookkeeper::gProb() const { return Prob; }

void CheMPS2::SyBookkeeper::SetDim( const int boundary, const int N, const int TwoS, const int irrep, const int value ) {

   if ( gFCIdim( boundary, N, TwoS, irrep ) != 0 ) {
      CURdim[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = value;
   }
}

void CheMPS2::SyBookkeeper::allToZeroAtLink( const int index ){
   for ( int N = gNmin( index ); N <= gNmax( index ); N++ ) {
      for ( int TwoS = gTwoSmin( index, N ); TwoS <= gTwoSmax( index, N ); TwoS += 2 ) {
         for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
            SetDim( index, N, TwoS, irrep, 0 );
         }
      }
   }
}

void CheMPS2::SyBookkeeper::fillFCIdim() {

   // On the left-hand side only the trivial symmetry sector is allowed
   for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
      FCIdim[ 0 ][ 0 ][ 0 ][ irrep ] = 0;
   }
   FCIdim[ 0 ][ 0 ][ 0 ][ 0 ] = 1;

   // Fill boundaries 1 to L from left to right
   fill_fci_dim_right( FCIdim, 1, gL() );

   // Remember the FCI virtual dimension at the RHS
   const int rhs = FCIdim[ gL() ][ 0 ][ 0 ][ gIrrep() ];

   // On the right-hand side only the targeted symmetry sector is allowed
   for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
      FCIdim[ gL() ][ 0 ][ 0 ][ irrep ] = 0;
   }
   FCIdim[ gL() ][ 0 ][ 0 ][ gIrrep() ] = std::min( 1, rhs );

   // Fill boundarties 0 to L - 1 from right to left
   fill_fci_dim_left( FCIdim, 0, gL() - 1 );
}

void CheMPS2::SyBookkeeper::fill_fci_dim_right( int **** storage, const int start, const int stop ) {

   for ( int boundary = start; boundary <= stop; boundary++ ) {
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
               const int value                                                                                 = std::min( CheMPS2::SYBK_dimensionCutoff,
                                           gDimPrivate( storage, boundary - 1, N, TwoS, irrep ) + gDimPrivate( storage, boundary - 1, N - 2, TwoS, irrep ) + gDimPrivate( storage, boundary - 1, N - 1, TwoS + 1, Irreps::directProd( irrep, gIrrep( boundary - 1 ) ) ) + gDimPrivate( storage, boundary - 1, N - 1, TwoS - 1, Irreps::directProd( irrep, gIrrep( boundary - 1 ) ) ) );
               storage[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = value;
            }
         }
      }
   }
}

void CheMPS2::SyBookkeeper::fill_fci_dim_left( int **** storage, const int start, const int stop ) {

   for ( int boundary = stop; boundary >= start; boundary-- ) {
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
               const int value                                                                                 = std::min( gDimPrivate( storage, boundary, N, TwoS, irrep ),
                                           std::min( CheMPS2::SYBK_dimensionCutoff,
                                                     gDimPrivate( storage, boundary + 1, N, TwoS, irrep ) + gDimPrivate( storage, boundary + 1, N + 2, TwoS, irrep ) + gDimPrivate( storage, boundary + 1, N + 1, TwoS + 1, Irreps::directProd( irrep, gIrrep( boundary ) ) ) + gDimPrivate( storage, boundary + 1, N + 1, TwoS - 1, Irreps::directProd( irrep, gIrrep( boundary ) ) ) ) );
               storage[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = value;
            }
         }
      }
   }
}

void CheMPS2::SyBookkeeper::CopyDim( int **** origin, int **** target ) {

   for ( int boundary = 0; boundary <= gL(); boundary++ ) {
      for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
         for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
            for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
               target[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ] = origin[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ];
            }
         }
      }
   }
}

void CheMPS2::SyBookkeeper::ScaleCURdim( const int virtual_dim, const int start, const int stop ) {

   for ( int boundary = start; boundary <= stop; boundary++ ) {

      const int totaldim = gTotDimAtBound( boundary );

      if ( totaldim > virtual_dim ) {
         double factor = ( 1.0 * virtual_dim ) / totaldim;
         for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
            for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
               for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
                  const int value = ( int ) ( ceil( factor * gCurrentDim( boundary, N, TwoS, irrep ) ) + 0.1 );
                  SetDim( boundary, N, TwoS, irrep, value );
               }
            }
         }
      }
   }
}

int CheMPS2::SyBookkeeper::gDimPrivate( int **** storage, const int boundary, const int N, const int TwoS, const int irrep ) const {

   if ( ( boundary < 0 ) || ( boundary > gL() ) ) { return 0; }
   if ( ( N > gNmax( boundary ) ) || ( N < gNmin( boundary ) ) ) { return 0; }
   if ( ( TwoS % 2 ) != ( gTwoSmin( boundary, N ) % 2 ) ) { return 0; }
   if ( ( TwoS < gTwoSmin( boundary, N ) ) || ( TwoS > gTwoSmax( boundary, N ) ) ) { return 0; }
   if ( ( irrep < 0 ) || ( irrep >= num_irreps ) ) { return 0; }
   return storage[ boundary ][ N - gNmin( boundary ) ][ ( TwoS - gTwoSmin( boundary, N ) ) / 2 ][ irrep ];
}

int CheMPS2::SyBookkeeper::gMaxDimAtBound( const int boundary ) const {

   int max_dim = 0;
   for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
      for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
         for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
            const int dim = gCurrentDim( boundary, N, TwoS, irrep );
            if ( dim > max_dim ) { max_dim = dim; }
         }
      }
   }
   return max_dim;
}
int CheMPS2::SyBookkeeper::gFCIDimAtBound( const int boundary ) const {
   int tot_dim = 0;
   for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
      for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
         for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
            tot_dim += gFCIdim( boundary, N, TwoS, irrep );
         }
      }
   }
   return tot_dim;
}

int CheMPS2::SyBookkeeper::gTotDimAtBound( const int boundary ) const {

   int tot_dim = 0;
   for ( int N = gNmin( boundary ); N <= gNmax( boundary ); N++ ) {
      for ( int TwoS = gTwoSmin( boundary, N ); TwoS <= gTwoSmax( boundary, N ); TwoS += 2 ) {
         for ( int irrep = 0; irrep < num_irreps; irrep++ ) {
            tot_dim += gCurrentDim( boundary, N, TwoS, irrep );
         }
      }
   }
   return tot_dim;
}

void CheMPS2::SyBookkeeper::restart( const int start, const int stop, const int virtual_dim ) {

   fill_fci_dim_right( CURdim, start, stop );
   fill_fci_dim_left( CURdim, start, stop );
   ScaleCURdim( virtual_dim, start, stop );
}

bool CheMPS2::SyBookkeeper::IsPossible() const {
   return ( gCurrentDim( gL(), gN(), gTwoS(), gIrrep() ) == 1 );
}

void CheMPS2::subspaceExpand( int index, bool movingRight, const SyBookkeeper * initBK, SyBookkeeper * sseBK ) {

   for ( int NL = initBK->gNmin( index ); NL <= initBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = initBK->gTwoSmin( index, NL ); TwoSL <= initBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < initBK->getNumberOfIrreps(); IL++ ) {
            const int dimL = movingRight ? initBK->gCurrentDim( index, NL, TwoSL, IL ) : initBK->gFCIdim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               sseBK->SetDim( index, NL, TwoSL, IL, initBK->gFCIdim( index, NL, TwoSL, IL ) );
               // for ( int NR = initBK->gNmin( index ); NR <= initBK->gNmax( index ); NR++ ) {
               //    for ( int TwoSR = initBK->gTwoSmin( index, NR ); TwoSR <= initBK->gTwoSmax( index, NR ); TwoSR += 2 ) {
               //       for ( int IR = 0; IR < initBK->getNumberOfIrreps(); IR++ ) {
               //          const int dimR = movingRight ? initBK->gFCIdim( index, NR, TwoSR, IR ) : initBK->gCurrentDim( index, NR, TwoSR, IR );
               //          if ( dimR > 0 ) {
               //          }
               //       }
               //    }
               // }
            }
         }
      }
   }
}

std::ostream & CheMPS2::operator<<( std::ostream & os, const CheMPS2::SyBookkeeper & bk ) {
   os << "#################################################################\n";

   os << "SymmetryBookkeeper with dimensions: " << std::endl;
   std::cout << "   matrix product state dimensions:\n";
   std::cout << "   ";
   for ( int i = 0; i < bk.gL() + 1; i++ ) {
      std::cout << std::setw( 10 ) << i;
   }
   std::cout << "\n";
   std::cout << "   ";
   for ( int i = 0; i < bk.gL() + 1; i++ ) {
      std::cout << std::setw( 10 ) << bk.gTotDimAtBound( i );
   }
   return os;
}
