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

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "CTwoDM.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "MyHDF5.h"
#include "Options.h"
#include "Special.h"
#include "Wigner.h"

using std::cout;
using std::endl;
using std::max;

CheMPS2::CTwoDM::CTwoDM( const SyBookkeeper * denBKIn, const Problem * ProbIn ) {

   denBK = denBKIn;
   Prob  = ProbIn;
   L     = denBK->gL();

   const long long size = ( ( long long ) L ) * ( ( long long ) L ) * ( ( long long ) L ) * ( ( long long ) L );
   assert( INT_MAX >= size );
   two_rdm_A = new dcomplex[ size ];
   two_rdm_B = new dcomplex[ size ];

   //Clear the storage so that an allreduce can be performed in the end
   for ( int cnt = 0; cnt < size; cnt++ ) {
      two_rdm_A[ cnt ] = 0.0;
   }
   for ( int cnt = 0; cnt < size; cnt++ ) {
      two_rdm_B[ cnt ] = 0.0;
   }
}

CheMPS2::CTwoDM::~CTwoDM() {

   delete[] two_rdm_A;
   delete[] two_rdm_B;
}

void CheMPS2::CTwoDM::set_2rdm_A_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4, const dcomplex value ) {

   //Prob assumes you use DMRG orbs...
   //Irrep sanity checks are performed in TwoDM::FillSite
   two_rdm_A[ cnt1 + L * ( cnt2 + L * ( cnt3 + L * cnt4 ) ) ] = value;
   two_rdm_A[ cnt2 + L * ( cnt1 + L * ( cnt4 + L * cnt3 ) ) ] = value;
   two_rdm_A[ cnt3 + L * ( cnt4 + L * ( cnt1 + L * cnt2 ) ) ] = std::conj( value );
   two_rdm_A[ cnt4 + L * ( cnt3 + L * ( cnt2 + L * cnt1 ) ) ] = std::conj( value );
}

void CheMPS2::CTwoDM::set_2rdm_B_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4, const dcomplex value ) {

   //Prob assumes you use DMRG orbs...
   //Irrep sanity checks are performed in TwoDM::FillSite
   two_rdm_B[ cnt1 + L * ( cnt2 + L * ( cnt3 + L * cnt4 ) ) ] = value;
   two_rdm_B[ cnt2 + L * ( cnt1 + L * ( cnt4 + L * cnt3 ) ) ] = value;
   two_rdm_B[ cnt3 + L * ( cnt4 + L * ( cnt1 + L * cnt2 ) ) ] = std::conj( value );
   two_rdm_B[ cnt4 + L * ( cnt3 + L * ( cnt2 + L * cnt1 ) ) ] = std::conj( value );
}

dcomplex CheMPS2::CTwoDM::getTwoDMA_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const {

   //Prob assumes you use DMRG orbs...
   const int irrep1 = Prob->gIrrep( cnt1 );
   const int irrep2 = Prob->gIrrep( cnt2 );
   const int irrep3 = Prob->gIrrep( cnt3 );
   const int irrep4 = Prob->gIrrep( cnt4 );
   if ( Irreps::directProd( irrep1, irrep2 ) == Irreps::directProd( irrep3, irrep4 ) ) {
      return two_rdm_A[ cnt1 + L * ( cnt2 + L * ( cnt3 + L * cnt4 ) ) ];
   }

   return 0.0;
}

dcomplex CheMPS2::CTwoDM::getTwoDMB_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const {

   //Prob assumes you use DMRG orbs...
   const int irrep1 = Prob->gIrrep( cnt1 );
   const int irrep2 = Prob->gIrrep( cnt2 );
   const int irrep3 = Prob->gIrrep( cnt3 );
   const int irrep4 = Prob->gIrrep( cnt4 );
   if ( Irreps::directProd( irrep1, irrep2 ) == Irreps::directProd( irrep3, irrep4 ) ) {
      return two_rdm_B[ cnt1 + L * ( cnt2 + L * ( cnt3 + L * cnt4 ) ) ];
   }

   return 0.0;
}

dcomplex CheMPS2::CTwoDM::get1RDM_DMRG( const int cnt1, const int cnt2 ) const {

   //Prob assumes you use DMRG orbs...
   const int irrep1 = Prob->gIrrep( cnt1 );
   const int irrep2 = Prob->gIrrep( cnt2 );
   if ( irrep1 == irrep2 ) {
      dcomplex value = 0.0;
      for ( int orbsum = 0; orbsum < L; orbsum++ ) {
         value += getTwoDMA_DMRG( cnt1, orbsum, cnt2, orbsum );
      }
      value = value / ( Prob->gN() - 1.0 );
      return value;
   }

   return 0.0;
}

dcomplex CheMPS2::CTwoDM::spin_density_dmrg( const int cnt1, const int cnt2 ) const {

   //Prob assumes you use DMRG orbs...
   const int irrep1 = Prob->gIrrep( cnt1 );
   const int irrep2 = Prob->gIrrep( cnt2 );
   if ( irrep1 == irrep2 ) {
      const int two_s = Prob->gTwoS();
      if ( two_s > 0 ) {
         dcomplex value = static_cast< dcomplex >( 2 - Prob->gN() ) * get1RDM_DMRG( cnt1, cnt2 );
         for ( int orb = 0; orb < Prob->gL(); orb++ ) {
            value -= ( getTwoDMA_DMRG( cnt1, orb, orb, cnt2 ) + getTwoDMB_DMRG( cnt1, orb, orb, cnt2 ) );
         }
         value = 1.5 * value / ( 0.5 * two_s + 1 );
         return value;
      }
   }

   return 0.0;
}

dcomplex CheMPS2::CTwoDM::getTwoDMA_HAM( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const {

   //Prob assumes you use DMRG orbs... f1 converts HAM orbs to DMRG orbs
   if ( Prob->gReorder() ) {
      return getTwoDMA_DMRG( Prob->gf1( cnt1 ), Prob->gf1( cnt2 ), Prob->gf1( cnt3 ), Prob->gf1( cnt4 ) );
   }
   return getTwoDMA_DMRG( cnt1, cnt2, cnt3, cnt4 );
}

dcomplex CheMPS2::CTwoDM::getTwoDMB_HAM( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const {

   //Prob assumes you use DMRG orbs... f1 converts HAM orbs to DMRG orbs
   if ( Prob->gReorder() ) {
      return getTwoDMB_DMRG( Prob->gf1( cnt1 ), Prob->gf1( cnt2 ), Prob->gf1( cnt3 ), Prob->gf1( cnt4 ) );
   }
   return getTwoDMB_DMRG( cnt1, cnt2, cnt3, cnt4 );
}

dcomplex CheMPS2::CTwoDM::get1RDM_HAM( const int cnt1, const int cnt2 ) const {

   //Prob assumes you use DMRG orbs... f1 converts HAM orbs to DMRG orbs
   if ( Prob->gReorder() ) {
      return get1RDM_DMRG( Prob->gf1( cnt1 ), Prob->gf1( cnt2 ) );
   }
   return get1RDM_DMRG( cnt1, cnt2 );
}

dcomplex CheMPS2::CTwoDM::spin_density_ham( const int cnt1, const int cnt2 ) const {

   //Prob assumes you use DMRG orbs... f1 converts HAM orbs to DMRG orbs
   if ( Prob->gReorder() ) {
      return spin_density_dmrg( Prob->gf1( cnt1 ), Prob->gf1( cnt2 ) );
   }
   return spin_density_dmrg( cnt1, cnt2 );
}

dcomplex CheMPS2::CTwoDM::trace() const {

   dcomplex val = 0.0;
   for ( int cnt1 = 0; cnt1 < L; cnt1++ ) {
      for ( int cnt2 = 0; cnt2 < L; cnt2++ ) {
         val += getTwoDMA_DMRG( cnt1, cnt2, cnt1, cnt2 );
      }
   }
   return val;
}

dcomplex CheMPS2::CTwoDM::energy() const {

   dcomplex val = 0.0;
   for ( int cnt1 = 0; cnt1 < L; cnt1++ ) {
      for ( int cnt2 = 0; cnt2 < L; cnt2++ ) {
         for ( int cnt3 = 0; cnt3 < L; cnt3++ ) {
            for ( int cnt4 = 0; cnt4 < L; cnt4++ ) {
               val += getTwoDMA_DMRG( cnt1, cnt2, cnt3, cnt4 ) * Prob->gMxElement( cnt1, cnt2, cnt3, cnt4 );
            }
         }
      }
   }
   val *= 0.5;
   return val + Prob->gEconst();
}

// void CheMPS2::CTwoDM::print_noon() const {

//    int lwork         = 3 * L;
//    dcomplex * OneRDM = new dcomplex[ L * L ];
//    dcomplex * work   = new dcomplex[ lwork ];
//    dcomplex * eigs   = new dcomplex[ L ];

//    Irreps my_irreps( Prob->gSy() );

//    for ( int irrep = 0; irrep < denBK->getNumberOfIrreps(); irrep++ ) {

//       int jump1 = 0;
//       for ( int orb1 = 0; orb1 < L; orb1++ ) {
//          if ( Prob->gIrrep( orb1 ) == irrep ) {
//             int jump2 = jump1;
//             for ( int orb2 = orb1; orb2 < L; orb2++ ) {
//                if ( Prob->gIrrep( orb2 ) == irrep ) {
//                   const dcomplex value        = get1RDM_DMRG( orb1, orb2 );
//                   OneRDM[ jump1 + L * jump2 ] = value;
//                   OneRDM[ jump2 + L * jump1 ] = value;
//                   jump2 += 1;
//                }
//             }
//             jump1 += 1;
//          }
//       }

//       if ( jump1 > 0 ) {
//          char jobz = 'N'; // Eigenvalues only
//          char uplo = 'U';
//          int lda   = L;
//          int info;
//          dsyev_( &jobz, &uplo, &jump1, OneRDM, &lda, eigs, work, &lwork, &info );
//          cout << "   NOON of irrep " << my_irreps.getIrrepName( irrep ) << " = [ ";
//          for ( int cnt = 0; cnt < jump1 - 1; cnt++ ) {
//             cout << eigs[ jump1 - 1 - cnt ] << " , ";
//          } // Print from large to small
//          cout << eigs[ 0 ] << " ]." << endl;
//       }
//    }
//    delete[] OneRDM;
//    delete[] work;
//    delete[] eigs;
// }

// void CheMPS2::CTwoDM::save_HAM( const string filename ) const {

//    // Create an array with the 2-RDM in the ORIGINAL HAM indices
//    const int total_size   = L * L * L * L;
//    dcomplex * local_array = new dcomplex[ total_size ];
//    for ( int ham4 = 0; ham4 < L; ham4++ ) {
//       for ( int ham3 = 0; ham3 < L; ham3++ ) {
//          for ( int ham2 = 0; ham2 < L; ham2++ ) {
//             for ( int ham1 = 0; ham1 < L; ham1++ ) {
//                local_array[ ham1 + L * ( ham2 + L * ( ham3 + L * ham4 ) ) ] = getTwoDMA_HAM( ham1, ham2, ham3, ham4 );
//             }
//          }
//       }
//    }

//    // (Re)create the HDF5 file with the 2-RDM
//    hid_t file_id      = H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
//    hsize_t dimarray   = total_size;
//    hid_t group_id     = H5Gcreate( file_id, "2-RDM", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
//    hid_t dataspace_id = H5Screate_simple( 1, &dimarray, NULL );
//    hid_t dataset_id   = H5Dcreate( group_id, "elements", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

//    H5Dwrite( dataset_id, H5T_NATIVE_dcomplex, H5S_ALL, H5S_ALL, H5P_DEFAULT, local_array );

//    H5Dclose( dataset_id );
//    H5Sclose( dataspace_id );
//    H5Gclose( group_id );
//    H5Fclose( file_id );

//    // Deallocate the array
//    delete[] local_array;

//    cout << "Saved the 2-RDM to the file " << filename << endl;
// }

// void CheMPS2::CTwoDM::save() const {

//    hid_t file_id    = H5Fcreate( CheMPS2::TWO_RDM_storagename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
//    hsize_t dimarray = L * L * L * L;
//    {
//       hid_t group_id = H5Gcreate( file_id, "two_rdm_A", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

//       hid_t dataspace_id = H5Screate_simple( 1, &dimarray, NULL );
//       hid_t dataset_id   = H5Dcreate( group_id, "elements", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
//       H5Dwrite( dataset_id, H5T_NATIVE_dcomplex, H5S_ALL, H5S_ALL, H5P_DEFAULT, two_rdm_A );

//       H5Dclose( dataset_id );
//       H5Sclose( dataspace_id );

//       H5Gclose( group_id );
//    }
//    {
//       hid_t group_id = H5Gcreate( file_id, "two_rdm_B", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

//       hid_t dataspace_id = H5Screate_simple( 1, &dimarray, NULL );
//       hid_t dataset_id   = H5Dcreate( group_id, "elements", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
//       H5Dwrite( dataset_id, H5T_NATIVE_dcomplex, H5S_ALL, H5S_ALL, H5P_DEFAULT, two_rdm_B );
//       H5Dclose( dataset_id );
//       H5Sclose( dataspace_id );

//       H5Gclose( group_id );
//    }
//    H5Fclose( file_id );
// }

// void CheMPS2::CTwoDM::read() {

//    hid_t file_id = H5Fopen( CheMPS2::TWO_RDM_storagename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
//    {
//       hid_t group_id = H5Gopen( file_id, "two_rdm_A", H5P_DEFAULT );

//       hid_t dataset_id = H5Dopen( group_id, "elements", H5P_DEFAULT );
//       H5Dread( dataset_id, H5T_NATIVE_dcomplex, H5S_ALL, H5S_ALL, H5P_DEFAULT, two_rdm_A );
//       H5Dclose( dataset_id );

//       H5Gclose( group_id );
//    }
//    {
//       hid_t group_id = H5Gopen( file_id, "two_rdm_B", H5P_DEFAULT );

//       hid_t dataset_id = H5Dopen( group_id, "elements", H5P_DEFAULT );
//       H5Dread( dataset_id, H5T_NATIVE_dcomplex, H5S_ALL, H5S_ALL, H5P_DEFAULT, two_rdm_B );
//       H5Dclose( dataset_id );

//       H5Gclose( group_id );
//    }
//    H5Fclose( file_id );

//    std::cout << "TwoDM::read : Everything loaded!" << std::endl;
// }

// void CheMPS2::CTwoDM::write2DMAfile( const string filename ) const {

//    int * psi2molpro = new int[ denBK->getNumberOfIrreps() ];
//    Irreps my_irreps( Prob->gSy() );
//    my_irreps.symm_psi2molpro( psi2molpro );

//    FILE * capturing;
//    capturing = fopen( filename.c_str(), "w" ); // "w" with fopen means truncate file
//    fprintf( capturing, " &2-RDM NORB= %d,NELEC= %d,MS2= %d,\n", L, Prob->gN(), Prob->gTwoS() );
//    fprintf( capturing, "  ORBSYM=" );
//    for ( int HamOrb = 0; HamOrb < L; HamOrb++ ) {
//       const int DMRGOrb = ( Prob->gReorder() ) ? Prob->gf1( HamOrb ) : HamOrb;
//       fprintf( capturing, "%d,", psi2molpro[ Prob->gIrrep( DMRGOrb ) ] );
//    }
//    fprintf( capturing, "\n  ISYM=%d,\n /\n", psi2molpro[ Prob->gIrrep() ] );
//    delete[] psi2molpro;

//    for ( int ham_p = 0; ham_p < L; ham_p++ ) {
//       const int dmrg_p = ( Prob->gReorder() ) ? Prob->gf1( ham_p ) : ham_p;
//       for ( int ham_q = 0; ham_q <= ham_p; ham_q++ ) { // p >= q
//          const int dmrg_q   = ( Prob->gReorder() ) ? Prob->gf1( ham_q ) : ham_q;
//          const int irrep_pq = Irreps::directProd( Prob->gIrrep( dmrg_p ), Prob->gIrrep( dmrg_q ) );
//          for ( int ham_r = 0; ham_r <= ham_p; ham_r++ ) { // p >= r
//             const int dmrg_r = ( Prob->gReorder() ) ? Prob->gf1( ham_r ) : ham_r;
//             for ( int ham_s = 0; ham_s <= ham_p; ham_s++ ) { // p >= s
//                const int dmrg_s   = ( Prob->gReorder() ) ? Prob->gf1( ham_s ) : ham_s;
//                const int irrep_rs = Irreps::directProd( Prob->gIrrep( dmrg_r ), Prob->gIrrep( dmrg_s ) );
//                if ( irrep_pq == irrep_rs ) {
//                   const int num_equal = ( ( ham_q == ham_p ) ? 1 : 0 ) + ( ( ham_r == ham_p ) ? 1 : 0 ) + ( ( ham_s == ham_p ) ? 1 : 0 );
//                   /*   1. p > q,r,s
//                        2. p==q > r,s
//                        3. p==r > q,s
//                        4. p==s > q,r
//                        5. p==q==r > s
//                        6. p==q==s > r
//                        7. p==r==s > q
//                        8. p==r==s==q
//                   While 2-4 are inequivalent ( num_equal == 1 ), 5-7 are equivalent ( num_equal == 2 ). Hence:  */
//                   if ( ( num_equal != 2 ) || ( ham_p > ham_s ) ) {
//                      const dcomplex value = getTwoDMA_DMRG( dmrg_p, dmrg_r, dmrg_q, dmrg_s );
//                      fprintf( capturing, " % 23.16E %3d %3d %3d %3d\n", value, ham_p + 1, ham_q + 1, ham_r + 1, ham_s + 1 );
//                   }
//                }
//             }
//          }
//       }
//    }

//    // 1-RDM in Hamiltonian indices with p >= q
//    const dcomplex prefactor = 1.0 / ( Prob->gN() - 1.0 );
//    for ( int ham_p = 0; ham_p < L; ham_p++ ) {
//       const int dmrg_p = ( Prob->gReorder() ) ? Prob->gf1( ham_p ) : ham_p;
//       for ( int ham_q = 0; ham_q <= ham_p; ham_q++ ) {
//          const int dmrg_q = ( Prob->gReorder() ) ? Prob->gf1( ham_q ) : ham_q;
//          if ( Prob->gIrrep( dmrg_p ) == Prob->gIrrep( dmrg_q ) ) {
//             dcomplex value = 0.0;
//             for ( int orbsum = 0; orbsum < L; orbsum++ ) {
//                value += getTwoDMA_DMRG( dmrg_p, orbsum, dmrg_q, orbsum );
//             }
//             value *= prefactor;
//             fprintf( capturing, " % 23.16E %3d %3d %3d %3d\n", value, ham_p + 1, ham_q + 1, 0, 0 );
//          }
//       }
//    }

//    // 0-RDM == Norm of the wavefunction?
//    fprintf( capturing, " % 23.16E %3d %3d %3d %3d", 1.0, 0, 0, 0, 0 );
//    fclose( capturing );
//    cout << "Created the file " << filename << "." << endl;
// }

void CheMPS2::CTwoDM::FillSite( CTensorT * denT, CTensorL *** Ltens, CTensorF0 **** F0tens, CTensorF1 **** F1tens, CTensorS0 **** S0tens, CTensorS1 **** S1tens ) {

#ifdef CHEMPS2_MPI_COMPILATION
   const int MPIRANK = MPIchemps2::mpi_rank();
#endif

   const int theindex = denT->gIndex();
   const int DIM      = max( denBK->gMaxDimAtBound( theindex ), denBK->gMaxDimAtBound( theindex + 1 ) );

#ifdef CHEMPS2_MPI_COMPILATION
   if ( MPIRANK == MPI_CHEMPS2_MASTER )
#endif
   {
      //Diagram 1
      const dcomplex d1 = doD1( denT );
      set_2rdm_A_DMRG( theindex, theindex, theindex, theindex, 2.0 * d1 );
      set_2rdm_B_DMRG( theindex, theindex, theindex, theindex, -2.0 * d1 );
   }

#pragma omp parallel
   {

      dcomplex * workmem  = new dcomplex[ DIM * DIM ];
      dcomplex * workmem2 = new dcomplex[ DIM * DIM ];

#pragma omp for schedule( static ) nowait
      for ( int j_index = theindex + 1; j_index < L; j_index++ ) {
         if ( denBK->gIrrep( j_index ) == denBK->gIrrep( theindex ) ) {
#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_q( L, j_index ) ) //Everyone owns the L-tensors --> task division based on Q-tensor ownership
#endif
            {
               //Diagram 2
               const dcomplex d2 = doD2( denT, Ltens[ theindex ][ j_index - theindex - 1 ], workmem );
               set_2rdm_A_DMRG( theindex, j_index, theindex, theindex, 2.0 * d2 );
               set_2rdm_B_DMRG( theindex, j_index, theindex, theindex, -2.0 * d2 );
            }
         }
      }

      const int dimTriangle        = L - theindex - 1;
      const int upperboundTriangle = ( dimTriangle * ( dimTriangle + 1 ) ) / 2;
      int result[ 2 ];
#pragma omp for schedule( static ) nowait
      for ( int global = 0; global < upperboundTriangle; global++ ) {
         Special::invert_triangle_two( global, result );
         const int j_index = L - 1 - result[ 1 ];
         const int k_index = j_index + result[ 0 ];
         if ( denBK->gIrrep( j_index ) == denBK->gIrrep( k_index ) ) {

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_absigma( j_index, k_index ) )
#endif
            {
               //Diagram 3
               const dcomplex d3 = doD3( denT, S0tens[ theindex ][ k_index - j_index ][ j_index - theindex - 1 ], workmem );
               set_2rdm_A_DMRG( theindex, theindex, j_index, k_index, 2.0 * d3 );
               set_2rdm_B_DMRG( theindex, theindex, j_index, k_index, -2.0 * d3 );
            }

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_cdf( L, j_index, k_index ) )
#endif
            {
               //Diagrams 4,5 & 6
               const dcomplex d4 = doD4( denT, F0tens[ theindex ][ k_index - j_index ][ j_index - theindex - 1 ], workmem );
               const dcomplex d5 = doD5( denT, F0tens[ theindex ][ k_index - j_index ][ j_index - theindex - 1 ], workmem );
               const dcomplex d6 = doD6( denT, F1tens[ theindex ][ k_index - j_index ][ j_index - theindex - 1 ], workmem );
               set_2rdm_A_DMRG( theindex, j_index, k_index, theindex, -2.0 * d4 - 2.0 * d5 - 3.0 * d6 );
               set_2rdm_B_DMRG( theindex, j_index, k_index, theindex, -2.0 * d4 - 2.0 * d5 + d6 );
               set_2rdm_A_DMRG( theindex, j_index, theindex, k_index, 4.0 * d4 + 4.0 * d5 );
               set_2rdm_B_DMRG( theindex, j_index, theindex, k_index, 2.0 * d6 );
            }
         }
      }

#pragma omp for schedule( static ) nowait
      for ( int g_index = 0; g_index < theindex; g_index++ ) {
         if ( denBK->gIrrep( g_index ) == denBK->gIrrep( theindex ) ) {
#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_q( L, g_index ) ) //Everyone owns the L-tensors --> task division based on Q-tensor ownership
#endif
            {
               //Diagram 7
               const dcomplex d7 = doD7( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], workmem );
               set_2rdm_A_DMRG( g_index, theindex, theindex, theindex, 2.0 * d7 );
               set_2rdm_B_DMRG( g_index, theindex, theindex, theindex, -2.0 * d7 );
            }
         }
      }

      const int globalsize8to12 = theindex * ( L - 1 - theindex );
#pragma omp for schedule( static ) nowait
      for ( int gj_index = 0; gj_index < globalsize8to12; gj_index++ ) {
         const int g_index = gj_index % theindex;
         const int j_index = ( gj_index / theindex ) + theindex + 1;
         const int I_g     = denBK->gIrrep( g_index );
         if ( denBK->gIrrep( g_index ) == denBK->gIrrep( j_index ) ) {
#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_absigma( g_index, j_index ) ) //Everyone owns the L-tensors --> task division based on ABSigma-tensor ownership
#endif
            {
               //Diagrams 8,9,10 & 11
               const dcomplex d8 = doD8( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], Ltens[ theindex ][ j_index - theindex - 1 ], workmem, workmem2, I_g );
               dcomplex d9, d10, d11;
               doD9andD10andD11( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], Ltens[ theindex ][ j_index - theindex - 1 ], workmem, workmem2, &d9, &d10, &d11, I_g );
               set_2rdm_A_DMRG( g_index, theindex, j_index, theindex, -4.0 * d8 - d9 );
               set_2rdm_A_DMRG( g_index, theindex, theindex, j_index, 2.0 * d8 + d11 );
               set_2rdm_B_DMRG( g_index, theindex, j_index, theindex, d9 - 2.0 * d10 );
               set_2rdm_B_DMRG( g_index, theindex, theindex, j_index, 2.0 * d8 + 2.0 * d10 - d11 );

               //Diagram 12
               const dcomplex d12 = doD12( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], Ltens[ theindex ][ j_index - theindex - 1 ], workmem, workmem2, I_g );
               set_2rdm_A_DMRG( g_index, j_index, theindex, theindex, 2.0 * d12 );
               set_2rdm_B_DMRG( g_index, j_index, theindex, theindex, -2.0 * d12 );
            }
         }
      }

      const int globalsize = theindex * upperboundTriangle;
#pragma omp for schedule( static ) nowait
      for ( int gjk_index = 0; gjk_index < globalsize; gjk_index++ ) {
         const int g_index = gjk_index % theindex;
         const int global  = gjk_index / theindex;
         Special::invert_triangle_two( global, result );
         const int j_index = L - 1 - result[ 1 ];
         const int k_index = j_index + result[ 0 ];
         const int I_g     = denBK->gIrrep( g_index );
         const int cnt1    = k_index - j_index;
         const int cnt2    = j_index - theindex - 1;

         if ( Irreps::directProd( I_g, denBK->gIrrep( theindex ) ) == Irreps::directProd( denBK->gIrrep( j_index ), denBK->gIrrep( k_index ) ) ) {
#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_absigma( j_index, k_index ) )
#endif
            {
               //Diagrams 13,14,15 & 16
               const dcomplex d13 = doD13( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], S0tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g );
               const dcomplex d14 = doD14( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], S0tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g );
               dcomplex d15       = 0.0;
               dcomplex d16       = 0.0;
               if ( k_index > j_index ) {
                  d15 = doD15( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], S1tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g );
                  d16 = doD16( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], S1tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g );
               }
               set_2rdm_A_DMRG( g_index, theindex, j_index, k_index, 2.0 * d13 + 2.0 * d14 + 3.0 * d15 + 3.0 * d16 );
               set_2rdm_A_DMRG( g_index, theindex, k_index, j_index, 2.0 * d13 + 2.0 * d14 - 3.0 * d15 - 3.0 * d16 );
               set_2rdm_B_DMRG( g_index, theindex, j_index, k_index, -2.0 * d13 - 2.0 * d14 + d15 + d16 );
               set_2rdm_B_DMRG( g_index, theindex, k_index, j_index, -2.0 * d13 - 2.0 * d14 - d15 - d16 );
            }

#ifdef CHEMPS2_MPI_COMPILATION
            if ( MPIRANK == MPIchemps2::owner_cdf( L, j_index, k_index ) )
#endif
            {
               //Diagrams 17,18,19 & 20
               const dcomplex d17 = doD17orD21( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F0tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, true );
               const dcomplex d18 = doD18orD22( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F0tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, true );
               const dcomplex d19 = doD19orD23( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F1tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, true );
               const dcomplex d20 = doD20orD24( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F1tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, true );
               set_2rdm_A_DMRG( g_index, j_index, k_index, theindex, -2.0 * d17 - 2.0 * d18 - 3.0 * d19 - 3.0 * d20 );
               set_2rdm_A_DMRG( g_index, j_index, theindex, k_index, 4.0 * d17 + 4.0 * d18 );
               set_2rdm_B_DMRG( g_index, j_index, k_index, theindex, -2.0 * d17 - 2.0 * d18 + d19 + d20 );
               set_2rdm_B_DMRG( g_index, j_index, theindex, k_index, 2.0 * d19 + 2.0 * d20 );

               //Diagrams 21,22,23 & 24
               const dcomplex d21 = doD17orD21( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F0tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, false );
               const dcomplex d22 = doD18orD22( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F0tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, false );
               const dcomplex d23 = doD19orD23( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F1tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, false );
               const dcomplex d24 = doD20orD24( denT, Ltens[ theindex - 1 ][ theindex - g_index - 1 ], F1tens[ theindex ][ cnt1 ][ cnt2 ], workmem, workmem2, I_g, false );
               set_2rdm_A_DMRG( g_index, k_index, j_index, theindex, -2.0 * d21 - 2.0 * d22 - 3.0 * d23 - 3.0 * d24 );
               set_2rdm_A_DMRG( g_index, k_index, theindex, j_index, 4.0 * d21 + 4.0 * d22 );
               set_2rdm_B_DMRG( g_index, k_index, j_index, theindex, -2.0 * d21 - 2.0 * d22 + d23 + d24 );
               set_2rdm_B_DMRG( g_index, k_index, theindex, j_index, 2.0 * d23 + 2.0 * d24 );
            }
         }
      }

      delete[] workmem;
      delete[] workmem2;
   }
}

void CheMPS2::CTwoDM::correct_higher_multiplicities() {

   if ( Prob->gTwoS() != 0 ) {
      dcomplex alpha = 1.0 / ( Prob->gTwoS() + 1.0 );
      int length     = L * L * L * L;
      int inc        = 1;
      zscal_( &length, &alpha, two_rdm_A, &inc );
      zscal_( &length, &alpha, two_rdm_B, &inc );
   }
}

dcomplex CheMPS2::CTwoDM::doD1( CTensorT * denT ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( theindex, NL ); TwoSL <= denBK->gTwoSmax( theindex, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            int dimL = denBK->gCurrentDim( theindex, NL, TwoSL, IL );
            int dimR = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSL, IL );
            if ( ( dimL > 0 ) && ( dimR > 0 ) ) {

               dcomplex * Tblock = denT->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );

               int length = dimL * dimR;
               int inc    = 1;

               #ifdef CHEMPS2_MKL
               dcomplex result;
               zdotc_( &result, &length , Tblock , &inc , Tblock , &inc );
               #else
                  const dcomplex result = zdotc_( &length , Tblock , &inc , Tblock , &inc );
               #endif

               total += static_cast< dcomplex >( TwoSL + 1 ) * std::conj( result );
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD2( CTensorT * denT, CTensorL * Lright, dcomplex * workmem ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( theindex, NL ); TwoSL <= denBK->gTwoSmax( theindex, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            for ( int TwoSR = TwoSL - 1; TwoSR <= TwoSL + 1; TwoSR += 2 ) {

               int IRup = Irreps::directProd( IL, denBK->gIrrep( theindex ) );

               int dimL     = denBK->gCurrentDim( theindex, NL, TwoSL, IL );
               int dimRdown = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSL, IL );
               int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSR, IRup );

               if ( ( dimL > 0 ) && ( dimRup > 0 ) && ( dimRdown > 0 ) ) {

                  dcomplex * Tdown  = denT->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );
                  dcomplex * Tup    = denT->gStorage( NL, TwoSL, IL, NL + 1, TwoSR, IRup );
                  dcomplex * Lblock = Lright->gStorage( NL + 1, TwoSR, IRup, NL + 2, TwoSL, IL );

                  char trans     = 'C';
                  char notrans   = 'N';
                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0; //set
                  zgemm_( &notrans, &trans, &dimL, &dimRup, &dimRdown, &alpha, Tdown, &dimL, Lblock, &dimRup, &beta, workmem, &dimL );

                  const dcomplex factor = Special::phase( TwoSL + 1 - TwoSR ) * 0.5 * sqrt( ( TwoSL + 1 ) * ( TwoSR + 1.0 ) );

                  int length = dimL * dimRup;
                  int inc    = 1;

                  #ifdef CHEMPS2_MKL
                  dcomplex result;
                  zdotc_( &result, &length , workmem , &inc , Tup , &inc );
                  #else
                     const dcomplex result = zdotc_( &length , workmem , &inc , Tup , &inc );
                  #endif

                  total += factor * std::conj( result );
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD3( CTensorT * denT, CTensorS0 * S0right, dcomplex * workmem ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( theindex, NL ); TwoSL <= denBK->gTwoSmax( theindex, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {

            int dimL     = denBK->gCurrentDim( theindex, NL, TwoSL, IL );
            int dimRdown = denBK->gCurrentDim( theindex + 1, NL, TwoSL, IL );
            int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSL, IL );

            if ( ( dimL > 0 ) && ( dimRup > 0 ) && ( dimRdown > 0 ) ) {

               dcomplex * Tdown   = denT->gStorage( NL, TwoSL, IL, NL, TwoSL, IL );
               dcomplex * Tup     = denT->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );
               dcomplex * S0block = S0right->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );

               char notrans   = 'N';
               dcomplex alpha = 1.0;
               dcomplex beta  = 0.0; //set
               zgemm_( &notrans, &notrans, &dimL, &dimRup, &dimRdown, &alpha, Tdown, &dimL, S0block, &dimRdown, &beta, workmem, &dimL );

               dcomplex factor = sqrt( 0.5 ) * ( TwoSL + 1 );

               int length = dimL * dimRup;
               int inc    = 1;

               #ifdef CHEMPS2_MKL
               dcomplex result;
               zdotc_( &result, &length , workmem , &inc , Tup , &inc );
               #else
                  const dcomplex result = zdotc_( &length , workmem , &inc , Tup , &inc );
               #endif

               total += factor * std::conj( result );
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD4( CTensorT * denT, CTensorF0 * F0right, dcomplex * workmem ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( theindex, NL ); TwoSL <= denBK->gTwoSmax( theindex, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {

            int dimL = denBK->gCurrentDim( theindex, NL, TwoSL, IL );
            int dimR = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSL, IL );

            if ( ( dimL > 0 ) && ( dimR > 0 ) ) {

               dcomplex * Tblock  = denT->gStorage( NL, TwoSL, IL, NL + 2, TwoSL, IL );
               dcomplex * F0block = F0right->gStorage( NL + 2, TwoSL, IL, NL + 2, TwoSL, IL );

               char notrans   = 'N';
               dcomplex alpha = 1.0;
               dcomplex beta  = 0.0; //set
               zgemm_( &notrans, &notrans, &dimL, &dimR, &dimR, &alpha, Tblock, &dimL, F0block, &dimR, &beta, workmem, &dimL );

               dcomplex factor = sqrt( 0.5 ) * ( TwoSL + 1.0 );

               int length = dimL * dimR;
               int inc    = 1;

               #ifdef CHEMPS2_MKL
               dcomplex result;
               zdotc_( &result, &length , workmem , &inc , Tblock , &inc );
               #else
                  const dcomplex result = zdotc_( &length , workmem , &inc , Tblock , &inc );
               #endif

               total += factor * std::conj( result );
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD5( CTensorT * denT, CTensorF0 * F0right, dcomplex * workmem ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( theindex, NL ); TwoSL <= denBK->gTwoSmax( theindex, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            for ( int TwoSR = TwoSL - 1; TwoSR <= TwoSL + 1; TwoSR += 2 ) {

               int IR   = Irreps::directProd( IL, denBK->gIrrep( theindex ) );
               int dimL = denBK->gCurrentDim( theindex, NL, TwoSL, IL );
               int dimR = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSR, IR );

               if ( ( dimL > 0 ) && ( dimR > 0 ) ) {

                  dcomplex * Tblock  = denT->gStorage( NL, TwoSL, IL, NL + 1, TwoSR, IR );
                  dcomplex * F0block = F0right->gStorage( NL + 1, TwoSR, IR, NL + 1, TwoSR, IR );

                  char notrans   = 'N';
                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0; //set
                  zgemm_( &notrans, &notrans, &dimL, &dimR, &dimR, &alpha, Tblock, &dimL, F0block, &dimR, &beta, workmem, &dimL );

                  dcomplex factor = 0.5 * sqrt( 0.5 ) * ( TwoSR + 1 );

                  int length = dimL * dimR;
                  int inc    = 1;

                  #ifdef CHEMPS2_MKL
                  dcomplex result;
                  zdotc_( &result, &length , workmem , &inc , Tblock , &inc );
                  #else
                     const dcomplex result = zdotc_( &length , workmem , &inc , Tblock , &inc );
                  #endif

                  total += factor * std::conj( result );
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD6( CTensorT * denT, CTensorF1 * F1right, dcomplex * workmem ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSL = denBK->gTwoSmin( theindex, NL ); TwoSL <= denBK->gTwoSmax( theindex, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < denBK->getNumberOfIrreps(); IL++ ) {
            for ( int TwoSRup = TwoSL - 1; TwoSRup <= TwoSL + 1; TwoSRup += 2 ) {
               for ( int TwoSRdown = TwoSL - 1; TwoSRdown <= TwoSL + 1; TwoSRdown += 2 ) {

                  int IR       = Irreps::directProd( IL, denBK->gIrrep( theindex ) );
                  int dimL     = denBK->gCurrentDim( theindex, NL, TwoSL, IL );
                  int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSRup, IR );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSRdown, IR );

                  if ( ( dimL > 0 ) && ( dimRup > 0 ) && ( dimRdown > 0 ) ) {

                     dcomplex * Tup     = denT->gStorage( NL, TwoSL, IL, NL + 1, TwoSRup, IR );
                     dcomplex * Tdown   = denT->gStorage( NL, TwoSL, IL, NL + 1, TwoSRdown, IR );
                     dcomplex * F1block = F1right->gStorage( NL + 1, TwoSRdown, IR, NL + 1, TwoSRup, IR );

                     char notrans   = 'N';
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //set
                     zgemm_( &notrans, &notrans, &dimL, &dimRup, &dimRdown, &alpha, Tdown, &dimL, F1block, &dimRdown, &beta, workmem, &dimL );

                     const dcomplex factor = sqrt( ( TwoSRup + 1 ) / 3.0 ) * ( TwoSRdown + 1 ) * Special::phase( TwoSL + TwoSRdown - 1 ) * Wigner::wigner6j( 1, 1, 2, TwoSRup, TwoSRdown, TwoSL );

                     int length = dimL * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem , &inc , Tup , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem , &inc , Tup , &inc );
                     #endif

                     total += factor * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD7( CTensorT * denT, CTensorL * Lleft, dcomplex * workmem ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NLup = denBK->gNmin( theindex ); NLup <= denBK->gNmax( theindex ); NLup++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NLup ); TwoSLup <= denBK->gTwoSmax( theindex, NLup ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NLup, TwoSLup, ILup );

            for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

               int IR       = Irreps::directProd( ILup, denBK->gIrrep( theindex ) );
               int dimLdown = denBK->gCurrentDim( theindex, NLup - 1, TwoSLdown, IR );
               int dimR     = denBK->gCurrentDim( theindex + 1, NLup + 1, TwoSLdown, IR );

               if ( ( dimLup > 0 ) && ( dimLdown > 0 ) && ( dimR > 0 ) ) {

                  dcomplex * Tup    = denT->gStorage( NLup, TwoSLup, ILup, NLup + 1, TwoSLdown, IR );
                  dcomplex * Tdown  = denT->gStorage( NLup - 1, TwoSLdown, IR, NLup + 1, TwoSLdown, IR );
                  dcomplex * Lblock = Lleft->gStorage( NLup - 1, TwoSLdown, IR, NLup, TwoSLup, ILup );

                  char trans     = 'C';
                  char notrans   = 'N';
                  dcomplex alpha = 1.0;
                  dcomplex beta  = 0.0; //set
                  zgemm_( &trans, &notrans, &dimLup, &dimR, &dimLdown, &alpha, Lblock, &dimLdown, Tdown, &dimLdown, &beta, workmem, &dimLup );

                  const dcomplex factor = 0.5 * sqrt( ( TwoSLdown + 1 ) * ( TwoSLup + 1.0 ) ) * Special::phase( TwoSLup - TwoSLdown + 3 );

                  int length = dimLup * dimR;
                  int inc    = 1;

                  #ifdef CHEMPS2_MKL
                  dcomplex result;
                  zdotc_( &result, &length , workmem , &inc , Tup , &inc );
                  #else
                     const dcomplex result = zdotc_( &length , workmem , &inc , Tup , &inc );
                  #endif

                  total += factor * std::conj( result );
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD8( CTensorT * denT, CTensorL * Lleft, CTensorL * Lright, dcomplex * workmem, dcomplex * workmem2, int Irrep_g ) {

   int theindex = denT->gIndex();

   dcomplex total = 0.0;

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );
            int dimRup = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSLup, ILup );

            if ( ( dimLup > 0 ) && ( dimRup > 0 ) ) {

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

                  int Idown = Irreps::directProd( ILup, Irrep_g );

                  int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, Idown );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSLdown, Idown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     dcomplex * Tup       = denT->gStorage( NL, TwoSLup, ILup, NL + 2, TwoSLup, ILup );
                     dcomplex * Tdown     = denT->gStorage( NL - 1, TwoSLdown, Idown, NL + 1, TwoSLdown, Idown );
                     dcomplex * LleftBlk  = Lleft->gStorage( NL - 1, TwoSLdown, Idown, NL, TwoSLup, ILup );
                     dcomplex * LrightBlk = Lright->gStorage( NL + 1, TwoSLdown, Idown, NL + 2, TwoSLup, ILup );

                     char trans     = 'C';
                     char notrans   = 'N';
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //set
                     zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LleftBlk, &dimLdown, Tdown, &dimLdown, &beta, workmem, &dimLup );

                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, LrightBlk, &dimRdown, &beta, workmem2, &dimLup );

                     dcomplex factor = -0.5 * ( TwoSLup + 1 );

                     int length = dimLup * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem2, &inc , Tup , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem2, &inc , Tup , &inc );
                     #endif

                     total += factor * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return total;
}

void CheMPS2::CTwoDM::doD9andD10andD11( CTensorT * denT, CTensorL * Lleft, CTensorL * Lright, dcomplex * workmem, dcomplex * workmem2, dcomplex * d9, dcomplex * d10, dcomplex * d11, int Irrep_g ) {

   d9[ 0 ]  = 0.0;
   d10[ 0 ] = 0.0;
   d11[ 0 ] = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );
            if ( dimLup > 0 ) {

               int IRup   = Irreps::directProd( ILup, denBK->gIrrep( theindex ) );
               int ILdown = Irreps::directProd( ILup, Irrep_g );
               int IRdown = Irreps::directProd( ILdown, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {
                  for ( int TwoSRup = TwoSLup - 1; TwoSRup <= TwoSLup + 1; TwoSRup += 2 ) {
                     for ( int TwoSRdown = TwoSRup - 1; TwoSRdown <= TwoSRup + 1; TwoSRdown += 2 ) {
                        if ( ( TwoSLdown >= 0 ) && ( TwoSRup >= 0 ) && ( TwoSRdown >= 0 ) && ( abs( TwoSLdown - TwoSRdown ) <= 1 ) ) {

                           int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                           int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSRup, IRup );
                           int dimRdown = denBK->gCurrentDim( theindex + 1, NL, TwoSRdown, IRdown );

                           if ( ( dimLdown > 0 ) && ( dimRup > 0 ) && ( dimRdown > 0 ) ) {

                              dcomplex * T_up      = denT->gStorage( NL, TwoSLup, ILup, NL + 1, TwoSRup, IRup );
                              dcomplex * T_down    = denT->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSRdown, IRdown );
                              dcomplex * LleftBlk  = Lleft->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, ILup );
                              dcomplex * LrightBlk = Lright->gStorage( NL, TwoSRdown, IRdown, NL + 1, TwoSRup, IRup );

                              char trans     = 'C';
                              char notrans   = 'N';
                              dcomplex alpha = 1.0;
                              dcomplex beta  = 0.0; //SET

                              zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LleftBlk, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                              zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, LrightBlk, &dimRdown, &beta, workmem2, &dimLup );

                              int length     = dimLup * dimRup;
                              int inc        = 1;

                              #ifdef CHEMPS2_MKL
                              dcomplex result;
                              zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                              #else
                                 const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                              #endif

                              dcomplex value = std::conj( result );

                              const dcomplex fact1 = Special::phase( TwoSLup + TwoSRdown + 2 ) * ( TwoSRup + 1 ) * sqrt( ( TwoSRdown + 1 ) * ( TwoSLup + 1.0 ) ) * Wigner::wigner6j( TwoSRup, 1, TwoSLup, TwoSLdown, 1, TwoSRdown );
                              const dcomplex fact2 = 2 * ( TwoSRup + 1 ) * sqrt( ( TwoSRdown + 1 ) * ( TwoSLup + 1.0 ) ) * Wigner::wigner6j( TwoSRup, TwoSLdown, 2, 1, 1, TwoSLup ) * Wigner::wigner6j( TwoSRup, TwoSLdown, 2, 1, 1, TwoSRdown );
                              const int fact3      = ( ( TwoSRdown == TwoSLup ) ? ( TwoSRup + 1 ) : 0 );

                              d9[ 0 ] += fact1 * value;
                              d10[ 0 ] += fact2 * value;
                              d11[ 0 ] += static_cast< dcomplex >( fact3 ) * value;
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

dcomplex CheMPS2::CTwoDM::doD12( CTensorT * denT, CTensorL * Lleft, CTensorL * Lright, dcomplex * workmem, dcomplex * workmem2, int Irrep_g ) {

   dcomplex d12 = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );
            int dimRup = denBK->gCurrentDim( theindex + 1, NL, TwoSLup, ILup );
            if ( ( dimLup > 0 ) && ( dimRup > 0 ) ) {

               int Idown = Irreps::directProd( ILup, Irrep_g );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

                  int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, Idown );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSLdown, Idown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     dcomplex * T_up      = denT->gStorage( NL, TwoSLup, ILup, NL, TwoSLup, ILup );
                     dcomplex * T_down    = denT->gStorage( NL - 1, TwoSLdown, Idown, NL + 1, TwoSLdown, Idown );
                     dcomplex * LleftBlk  = Lleft->gStorage( NL - 1, TwoSLdown, Idown, NL, TwoSLup, ILup );
                     dcomplex * LrightBlk = Lright->gStorage( NL, TwoSLup, ILup, NL + 1, TwoSLdown, Idown );

                     char trans     = 'C';
                     char notrans   = 'N';
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //SET

                     zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, LleftBlk, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                     zgemm_( &notrans, &trans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, LrightBlk, &dimRup, &beta, workmem2, &dimLup );

                     const dcomplex factor = Special::phase( TwoSLdown + 1 - TwoSLup ) * 0.5 * sqrt( ( TwoSLup + 1 ) * ( TwoSLdown + 1.0 ) );

                     int length = dimLup * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                     #endif

                     d12 += factor * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return d12;
}

dcomplex CheMPS2::CTwoDM::doD13( CTensorT * denT, CTensorL * Lleft, CTensorS0 * S0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g ) {

   dcomplex d13 = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );
            int dimRup = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSLup, ILup );

            if ( ( dimLup > 0 ) && ( dimRup > 0 ) ) {

               int ILdown = Irreps::directProd( ILup, Irrep_g );
               int IRdown = Irreps::directProd( ILdown, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

                  int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL, TwoSLup, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                     dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL + 2, TwoSLup, ILup );
                     dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, IRdown );
                     dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, ILup );
                     dcomplex * S0block = S0right->gStorage( NL, TwoSLup, IRdown, NL + 2, TwoSLup, ILup );

                     char trans     = 'C';
                     char notrans   = 'N';
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //SET

                     zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, S0block, &dimRdown, &beta, workmem2, &dimLup );

                     dcomplex factor = -0.5 * sqrt( 0.5 ) * ( TwoSLup + 1 );

                     int length = dimLup * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                     #endif

                     d13 += factor * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return d13;
}

dcomplex CheMPS2::CTwoDM::doD14( CTensorT * denT, CTensorL * Lleft, CTensorS0 * S0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g ) {

   dcomplex d14 = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );

            if ( dimLup > 0 ) {

               int Idown = Irreps::directProd( ILup, Irrep_g );
               int IRup  = Irreps::directProd( ILup, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

                  int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSLdown, IRup );
                  int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, Idown );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL - 1, TwoSLdown, Idown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) && ( dimRup > 0 ) ) {

                     dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL + 1, TwoSLdown, IRup );
                     dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, Idown, NL - 1, TwoSLdown, Idown );
                     dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, Idown, NL, TwoSLup, ILup );
                     dcomplex * S0block = S0right->gStorage( NL - 1, TwoSLdown, Idown, NL + 1, TwoSLdown, IRup );

                     char trans     = 'C';
                     char notrans   = 'N';
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //SET

                     zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                     zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, S0block, &dimRdown, &beta, workmem2, &dimLup );

                     const dcomplex factor = Special::phase( TwoSLdown + 1 - TwoSLup ) * 0.5 * sqrt( 0.5 * ( TwoSLup + 1 ) * ( TwoSLdown + 1 ) );

                     int length = dimLup * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                     #endif

                     d14 += factor * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return d14;
}

dcomplex CheMPS2::CTwoDM::doD15( CTensorT * denT, CTensorL * Lleft, CTensorS1 * S1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g ) {

   dcomplex d15 = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );
            int dimRup = denBK->gCurrentDim( theindex + 1, NL + 2, TwoSLup, ILup );

            if ( ( dimLup > 0 ) && ( dimRup > 0 ) ) {

               int ILdown = Irreps::directProd( ILup, Irrep_g );
               int IRdown = Irreps::directProd( ILdown, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {
                  for ( int TwoSRdown = TwoSLdown - 1; TwoSRdown <= TwoSLdown + 1; TwoSRdown += 2 ) {

                     int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = denBK->gCurrentDim( theindex + 1, NL, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) ) {

                        dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL + 2, TwoSLup, ILup );
                        dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSRdown, IRdown );
                        dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, ILup );
                        dcomplex * S1block = S1right->gStorage( NL, TwoSRdown, IRdown, NL + 2, TwoSLup, ILup );

                        char trans     = 'C';
                        char notrans   = 'N';
                        dcomplex alpha = 1.0;
                        dcomplex beta  = 0.0; //SET

                        zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, S1block, &dimRdown, &beta, workmem2, &dimLup );

                        const dcomplex factor = Special::phase( TwoSLdown + TwoSLup + 1 ) * ( TwoSLup + 1 ) * sqrt( ( TwoSRdown + 1 ) / 3.0 ) * Wigner::wigner6j( 1, 1, 2, TwoSLup, TwoSRdown, TwoSLdown );

                        int length = dimLup * dimRup;
                        int inc    = 1;

                        #ifdef CHEMPS2_MKL
                        dcomplex result;
                        zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                        #else
                           const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                        #endif

                        d15 += factor * std::conj( result );
                     }
                  }
               }
            }
         }
      }
   }

   return d15;
}

dcomplex CheMPS2::CTwoDM::doD16( CTensorT * denT, CTensorL * Lleft, CTensorS1 * S1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g ) {

   dcomplex d16 = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );

            if ( dimLup > 0 ) {

               int Idown = Irreps::directProd( ILup, Irrep_g );
               int IRup  = Irreps::directProd( ILup, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {
                  for ( int TwoSRup = TwoSLup - 1; TwoSRup <= TwoSLup + 1; TwoSRup += 2 ) {

                     int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSRup, IRup );
                     int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, Idown );
                     int dimRdown = denBK->gCurrentDim( theindex + 1, NL - 1, TwoSLdown, Idown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) && ( dimRup > 0 ) ) {

                        dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL + 1, TwoSRup, IRup );
                        dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, Idown, NL - 1, TwoSLdown, Idown );
                        dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, Idown, NL, TwoSLup, ILup );
                        dcomplex * S1block = S1right->gStorage( NL - 1, TwoSLdown, Idown, NL + 1, TwoSRup, IRup );

                        char trans     = 'C';
                        char notrans   = 'N';
                        dcomplex alpha = 1.0;
                        dcomplex beta  = 0.0; //SET

                        zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                        zgemm_( &notrans, &notrans, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, S1block, &dimRdown, &beta, workmem2, &dimLup );

                        const dcomplex factor = Special::phase( TwoSRup + TwoSLdown + 2 ) * ( TwoSRup + 1 ) * sqrt( ( TwoSLup + 1 ) / 3.0 ) * Wigner::wigner6j( 1, 1, 2, TwoSRup, TwoSLdown, TwoSLup );

                        int length = dimLup * dimRup;
                        int inc    = 1;

                        #ifdef CHEMPS2_MKL
                        dcomplex result;
                        zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                        #else
                           const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                        #endif

                        d16 += factor * std::conj( result );
                     }
                  }
               }
            }
         }
      }
   }

   return d16;
}

dcomplex CheMPS2::CTwoDM::doD17orD21( CTensorT * denT, CTensorL * Lleft, CTensorF0 * F0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD17 ) {

   dcomplex total = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );

            if ( dimLup > 0 ) {

               int ILdown = Irreps::directProd( ILup, Irrep_g );
               int IRdown = Irreps::directProd( ILdown, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

                  int dimRup   = denBK->gCurrentDim( theindex + 1, NL, TwoSLup, ILup );
                  int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL, TwoSLup, IRdown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) && ( dimRup > 0 ) ) {

                     dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL, TwoSLup, ILup );
                     dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, IRdown );
                     dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, ILup );
                     dcomplex * F0block = ( shouldIdoD17 ) ? F0right->gStorage( NL, TwoSLup, IRdown, NL, TwoSLup, ILup )
                                                           : F0right->gStorage( NL, TwoSLup, ILup, NL, TwoSLup, IRdown );

                     char trans     = 'C';
                     char notrans   = 'N';
                     char var       = ( shouldIdoD17 ) ? notrans : trans;
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //SET
                     int dimvar     = ( shouldIdoD17 ) ? dimRdown : dimRup;

                     zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                     zgemm_( &notrans, &var, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, F0block, &dimvar, &beta, workmem2, &dimLup );

                     int length = dimLup * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                     #endif

                     total += sqrt( 0.5 ) * 0.5 * ( TwoSLup + 1 ) * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD18orD22( CTensorT * denT, CTensorL * Lleft, CTensorF0 * F0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD18 ) {

   dcomplex total = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );

            if ( dimLup > 0 ) {

               int Idown = Irreps::directProd( ILup, Irrep_g );
               int IRup  = Irreps::directProd( ILup, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {

                  int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSLdown, IRup );
                  int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, Idown );
                  int dimRdown = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSLdown, Idown );

                  if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) && ( dimRup > 0 ) ) {

                     dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL + 1, TwoSLdown, IRup );
                     dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, Idown, NL + 1, TwoSLdown, Idown );
                     dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, Idown, NL, TwoSLup, ILup );
                     dcomplex * F0block = ( shouldIdoD18 ) ? F0right->gStorage( NL + 1, TwoSLdown, Idown, NL + 1, TwoSLdown, IRup )
                                                           : F0right->gStorage( NL + 1, TwoSLdown, IRup, NL + 1, TwoSLdown, Idown );

                     char trans     = 'C';
                     char notrans   = 'N';
                     char var       = ( shouldIdoD18 ) ? notrans : trans;
                     dcomplex alpha = 1.0;
                     dcomplex beta  = 0.0; //SET
                     int dimvar     = ( shouldIdoD18 ) ? dimRdown : dimRup;

                     zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                     zgemm_( &notrans, &var, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, F0block, &dimvar, &beta, workmem2, &dimLup );

                     const dcomplex factor = Special::phase( TwoSLdown + 1 - TwoSLup ) * 0.5 * sqrt( 0.5 * ( TwoSLup + 1 ) * ( TwoSLdown + 1 ) );

                     int length = dimLup * dimRup;
                     int inc    = 1;

                     #ifdef CHEMPS2_MKL
                     dcomplex result;
                     zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                     #else
                        const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                     #endif

                     total += factor * std::conj( result );
                  }
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD19orD23( CTensorT * denT, CTensorL * Lleft, CTensorF1 * F1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD19 ) {

   dcomplex total = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );

            if ( dimLup > 0 ) {

               int ILdown = Irreps::directProd( ILup, Irrep_g );
               int IRdown = Irreps::directProd( ILdown, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {
                  for ( int TwoSRdown = TwoSLdown - 1; TwoSRdown <= TwoSLdown + 1; TwoSRdown += 2 ) {

                     int dimRup   = denBK->gCurrentDim( theindex + 1, NL, TwoSLup, ILup );
                     int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, ILdown );
                     int dimRdown = denBK->gCurrentDim( theindex + 1, NL, TwoSRdown, IRdown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) && ( dimRup > 0 ) ) {

                        dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL, TwoSLup, ILup );
                        dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSRdown, IRdown );
                        dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, ILdown, NL, TwoSLup, ILup );
                        dcomplex * F1block = ( shouldIdoD19 ) ? F1right->gStorage( NL, TwoSRdown, IRdown, NL, TwoSLup, ILup )
                                                              : F1right->gStorage( NL, TwoSLup, ILup, NL, TwoSRdown, IRdown );

                        char trans     = 'C';
                        char notrans   = 'N';
                        char var       = ( shouldIdoD19 ) ? notrans : trans;
                        dcomplex alpha = 1.0;
                        dcomplex beta  = 0.0; //SET
                        int dimvar     = ( shouldIdoD19 ) ? dimRdown : dimRup;

                        zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                        zgemm_( &notrans, &var, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, F1block, &dimvar, &beta, workmem2, &dimLup );

                        dcomplex factor = 0.0;
                        if ( shouldIdoD19 ) {
                           factor = Special::phase( TwoSLdown + TwoSRdown - 1 ) * ( TwoSRdown + 1 ) * sqrt( ( TwoSLup + 1 ) / 3.0 ) * Wigner::wigner6j( 1, 1, 2, TwoSLup, TwoSRdown, TwoSLdown );
                        } else {
                           factor = Special::phase( TwoSLdown + TwoSLup - 1 ) * ( TwoSLup + 1 ) * sqrt( ( TwoSRdown + 1 ) / 3.0 ) * Wigner::wigner6j( 1, 1, 2, TwoSLup, TwoSRdown, TwoSLdown );
                        }

                        int length = dimLup * dimRup;
                        int inc    = 1;

                        #ifdef CHEMPS2_MKL
                        dcomplex result;
                        zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                        #else
                           const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                        #endif

                        total += factor * std::conj( result );
                     }
                  }
               }
            }
         }
      }
   }

   return total;
}

dcomplex CheMPS2::CTwoDM::doD20orD24( CTensorT * denT, CTensorL * Lleft, CTensorF1 * F1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD20 ) {

   dcomplex total = 0.0;

   int theindex = denT->gIndex();

   for ( int NL = denBK->gNmin( theindex ); NL <= denBK->gNmax( theindex ); NL++ ) {
      for ( int TwoSLup = denBK->gTwoSmin( theindex, NL ); TwoSLup <= denBK->gTwoSmax( theindex, NL ); TwoSLup += 2 ) {
         for ( int ILup = 0; ILup < denBK->getNumberOfIrreps(); ILup++ ) {

            int dimLup = denBK->gCurrentDim( theindex, NL, TwoSLup, ILup );

            if ( dimLup > 0 ) {

               int Idown = Irreps::directProd( ILup, Irrep_g );
               int IRup  = Irreps::directProd( ILup, denBK->gIrrep( theindex ) );

               for ( int TwoSLdown = TwoSLup - 1; TwoSLdown <= TwoSLup + 1; TwoSLdown += 2 ) {
                  for ( int TwoSRup = TwoSLup - 1; TwoSRup <= TwoSLup + 1; TwoSRup += 2 ) {

                     int dimRup   = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSRup, IRup );
                     int dimLdown = denBK->gCurrentDim( theindex, NL - 1, TwoSLdown, Idown );
                     int dimRdown = denBK->gCurrentDim( theindex + 1, NL + 1, TwoSLdown, Idown );

                     if ( ( dimLdown > 0 ) && ( dimRdown > 0 ) && ( dimRup > 0 ) ) {

                        dcomplex * T_up    = denT->gStorage( NL, TwoSLup, ILup, NL + 1, TwoSRup, IRup );
                        dcomplex * T_down  = denT->gStorage( NL - 1, TwoSLdown, Idown, NL + 1, TwoSLdown, Idown );
                        dcomplex * Lblock  = Lleft->gStorage( NL - 1, TwoSLdown, Idown, NL, TwoSLup, ILup );
                        dcomplex * F1block = ( shouldIdoD20 ) ? F1right->gStorage( NL + 1, TwoSLdown, Idown, NL + 1, TwoSRup, IRup )
                                                              : F1right->gStorage( NL + 1, TwoSRup, IRup, NL + 1, TwoSLdown, Idown );

                        char trans     = 'C';
                        char notrans   = 'N';
                        char var       = ( shouldIdoD20 ) ? notrans : trans;
                        dcomplex alpha = 1.0;
                        dcomplex beta  = 0.0; //SET
                        int dimvar     = ( shouldIdoD20 ) ? dimRdown : dimRup;

                        zgemm_( &trans, &notrans, &dimLup, &dimRdown, &dimLdown, &alpha, Lblock, &dimLdown, T_down, &dimLdown, &beta, workmem, &dimLup );

                        zgemm_( &notrans, &var, &dimLup, &dimRup, &dimRdown, &alpha, workmem, &dimLup, F1block, &dimvar, &beta, workmem2, &dimLup );

                        dcomplex factor = 0.0;
                        if ( shouldIdoD20 ) {
                           factor = Special::phase( 2 * TwoSLup ) * sqrt( ( TwoSLup + 1 ) * ( TwoSRup + 1 ) * ( TwoSLdown + 1 ) / 3.0 ) * Wigner::wigner6j( 1, 1, 2, TwoSRup, TwoSLdown, TwoSLup );
                        } else {
                           factor = Special::phase( 2 * TwoSLup + TwoSRup - TwoSLdown ) * ( TwoSRup + 1 ) * sqrt( ( TwoSLup + 1 ) / 3.0 ) * Wigner::wigner6j( 1, 1, 2, TwoSRup, TwoSLdown, TwoSLup );
                        }

                        int length = dimLup * dimRup;
                        int inc    = 1;

                        #ifdef CHEMPS2_MKL
                        dcomplex result;
                        zdotc_( &result, &length , workmem2, &inc , T_up , &inc );
                        #else
                           const dcomplex result = zdotc_( &length , workmem2, &inc , T_up , &inc );
                        #endif

                        total += factor * std::conj( result );
                     }
                  }
               }
            }
         }
      }
   }

   return total;
}
