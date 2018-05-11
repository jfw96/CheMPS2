
#include "TimeEvolution.h"
#include <assert.h>
#include <iomanip>
#include <iostream>

#include "COneDM.h"
#include "HamiltonianOperator.h"
#include "Lapack.h"
#include "TwoDMBuilder.h"

CheMPS2::TimeEvolution::TimeEvolution( Problem * probIn, ConvergenceScheme * schemeIn, hid_t HDF5FILEIDIN )
    : prob( probIn ), scheme( schemeIn ), HDF5FILEID( HDF5FILEIDIN ), L( probIn->gL() ) {

   start        = time( NULL );
   tm * tmstart = localtime( &start );
   std::ostringstream text;

   std::cout << hashline;
   std::cout << "### "
             << "Starting to run a time evolution calculation"
             << " on " << tmstart->tm_year + 1900 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mon + 1 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mday << " ";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_hour << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_min << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_sec << "\n";

   std::cout << hashline;

   assert( probIn->checkConsistency() );

   prob->construct_mxelem();

   hid_t inputGroupID             = ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) ? H5Gcreate( HDF5FILEID, "/Input", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   hid_t systemPropertiesID       = ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) ? H5Gcreate( HDF5FILEID, "/Input/SystemProperties", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   hid_t waveFunctionPropertiesID = ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) ? H5Gcreate( HDF5FILEID, "/Input/WaveFunctionProperties", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   hsize_t dimarray1              = 1;

   std::cout << "\n";
   std::cout << "   System Properties\n";
   std::cout << "\n";

   std::cout << "\n";
   std::cout << "   L = " << L << "\n";
   HDF5_MAKE_DATASET( systemPropertiesID, "L", 1, &dimarray1, H5T_STD_I32LE, &L );

   int Sy = prob->gSy();
   std::cout << "   Sy = " << Sy << "\n";
   HDF5_MAKE_DATASET( systemPropertiesID, "Sy", 1, &dimarray1, H5T_STD_I32LE, &Sy );

   std::cout << "   Irreps =";
   int * irreps = new int[ L ];
   for ( int i = 0; i < L; i++ ) {
      irreps[ i ] = prob->gIrrep( i );
      std::cout << std::setfill( ' ' ) << std::setw( 10 ) << irreps[ i ];
   }
   std::cout << "\n";
   hsize_t Lsize = L;
   HDF5_MAKE_DATASET( systemPropertiesID, "Irrep", 1, &Lsize, H5T_STD_I32LE, irreps );
   delete[] irreps;

   double Econst = prob->gEconst();
   HDF5_MAKE_DATASET( systemPropertiesID, "Econst", 1, &dimarray1, H5T_NATIVE_DOUBLE, &Econst );

   std::cout << "\n";

   int N = prob->gN();
   std::cout << "   N = " << N << "\n";
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "N", 1, &dimarray1, H5T_STD_I32LE, &N );

   int TwoS = prob->gTwoS();
   std::cout << "   TwoS = " << TwoS << "\n";
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "TwoS", 1, &dimarray1, H5T_STD_I32LE, &TwoS );

   int I = prob->gIrrep();
   std::cout << "   I = " << I << "\n";
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "I", 1, &dimarray1, H5T_STD_I32LE, &I );

   std::cout << "\n";

   std::cout << "   Orbital ordering:\n";
   std::cout << "      ";
   int * ham2dmrg = new int[ L + 1 ];
   for ( int i = 0; i < L; i++ ) {
      ham2dmrg[ i ] = prob->gf2( i );
      std::cout << std::setfill( ' ' ) << std::setw( 2 ) << i << " -> " << std::setfill( ' ' ) << std::setw( 1 ) << ham2dmrg[ i ] << "  ";
   }
   std::cout << "\n";
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "Ham2DMRG", 1, &Lsize, H5T_STD_I32LE, ham2dmrg );
   delete[] ham2dmrg;

   std::cout << "\n";

   std::cout << "   full ci matrix product state dimensions:\n";
   std::cout << "   ";
   for ( int i = 0; i < L + 1; i++ ) {
      std::cout << std::setfill( ' ' ) << std::setw( 10 ) << i;
   }
   std::cout << "\n";
   std::cout << "   ";

   CheMPS2::SyBookkeeper * tempBK = new CheMPS2::SyBookkeeper( prob, 0 );
   int * fcidims                  = new int[ L + 1 ];
   for ( int i = 0; i < L + 1; i++ ) {
      fcidims[ i ] = tempBK->gFCIDimAtBound( i );
      std::cout << std::setfill( ' ' ) << std::setw( 10 ) << fcidims[ i ];
   }
   std::cout << "\n";
   hsize_t Lposize = L + 1;
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "FCIDims", 1, &Lposize, H5T_STD_I32LE, fcidims );
   delete[] fcidims;
   delete tempBK;

   std::cout << "\n";
   std::cout << hashline;
}

CheMPS2::TimeEvolution::~TimeEvolution() {

   std::time_t now = time( NULL );
   tm * tmstart    = localtime( &now );
   std::ostringstream text;

   std::cout << hashline;
   std::cout << "### "
             << "Finished to run a time evolution calculation"
             << " on " << tmstart->tm_year + 1900 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mon + 1 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mday << " ";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_hour << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_min << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_sec << "\n";

   std::cout << "### Calculation took " << ( time( NULL ) - start ) / 60.0 << " minutes\n"
             << hashline;

   std::cout << hashline;
}

void CheMPS2::TimeEvolution::HDF5_MAKE_DATASET( hid_t setID, const char * name, int rank, const hsize_t * dims, hid_t typeID, const void * data ) {
   if ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) {
      H5LTmake_dataset( setID, name, rank, dims, typeID, data );
   }
}

void CheMPS2::TimeEvolution::doStep_arnoldi( const double time_step, const double time_final, const int kry_size, const bool doImaginary, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   int krylovSpaceDimension = kry_size;

   dcomplex step = doImaginary ? -time_step : dcomplex( 0.0, -1.0 * time_step );

   HamiltonianOperator * op = new HamiltonianOperator( prob );

   CTensorT *** krylovBasisVectors          = new CTensorT **[ krylovSpaceDimension ];
   SyBookkeeper ** krylovBasisSyBookkeepers = new SyBookkeeper *[ krylovSpaceDimension ];

   dcomplex * krylovHamiltonian = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   dcomplex * overlaps          = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   // Step 1
   krylovBasisVectors[ 0 ]                           = mpsIn;
   krylovBasisSyBookkeepers[ 0 ]                     = bkIn;
   krylovHamiltonian[ 0 + 0 * krylovSpaceDimension ] = op->Overlap( krylovBasisVectors[ 0 ], krylovBasisSyBookkeepers[ 0 ], krylovBasisVectors[ 0 ], krylovBasisSyBookkeepers[ 0 ] );
   overlaps[ 0 + 0 * krylovSpaceDimension ]          = overlap( krylovBasisVectors[ 0 ], krylovBasisVectors[ 0 ] );

   for ( int kry = 1; kry < krylovSpaceDimension; kry++ ) {
      SyBookkeeper * bkTemp = new SyBookkeeper( *krylovBasisSyBookkeepers[ kry - 1 ] );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }

      dcomplex * coefs            = new dcomplex[ kry ];
      CTensorT *** states         = new CTensorT **[ kry ];
      SyBookkeeper ** bookkeepers = new SyBookkeeper *[ kry ];

      for ( int i = 0; i < kry; i++ ) {
         coefs[ i ]       = -krylovHamiltonian[ i + ( kry - 1 ) * krylovSpaceDimension ] / overlaps[ i + i * krylovSpaceDimension ];
         states[ i ]      = krylovBasisVectors[ i ];
         bookkeepers[ i ] = krylovBasisSyBookkeepers[ i ];
      }

      op->DSApplyAndAdd( krylovBasisVectors[ kry - 1 ], krylovBasisSyBookkeepers[ kry - 1 ],
                         kry, coefs, states, bookkeepers,
                         mpsTemp, bkTemp,
                         scheme );

      delete[] coefs;
      delete[] states;
      delete[] bookkeepers;

      krylovBasisVectors[ kry ]       = mpsTemp;
      krylovBasisSyBookkeepers[ kry ] = bkTemp;

      for ( int i = 0; i <= kry; i++ ) {
         overlaps[ i + kry * krylovSpaceDimension ]          = overlap( krylovBasisVectors[ i ], krylovBasisVectors[ kry ] );
         overlaps[ kry + i * krylovSpaceDimension ]          = std::conj( overlaps[ i + kry * krylovSpaceDimension ] );
         krylovHamiltonian[ i + kry * krylovSpaceDimension ] = op->Overlap( krylovBasisVectors[ i ], krylovBasisSyBookkeepers[ i ], krylovBasisVectors[ kry ], krylovBasisSyBookkeepers[ kry ] );
         krylovHamiltonian[ kry + i * krylovSpaceDimension ] = std::conj( krylovHamiltonian[ i + kry * krylovSpaceDimension ] );
      }
   }

   if ( std::abs( overlaps[ krylovSpaceDimension ] ) > 1e-6 ) {
      std::cout << "CHEMPS2::TIME WARNING: "
                << " Krylov vectors not completely orthonormal. |< kry_0 | kry_last>| is " << overlaps[ krylovSpaceDimension ] << std::endl;
   }

   int one                 = 1;
   int sqr                 = krylovSpaceDimension * krylovSpaceDimension;
   dcomplex * overlaps_inv = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   zcopy_( &sqr, overlaps, &one, overlaps_inv, &one );

   int info_lu;
   int * piv = new int[ krylovSpaceDimension ];
   zgetrf_( &krylovSpaceDimension, &krylovSpaceDimension, overlaps_inv, &krylovSpaceDimension, piv, &info_lu );

   dcomplex * work = new dcomplex[ krylovSpaceDimension ];
   int info_inve;

   zgetri_( &krylovSpaceDimension, overlaps_inv, &krylovSpaceDimension, piv, work, &krylovSpaceDimension, &info_inve );

   for ( int i = 0; i < krylovSpaceDimension; i++ ) {
      for ( int j = i + 1; j < krylovSpaceDimension; j++ ) {
         overlaps_inv[ i + krylovSpaceDimension * j ] = overlaps_inv[ j + krylovSpaceDimension * i ];
      }
   }

   dcomplex * toExp = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   char notrans     = 'N';
   dcomplex zeroC   = 0.0;
   zgemm_( &notrans, &notrans, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension,
           &step, overlaps_inv, &krylovSpaceDimension, krylovHamiltonian, &krylovSpaceDimension, &zeroC, toExp, &krylovSpaceDimension );

   int deg        = 6;
   double bla     = 1.0;
   int lwsp       = 4 * krylovSpaceDimension * krylovSpaceDimension + deg + 1;
   dcomplex * wsp = new dcomplex[ lwsp ];
   int * ipiv     = new int[ krylovSpaceDimension ];
   int iexph      = 0;
   int ns         = 0;
   int info;

   zgpadm_( &deg, &krylovSpaceDimension, &bla, toExp, &krylovSpaceDimension,
            wsp, &lwsp, ipiv, &iexph, &ns, &info );

   dcomplex * exph = &wsp[ iexph - 1 ];

   dcomplex * result = new dcomplex[ krylovSpaceDimension ];
   for ( int i = 0; i < krylovSpaceDimension; i++ ) {
      result[ i ] = exph[ i + krylovSpaceDimension * 0 ];
   }

   op->DSSum( krylovSpaceDimension, result, &krylovBasisVectors[ 0 ], &krylovBasisSyBookkeepers[ 0 ],
              mpsOut, bkOut,
              scheme );

   delete[] result;
   delete[] wsp;
   delete[] ipiv;
   delete[] overlaps;
   delete[] overlaps_inv;
   delete[] krylovHamiltonian;
   delete[] toExp;
   delete[] piv;
   delete[] work;

   for ( int cnt = 1; cnt < krylovSpaceDimension; cnt++ ) {
      for ( int site = 0; site < L; site++ ) {
         delete krylovBasisVectors[ cnt ][ site ];
      }
      delete[] krylovBasisVectors[ cnt ];
      delete krylovBasisSyBookkeepers[ cnt ];
   }
   delete[] krylovBasisVectors;
   delete[] krylovBasisSyBookkeepers;

   delete op;
}

// void CheMPS2::TimeEvolution::doStep_krylov( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

// dcomplex step = doImaginary ? -scheme->get_time_step( currentInstruction ) : dcomplex( 0.0, -1.0 * -scheme->get_time_step( currentInstruction ) );

// HamiltonianOperator * op = new HamiltonianOperator( prob );

// std::vector< CTensorT ** > krylovBasisVectors;
// std::vector< SyBookkeeper * > krylovBasisSyBookkeepers;
// std::vector< dcomplex > krylovHamiltonianDiagonal;
// std::vector< dcomplex > krylovHamiltonianOffDiagonal;

// // Step 1
// krylovBasisVectors.push_back( mpsIn );
// krylovBasisSyBookkeepers.push_back( bkIn );
// krylovHamiltonianDiagonal.push_back( op->ExpectationValue( krylovBasisVectors.back(), krylovBasisSyBookkeepers.back() ) );

// while ( true ) {

//    SyBookkeeper * bkTemp = new SyBookkeeper( *bkIn );
//    CTensorT ** mpsTemp   = new CTensorT *[ L ];
//    for ( int index = 0; index < L; index++ ) {
//       mpsTemp[ index ] = new CTensorT( mpsIn[ index ] );
//       mpsTemp[ index ]->random();
//    }

//    if ( krylovBasisVectors.size() == 1 ) {
//       dcomplex coef[]              = {-krylovHamiltonianDiagonal.back()};
//       CTensorT ** states[]         = {krylovBasisVectors.back()};
//       SyBookkeeper * bookkeepers[] = {krylovBasisVectorBookkeepers.back()};

//       op->DSApplyAndAdd( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(),
//                          1, coef, states, bookkeepers,
//                          mpsTemp, bkTemp,
//                          scheme->get_max_sweeps( currentInstruction ) );
//    } else {
//       // dcomplex coef[]              = {-krylovHamiltonianOffDiagonal.back(), -krylovHamiltonianDiagonal.back()};
//       // CTensorT ** states[]         = {krylovBasisVectors.end()[ -2 ], krylovBasisVectors.back()};
//       // SyBookkeeper * bookkeepers[] = {krylovBasisVectorBookkeepers.end()[ -2 ], krylovBasisVectorBookkeepers.back()};

//       // op->DSApplyAndAdd( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(),
//       //                    2, coef, states, bookkeepers,
//       //                    mpsTemp, bkTemp,
//       //                    scheme->get_max_sweeps( currentInstruction ) );

//       std::vector< dcomplex > coef;
//       std::vector< CTensorT ** > states;
//       std::vector< SyBookkeeper * > bookkeepers;

//       coef.push_back( -krylovHamiltonianOffDiagonal.back() );
//       states.push_back( krylovBasisVectors.end()[ -2 ] );
//       bookkeepers.push_back( krylovBasisVectorBookkeepers.end()[ -2 ] );

//       for ( int i = 0; i < krylovBasisVectors.size(); i++ ) {
//          coef.push_back( -op->Overlap( krylovBasisVectors[ i ], krylovBasisVectorBookkeepers[ i ], krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );
//          states.push_back( krylovBasisVectors[ i ] );
//          bookkeepers.push_back( krylovBasisVectorBookkeepers[ i ] );
//       }

//       coef.erase( coef.begin() + 1 );
//       states.erase( states.begin() + 1 );
//       bookkeepers.erase( bookkeepers.begin() + 1 );

//       // coef.push_back( -op->Overlap( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(), krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );
//       //             states.push_back( krylovBasisVectors.back() );
//       //             bookkeepers.push_back( krylovBasisVectorBookkeepers.back() );

//       op->DSApplyAndAdd( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(),
//                          states.size(), &coef[ 0 ], &states[ 0 ], &bookkeepers[ 0 ],
//                          mpsTemp, bkTemp,
//                          scheme->get_max_sweeps( currentInstruction ) );
//    }

//    dcomplex beta = norm( mpsTemp );
//    std::cout << "beta: " << beta << std::endl;
//    if ( abs( beta ) < 1e-10 || krylovBasisVectors.size() > 20 ) {
//       break;
//    }
//    krylovHamiltonianOffDiagonal.push_back( beta );
//    mpsTemp[ 0 ]->number_operator( 0.0, 1.0 / beta );
//    krylovBasisVectors.push_back( mpsTemp );
//    krylovBasisVectorBookkeepers.push_back( bkTemp );
//    std::cout << std::real( op->Overlap( krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ], krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ] ) ) << std::endl;
//    std::cout << std::real( op->Overlap( krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ], krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ] ) ) << std::endl;
//    krylovHamiltonianDiagonal.push_back( op->ExpectationValue( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );
//    std::cout << "alpha: " << krylovHamiltonianDiagonal.back() << std::endl;
// }

// int krylovSpaceDimension = krylovBasisVectors.size();

// // std::cout << std::endl;
// // for ( int i = 0; i < krylovSpaceDimension; i++ ) {
// //    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
// //       std::cout << std::real( overlap( krylovBasisVectors[ i ], krylovBasisVectors[ j ] ) ) << " ";
// //    }
// //    std::cout << std::endl;
// // }
// // std::cout << std::endl;

// // std::cout << std::endl;
// // for ( int i = 0; i < krylovSpaceDimension; i++ ) {
// //    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
// //       std::cout << std::real( op->Overlap( krylovBasisVectors[ i ], krylovBasisVectorBookkeepers[ i ], krylovBasisVectors[ j ], krylovBasisVectorBookkeepers[ j ] ) ) << " ";
// //    }
// //    std::cout << std::endl;
// // }
// // std::cout << std::endl;

// dcomplex * krylovHamiltonian = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
//       if ( i == j ) {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = krylovHamiltonianDiagonal[ i ];
//       } else if ( i == j - 1 ) {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = krylovHamiltonianOffDiagonal[ i ];
//       } else if ( i == j + 1 ) {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = krylovHamiltonianOffDiagonal[ j ];
//       } else {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = 0.0;
//       }
//    }
// }

// std::cout << std::endl;
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
//       std::cout << std::real( krylovHamiltonian[ i + krylovSpaceDimension * j ] ) << " ";
//    }
//    std::cout << std::endl;
// }
// std::cout << std::endl;

// char jobz       = 'V';
// char uplo       = 'U';
// double * evals  = new double[ krylovSpaceDimension ];
// int lwork       = 2 * krylovSpaceDimension - 1;
// dcomplex * work = new dcomplex[ lwork ];
// double * rwork  = new double[ 3 * krylovSpaceDimension - 2 ];
// int info;

// zheev_( &jobz, &uplo, &krylovSpaceDimension, krylovHamiltonian, &krylovSpaceDimension, evals, work, &lwork, rwork, &info );

// dcomplex * firstVec = new dcomplex[ krylovSpaceDimension ];
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    firstVec[ i ] = exp( evals[ i ] * step ) * krylovHamiltonian[ krylovSpaceDimension * i ];
// }

// char notrans      = 'N';
// int onedim        = 1;
// dcomplex one      = 1.0;
// dcomplex zero     = 0.0;
// dcomplex * result = new dcomplex[ krylovSpaceDimension ];
// zgemm_( &notrans, &notrans, &krylovSpaceDimension, &onedim, &krylovSpaceDimension,
//         &one, krylovHamiltonian, &krylovSpaceDimension,
//         firstVec, &krylovSpaceDimension, &zero, result, &krylovSpaceDimension );

// op->DSSum( krylovSpaceDimension, result, &krylovBasisVectors[ 0 ], &krylovBasisVectorBookkeepers[ 0 ], mpsOut, bkOut, scheme->get_max_sweeps( currentInstruction ) );

// delete op;

// dcomplex step = doImaginary ? -scheme->get_time_step( currentInstruction ) : dcomplex( 0.0, -1.0 * -scheme->get_time_step( currentInstruction ) );

// HamiltonianOperator * op = new HamiltonianOperator( prob );

// std::vector< CTensorT ** > krylovBasisVectors;
// std::vector< SyBookkeeper * > krylovBasisVectorBookkeepers;
// std::vector< dcomplex > krylovHamiltonianDiagonal;
// std::vector< dcomplex > krylovHamiltonianOffDiagonal;

// // Step 1
// krylovBasisVectors.push_back( mpsIn );
// krylovBasisVectorBookkeepers.push_back( bkIn );
// krylovHamiltonianDiagonal.push_back( op->ExpectationValue( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );

// while ( true ) {

//    SyBookkeeper * bkTemp = new SyBookkeeper( *bkIn );
//    CTensorT ** mpsTemp   = new CTensorT *[ L ];
//    for ( int index = 0; index < L; index++ ) {
//       mpsTemp[ index ] = new CTensorT( mpsIn[ index ] );
//       mpsTemp[ index ]->random();
//    }

//    if ( krylovBasisVectors.size() == 1 ) {
//       dcomplex coef[]              = {-krylovHamiltonianDiagonal.back()};
//       CTensorT ** states[]         = {krylovBasisVectors.back()};
//       SyBookkeeper * bookkeepers[] = {krylovBasisVectorBookkeepers.back()};

//       op->DSApplyAndAdd( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(),
//                          1, coef, states, bookkeepers,
//                          mpsTemp, bkTemp,
//                          scheme->get_max_sweeps( currentInstruction ) );
//    } else {
//       // dcomplex coef[]              = {-krylovHamiltonianOffDiagonal.back(), -krylovHamiltonianDiagonal.back()};
//       // CTensorT ** states[]         = {krylovBasisVectors.end()[ -2 ], krylovBasisVectors.back()};
//       // SyBookkeeper * bookkeepers[] = {krylovBasisVectorBookkeepers.end()[ -2 ], krylovBasisVectorBookkeepers.back()};

//       // op->DSApplyAndAdd( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(),
//       //                    2, coef, states, bookkeepers,
//       //                    mpsTemp, bkTemp,
//       //                    scheme->get_max_sweeps( currentInstruction ) );

//       std::vector< dcomplex > coef;
//       std::vector< CTensorT ** > states;
//       std::vector< SyBookkeeper * > bookkeepers;

//       coef.push_back( -krylovHamiltonianOffDiagonal.back() );
//       states.push_back( krylovBasisVectors.end()[ -2 ] );
//       bookkeepers.push_back( krylovBasisVectorBookkeepers.end()[ -2 ] );

//       for ( int i = 0; i < krylovBasisVectors.size(); i++ ) {
//          coef.push_back( -op->Overlap( krylovBasisVectors[ i ], krylovBasisVectorBookkeepers[ i ], krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );
//          states.push_back( krylovBasisVectors[ i ] );
//          bookkeepers.push_back( krylovBasisVectorBookkeepers[ i ] );
//       }

//       coef.erase( coef.begin() + 1 );
//       states.erase( states.begin() + 1 );
//       bookkeepers.erase( bookkeepers.begin() + 1 );

//       // coef.push_back( -op->Overlap( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(), krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );
//       //             states.push_back( krylovBasisVectors.back() );
//       //             bookkeepers.push_back( krylovBasisVectorBookkeepers.back() );

//       op->DSApplyAndAdd( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back(),
//                          states.size(), &coef[ 0 ], &states[ 0 ], &bookkeepers[ 0 ],
//                          mpsTemp, bkTemp,
//                          scheme->get_max_sweeps( currentInstruction ) );
//    }

//    dcomplex beta = norm( mpsTemp );
//    std::cout << "beta: " << beta << std::endl;
//    if ( abs( beta ) < 1e-10 || krylovBasisVectors.size() > 20 ) {
//       break;
//    }
//    krylovHamiltonianOffDiagonal.push_back( beta );
//    mpsTemp[ 0 ]->number_operator( 0.0, 1.0 / beta );
//    krylovBasisVectors.push_back( mpsTemp );
//    krylovBasisVectorBookkeepers.push_back( bkTemp );
//    std::cout << std::real( op->Overlap( krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ], krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ] ) ) << std::endl;
//    std::cout << std::real( op->Overlap( krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ], krylovBasisVectors[ 1 ], krylovBasisVectorBookkeepers[ 1 ] ) ) << std::endl;
//    krylovHamiltonianDiagonal.push_back( op->ExpectationValue( krylovBasisVectors.back(), krylovBasisVectorBookkeepers.back() ) );
//    std::cout << "alpha: " << krylovHamiltonianDiagonal.back() << std::endl;
// }

// int krylovSpaceDimension = krylovBasisVectors.size();

// std::cout << std::endl;
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
//       std::cout << std::real( overlap( krylovBasisVectors[ i ], krylovBasisVectors[ j ] ) ) << " ";
//    }
//    std::cout << std::endl;
// }
// std::cout << std::endl;

// std::cout << std::endl;
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
//       std::cout << std::real( op->Overlap( krylovBasisVectors[ i ], krylovBasisVectorBookkeepers[ i ], krylovBasisVectors[ j ], krylovBasisVectorBookkeepers[ j ] ) ) << " ";
//    }
//    std::cout << std::endl;
// }
// std::cout << std::endl;

// dcomplex * krylovHamiltonian = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
//       if ( i == j ) {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = krylovHamiltonianDiagonal[ i ];
//       } else if ( i == j - 1 ) {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = krylovHamiltonianOffDiagonal[ i ];
//       } else if ( i == j + 1 ) {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = krylovHamiltonianOffDiagonal[ j ];
//       } else {
//          krylovHamiltonian[ i + krylovSpaceDimension * j ] = 0.0;
//       }
//    }
// }

// std::cout << std::endl;
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    for ( int j = 0; j < krylovSpaceDimension; j++ ) {
//       std::cout << std::real( krylovHamiltonian[ i + krylovSpaceDimension * j ] ) << " ";
//    }
//    std::cout << std::endl;
// }
// std::cout << std::endl;

// char jobz       = 'V';
// char uplo       = 'U';
// double * evals  = new double[ krylovSpaceDimension ];
// int lwork       = 2 * krylovSpaceDimension - 1;
// dcomplex * work = new dcomplex[ lwork ];
// double * rwork  = new double[ 3 * krylovSpaceDimension - 2 ];
// int info;

// zheev_( &jobz, &uplo, &krylovSpaceDimension, krylovHamiltonian, &krylovSpaceDimension, evals, work, &lwork, rwork, &info );

// dcomplex * firstVec = new dcomplex[ krylovSpaceDimension ];
// for ( int i = 0; i < krylovSpaceDimension; i++ ) {
//    firstVec[ i ] = exp( evals[ i ] * step ) * krylovHamiltonian[ krylovSpaceDimension * i ];
// }

// char notrans      = 'N';
// int onedim        = 1;
// dcomplex one      = 1.0;
// dcomplex zero     = 0.0;
// dcomplex * result = new dcomplex[ krylovSpaceDimension ];
// zgemm_( &notrans, &notrans, &krylovSpaceDimension, &onedim, &krylovSpaceDimension,
//         &one, krylovHamiltonian, &krylovSpaceDimension,
//         firstVec, &krylovSpaceDimension, &zero, result, &krylovSpaceDimension );

// op->DSSum( krylovSpaceDimension, result, &krylovBasisVectors[ 0 ], &krylovBasisVectorBookkeepers[ 0 ], mpsOut, bkOut, scheme->get_max_sweeps( currentInstruction ) );

// delete op;
// }

void CheMPS2::TimeEvolution::Propagate( SyBookkeeper * initBK, CTensorT ** initMPS,
                                        const double time_step, const double time_final,
                                        const int kry_size, const bool doImaginary, const bool doDumpFCI ) {
   std::cout << "\n";
   std::cout << "   Starting to propagate MPS\n";
   std::cout << "\n";

   hid_t outputID    = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, "/Output", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   hsize_t dimarray1 = 1;

   HamiltonianOperator * hamOp = new HamiltonianOperator( prob );

   SyBookkeeper * MPSBK = new SyBookkeeper( *initBK );
   CTensorT ** MPS      = new CTensorT *[ L ];
   for ( int index = 0; index < L; index++ ) {
      MPS[ index ] = new CTensorT( initMPS[ index ] );
   }

   double firstEnergy = 0;

   for ( double t = 0.0; t < time_final; t += time_step ) {
      char dataPointname[ 1024 ];
      sprintf( dataPointname, "/Output/DataPoint%.5f", t );
      hid_t dataPointID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataPointname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;

      std::cout << hashline;
      std::cout << "\n";
      std::cout << "   t = " << t << "\n";
      HDF5_MAKE_DATASET( dataPointID, "t", 1, &dimarray1, H5T_NATIVE_DOUBLE, &t );

      double Tmax = time_final;
      std::cout << "   Tmax = " << Tmax << "\n";
      HDF5_MAKE_DATASET( dataPointID, "Tmax", 1, &dimarray1, H5T_NATIVE_DOUBLE, &Tmax );

      double dt = time_step;
      std::cout << "   dt = " << dt << "\n";
      HDF5_MAKE_DATASET( dataPointID, "dt", 1, &dimarray1, H5T_NATIVE_DOUBLE, &dt );

      std::cout << "   KryS = " << kry_size << "\n";
      HDF5_MAKE_DATASET( dataPointID, "KryS", 1, &dimarray1, H5T_STD_I32LE, &kry_size );

      std::cout << "\n";

      std::cout << "   matrix product state dimensions:\n";
      std::cout << "   ";
      for ( int i = 0; i < L + 1; i++ ) {
         std::cout << std::setw( 5 ) << i;
      }
      std::cout << "\n";
      std::cout << "   ";
      int * actdims = new int[ L + 1 ];
      for ( int i = 0; i < L + 1; i++ ) {
         actdims[ i ] = MPSBK->gTotDimAtBound( i );
         std::cout << std::setw( 5 ) << actdims[ i ];
      }
      std::cout << "\n";
      hsize_t Lposize = L + 1;
      HDF5_MAKE_DATASET( dataPointID, "MPSDims", 1, &Lposize, H5T_STD_I32LE, actdims );
      delete[] actdims;

      std::cout << "\n";

      std::cout << "   MaxM = ";
      hsize_t numInst = scheme->get_number();
      int * MaxMs     = new int[ numInst ];
      for ( int inst = 0; inst < numInst; inst++ ) {
         MaxMs[ inst ] = scheme->get_D( inst );
         std::cout << MaxMs[ inst ] << " ";
      }
      std::cout << "\n";
      HDF5_MAKE_DATASET( dataPointID, "MaxMs", 1, &numInst, H5T_STD_I32LE, MaxMs );
      delete[] MaxMs;

      std::cout << "   CutO = ";
      int * CutOs = new int[ numInst ];
      for ( int inst = 0; inst < numInst; inst++ ) {
         CutOs[ inst ] = scheme->get_cut_off( inst );
         std::cout << CutOs[ inst ] << " ";
      }
      std::cout << "\n";
      HDF5_MAKE_DATASET( dataPointID, "CutOs", 1, &numInst, H5T_STD_I32LE, CutOs );
      delete[] CutOs;

      std::cout << "   NSwes = ";
      int * NSwes = new int[ numInst ];
      for ( int inst = 0; inst < numInst; inst++ ) {
         NSwes[ inst ] = scheme->get_max_sweeps( inst );
         std::cout << NSwes[ inst ] << " ";
      }
      std::cout << "\n";
      HDF5_MAKE_DATASET( dataPointID, "NSwes", 1, &numInst, H5T_STD_I32LE, NSwes );
      delete[] NSwes;

      std::cout << "\n";

      double normOfMPS = norm( MPS );
      std::cout << "   Norm      = " << normOfMPS << "\n";
      HDF5_MAKE_DATASET( dataPointID, "Norm", 1, &dimarray1, H5T_NATIVE_DOUBLE, &normOfMPS );

      double energy = std::real( hamOp->ExpectationValue( MPS, MPSBK ) );
      std::cout << "   Energy    = " << energy << "\n";
      HDF5_MAKE_DATASET( dataPointID, "Energy", 1, &dimarray1, H5T_NATIVE_DOUBLE, &energy );

      if ( t == 0.0 ) {
         firstEnergy = energy;
      }

      COneDM * theodm = new COneDM( MPS, MPSBK );
      hsize_t Lsq[ 2 ];
      Lsq[ 0 ]        = L;
      Lsq[ 1 ]        = L;
      double * oedmre = new double[ L * L ];
      double * oedmim = new double[ L * L ];
      theodm->gOEDMRe( oedmre );
      theodm->gOEDMIm( oedmim );

      std::cout << "\n";
      std::cout << "  occupation numbers of molecular orbitals:\n";
      std::cout << "   ";
      for ( int i = 0; i < L; i++ ) {
         std::cout << std::setw( 20 ) << std::fixed << std::setprecision( 15 ) << oedmre[ i + L * i ];
      }
      std::cout << "\n";

      HDF5_MAKE_DATASET( dataPointID, "OEDM_REAL", 2, Lsq, H5T_NATIVE_DOUBLE, oedmre );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_IMAG", 2, Lsq, H5T_NATIVE_DOUBLE, oedmim );
      delete[] oedmre;
      delete[] oedmim;
      delete theodm;
      if ( doDumpFCI ) {
         std::vector< std::vector< int > > alphasOut;
         std::vector< std::vector< int > > betasOut;
         std::vector< double > coefsRealOut;
         std::vector< double > coefsImagOut;
         hsize_t Lsize = L;
         getFCITensor( prob, MPS, alphasOut, betasOut, coefsRealOut, coefsImagOut );

         char dataFCIName[ 1024 ];
         sprintf( dataFCIName, "%s/FCICOEF", dataPointname );
         hid_t FCIID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataFCIName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;

         for ( int l = 0; l < alphasOut.size(); l++ ) {
            char dataFCINameN[ 1024 ];
            sprintf( dataFCINameN, "%s/FCICOEF/%i", dataPointname, l );
            hid_t FCIID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataFCINameN, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
            HDF5_MAKE_DATASET( FCIID, "FCI_ALPHAS", 1, &Lsize, H5T_NATIVE_INT, &alphasOut[ l ][ 0 ] );
            HDF5_MAKE_DATASET( FCIID, "FCI_BETAS", 1, &Lsize, H5T_NATIVE_INT, &betasOut[ l ][ 0 ] );
            HDF5_MAKE_DATASET( FCIID, "FCI_REAL", 1, &dimarray1, H5T_NATIVE_DOUBLE, &coefsRealOut[ l ] );
            HDF5_MAKE_DATASET( FCIID, "FCI_IMAG", 1, &dimarray1, H5T_NATIVE_DOUBLE, &coefsImagOut[ l ] );
         }
      }
      std::cout << "\n";

      if ( t + time_step < time_final ) {
         SyBookkeeper * MPSBKDT = new SyBookkeeper( *MPSBK );
         CTensorT ** MPSDT      = new CTensorT *[ L ];
         for ( int index = 0; index < L; index++ ) {
            MPSDT[ index ] = new CTensorT( index, MPSBKDT );
            MPSDT[ index ]->random();
         }

         doStep_arnoldi( time_step, time_final, kry_size, doImaginary, MPS, MPSBK, MPSDT, MPSBKDT );

         for ( int site = 0; site < L; site++ ) {
            delete MPS[ site ];
         }
         delete[] MPS;
         delete MPSBK;

         MPS   = MPSDT;
         MPSBK = MPSBKDT;

         if ( doImaginary ) {
            double normDT = norm( MPS );
            MPS[ 0 ]->number_operator( 0.0, 1.0 / normDT );
         }
      }

      std::cout << hashline;
   }

   for ( int site = 0; site < L; site++ ) {
      delete MPS[ site ];
   }
   delete[] MPS;
   delete MPSBK;
   delete hamOp;
}