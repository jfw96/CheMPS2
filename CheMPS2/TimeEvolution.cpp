
#include "TimeEvolution.h"
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include "COneDM.h"
#include "CTwoDMBuilder.h"
#include "HamiltonianOperator.h"
#include "Lapack.h"
#include "TwoDMBuilder.h"

CheMPS2::TimeEvolution::TimeEvolution( Problem * probIn, ConvergenceScheme * schemeIn, hid_t HDF5FILEIDIN )
    : prob( probIn ), scheme( schemeIn ), HDF5FILEID( HDF5FILEIDIN ), L( probIn->gL() ) {

   start        = time( NULL );
   tm * tmstart = localtime( &start );
   std::ostringstream text;

   std::cout << hashline;
   std::cout << "### Starting to run a time evolution calculation on " << tmstart->tm_year + 1900 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mon + 1 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mday << " ";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_hour << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_min << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_sec << "\n";
   std::cout << hashline;

   assert( probIn->checkConsistency() );
   prob->construct_mxelem();

   const int Sy        = prob->gSy();
   const int N         = prob->gN();
   int * irreps        = new int[ L ];
   const int TwoS      = prob->gTwoS();
   const double Econst = prob->gEconst();
   const int I         = prob->gIrrep();
   int * ham2dmrg       = new int[ L + 1 ];
   int * fcidims        = new int[ L + 1 ];
   CheMPS2::SyBookkeeper * tempBK = new CheMPS2::SyBookkeeper( prob, 0 );

   std::cout << "\n";
   std::cout << "   System Properties\n";
   std::cout << "\n";
   std::cout << "\n";
   std::cout << "   L = " << L << "\n";
   std::cout << "   Sy = " << Sy << "\n";
   std::cout << "   Irreps =";
   for ( int i = 0; i < L; i++ ) { irreps[ i ] = prob->gIrrep( i ); std::cout << std::setfill( ' ' ) << std::setw( 10 ) << irreps[ i ]; }
   std::cout << "\n";
   std::cout << "\n";
   std::cout << "   N = " << N << "\n";
   std::cout << "   TwoS = " << TwoS << "\n";
   std::cout << "   I = " << I << "\n";
   std::cout << "\n";
   std::cout << "   Orbital ordering:\n";
   std::cout << "      ";
   for ( int i = 0; i < L; i++ ) { ham2dmrg[ i ] = prob->gf2( i ); std::cout << std::setfill( ' ' ) << std::setw( 2 ) << i << " -> " << std::setfill( ' ' ) << std::setw( 1 ) << ham2dmrg[ i ] << "  "; }
   std::cout << "\n";
   std::cout << "\n";
   std::cout << "   full ci matrix product state dimensions:\n";
   std::cout << "   ";
   for ( int i = 0; i < L + 1; i++ ) { std::cout << std::setfill( ' ' ) << std::setw( 10 ) << i; }
   std::cout << "\n";
   std::cout << "   ";
   for ( int i = 0; i < L + 1; i++ ) { fcidims[ i ] = tempBK->gFCIDimAtBound( i ); std::cout << std::setfill( ' ' ) << std::setw( 10 ) << fcidims[ i ]; }
   std::cout << "\n";
   std::cout << "\n";
   std::cout << hashline;

   const hid_t inputGroupID             = ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) ? H5Gcreate( HDF5FILEID, "/Input", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   const hid_t systemPropertiesID       = ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) ? H5Gcreate( HDF5FILEID, "/Input/SystemProperties", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   const hid_t waveFunctionPropertiesID = ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) ? H5Gcreate( HDF5FILEID, "/Input/WaveFunctionProperties", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   const hsize_t dimarray1              = 1;
   const hsize_t Lsize = L;
   const hsize_t Lposize = L + 1;
   HDF5_MAKE_DATASET( systemPropertiesID,       "L",        1, &dimarray1, H5T_STD_I32LE, &L          );
   HDF5_MAKE_DATASET( systemPropertiesID,       "Sy",       1, &dimarray1, H5T_STD_I32LE, &Sy         );
   HDF5_MAKE_DATASET( systemPropertiesID,       "Irrep",    1, &Lsize,     H5T_STD_I32LE, irreps      );
   HDF5_MAKE_DATASET( systemPropertiesID,       "Econst",   1, &dimarray1, H5T_NATIVE_DOUBLE, &Econst );
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "N",        1, &dimarray1, H5T_STD_I32LE, &N          );
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "TwoS",     1, &dimarray1, H5T_STD_I32LE, &TwoS       );
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "I",        1, &dimarray1, H5T_STD_I32LE, &I          );
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "Ham2DMRG", 1, &Lsize,     H5T_STD_I32LE, ham2dmrg    );
   HDF5_MAKE_DATASET( waveFunctionPropertiesID, "FCIDims",  1, &Lposize,   H5T_STD_I32LE, fcidims     );
   delete[] irreps;
   delete[] ham2dmrg;
   delete[] fcidims;
   delete tempBK;

}

CheMPS2::TimeEvolution::~TimeEvolution() {

   std::time_t now = time( NULL );
   tm * tmstart    = localtime( &now );
   std::ostringstream text;

   std::cout << hashline;
   std::cout << "### Finished to run a time evolution calculation on " << tmstart->tm_year + 1900 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mon + 1 << "-";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_mday << " ";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_hour << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_min << ":";
   std::cout << std::setfill( '0' ) << std::setw( 2 ) << tmstart->tm_sec << "\n";

   std::cout << "### Calculation took " << ( time( NULL ) - start ) / 60.0 << " minutes\n" << hashline;
   std::cout << hashline;
}

void CheMPS2::TimeEvolution::HDF5_MAKE_DATASET( hid_t setID, const char * name, int rank, const hsize_t * dims, hid_t typeID, const void * data ) {
   if ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) {
      H5LTmake_dataset( setID, name, rank, dims, typeID, data );
   }
}

void CheMPS2::TimeEvolution::doStep_arnoldi( const double time_step, const double time_final, const int kry_size, dcomplex offset, const bool doImaginary, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

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
   
   std::cout << "   Krylov space vectors:\n";
   std::cout << "      i = " << 0 << " ";
   std::cout << "MPS dimensions:";
   for (int i = 0; i < prob->gL(); i++){
      std::cout << " " << krylovBasisSyBookkeepers[ 0 ]->gTotDimAtBound( i );
   }
   std::cout << std::endl;

   for ( int kry = 1; kry < krylovSpaceDimension; kry++ ) {

      struct timeval start, end;
      gettimeofday( &start, NULL );

      SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D ( 0 ) );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }
      
      double normTemp = norm( mpsTemp );
      for( int idx = 0; idx < prob->gL(); idx++ ){
         mpsTemp[ idx ]->number_operator( 0.0,  std::pow( normTemp, - 1.0 / prob->gL() ) );
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
                         0, coefs, states, bookkeepers,
                         mpsTemp, bkTemp,
                         scheme );

      normTemp = norm( mpsTemp );
      for( int idx = 0; idx < prob->gL(); idx++ ){
         mpsTemp[ idx ]->number_operator( 0.0,  std::pow( normTemp, - 1.0 / prob->gL() ) );
      }

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

      gettimeofday( &end, NULL );
      const double elapsed = ( end.tv_sec - start.tv_sec ) + 1e-6 * ( end.tv_usec - start.tv_usec );

      std::cout << "      i = " << kry << " ";
      std::cout << "MPS dimensions:";
      for (int i = 0; i <= prob->gL(); i++){
         std::cout << " " << krylovBasisSyBookkeepers[ kry ]->gTotDimAtBound( i );
      }
      std::cout << " time elapsed: " << elapsed << " seconds\n";
      
   }
   std::cout << "\n";

   // int inc = 1;
   // int sze = krylovSpaceDimension * krylovSpaceDimension;

   // zaxpy_( &sze, &offset, overlaps, &inc, krylovHamiltonian, &inc  );

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Building S and H
   ////
   ////////////////////////////////////////////////////////////////////////////////////////
   
   for ( int irow = 0; irow < krylovSpaceDimension; irow++ ){
      for ( int icol = 0; icol < krylovSpaceDimension; icol++ ){
         std::cout << std::real( krylovHamiltonian[ irow +  icol * krylovSpaceDimension ] ) << " ";
      }
      std::cout << std::endl;
   }

   for ( int irow = 0; irow < krylovSpaceDimension; irow++ ){
      for ( int icol = 0; icol < krylovSpaceDimension; icol++ ){
         std::cout << std::real( overlaps[ irow +  icol * krylovSpaceDimension ] ) << " ";
      }
      std::cout << std::endl;
   }

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Calculating the inverse
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

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

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Multiplying N^-1 and H
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   dcomplex * toExp = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   char notrans     = 'N';
   dcomplex zeroC   = 0.0;
   dcomplex oneC    = 1.0;
   zgemm_( &notrans, &notrans, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension,
           &step, overlaps_inv, &krylovSpaceDimension, krylovHamiltonian, &krylovSpaceDimension, &zeroC, toExp, &krylovSpaceDimension );

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Calculate the matrix exponential
   ////
   ////////////////////////////////////////////////////////////////////////////////////////
   int deg        = 20;
   double bla     = 1.0;
   int lwsp       = 4 * krylovSpaceDimension * krylovSpaceDimension + deg + 1;
   dcomplex * wsp = new dcomplex[ lwsp ];
   int * ipiv     = new int[ krylovSpaceDimension ];
   int iexph      = 0;
   int ns         = 0;
   int info;

   zgpadm_( &deg, &krylovSpaceDimension, &bla, toExp, &krylovSpaceDimension,
            wsp, &lwsp, ipiv, &iexph, &ns, &info );
   assert( info == 0 );

   dcomplex * theExp = &wsp[ iexph - 1 ];

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Test new coefficients and sum MPS
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   dcomplex * result = new dcomplex[ krylovSpaceDimension ];
   for ( int i = 0; i < krylovSpaceDimension; i++ ) {
      result[ i ] = theExp[ i + krylovSpaceDimension * 0 ];
   }

   dcomplex tobeone = 0.0;
   for( int i = 0; i < krylovSpaceDimension; i++ ){
      for( int j = 0; j < krylovSpaceDimension; j++ ){
         tobeone += std::conj( result[ i ] ) * result[ j ] * overlaps[ i + krylovSpaceDimension * j ];
      }   
   }

   if ( ( std::abs( tobeone - overlaps[ 0 ] ) > 1e-9 ) ) {
      std::cout << "CHEMPS2::TIME WARNING: "
                << " Numerical Problem with non-orthonormal Krylov basis. Is your state close to an energy eigenstate?\n";
      std::cout << "CHEMPS2::TIME WARNING: "
                << " Norm will be off by " << std::abs( tobeone - overlaps[ 0 ] ) << " due to Krylov evolution. Should be 0.0\n";
   }

   op->DSSum( krylovSpaceDimension, result, &krylovBasisVectors[ 0 ], &krylovBasisSyBookkeepers[ 0 ], mpsOut, bkOut, scheme );

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Delete all the allocated stuff
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

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

void CheMPS2::TimeEvolution::Propagate( SyBookkeeper * initBK, CTensorT ** initMPS,
                                        const double time_step, const double time_final,
                                        const int kry_size, const bool doImaginary, 
                                        const bool doDumpFCI, const bool doDump2RDM ) {
   std::cout << "\n";
   std::cout << "   Starting to propagate MPS\n";
   std::cout << "\n";

   const hid_t outputID    = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, "/Output", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   const hsize_t dimarray1 = 1;

   struct timeval start;
   gettimeofday( &start, NULL );

   HamiltonianOperator * hamOp = new HamiltonianOperator( prob );

   SyBookkeeper * MPSBK = new SyBookkeeper( *initBK );
   CTensorT ** MPS      = new CTensorT *[ L ];
   for ( int index = 0; index < L; index++ ) {
      MPS[ index ] = new CTensorT( initMPS[ index ] );
   }

   double first_energy;

   for ( double t = 0.0; t < time_final; t += time_step ) {

      int * actdims          = new int[ L + 1 ];
      const hsize_t numInst  = scheme->get_number();
      int * MaxMs            = new int[ numInst ];
      int * CutOs            = new int[ numInst ];
      int * NSwes            = new int[ numInst ];
      const double normOfMPS = norm( MPS );
      const double energy    = std::real( hamOp->ExpectationValue( MPS, MPSBK ) );
      COneDM * theodm        = new COneDM( MPS, MPSBK, prob );
      double * oedmre        = new double[ L * L ];
      double * oedmim        = new double[ L * L ];
      double * oedmdmrgre    = new double[ L * L ];
      double * oedmdmrgim    = new double[ L * L ];
      theodm->gOEDMReHamil( oedmre );
      theodm->gOEDMImHamil( oedmim );
      theodm->gOEDMReDMRG( oedmdmrgre );
      theodm->gOEDMImDMRG( oedmdmrgim );

      if ( t == 0.0 ){
         first_energy = energy;
      }

      struct timeval end;
      gettimeofday( &end, NULL );
      const double elapsed = ( end.tv_sec - start.tv_sec ) + 1e-6 * ( end.tv_usec - start.tv_usec );

      std::cout << hashline;
      std::cout                                                 << "\n";
      std::cout << "   MPS time step"                           << "\n";
      std::cout                                                 << "\n";
      std::cout << "   Duration since start " << elapsed << " seconds\n";
      std::cout                                                 << "\n";
      std::cout << "   t    = " << t                            << "\n";
      std::cout << "   Tmax = " << time_final                   << "\n";
      std::cout << "   dt   = " << time_step                    << "\n";
      std::cout << "   KryS = " << kry_size                     << "\n";
      std::cout                                                 << "\n";
      std::cout << "   matrix product state dimensions:             \n";
      std::cout << "   ";
      for ( int i = 0; i < L + 1; i++ ) { std::cout << std::setw( 5 ) << i; }
      std::cout                                                 << "\n";
      std::cout << "   ";
      for ( int i = 0; i < L + 1; i++ ) { actdims[ i ] = MPSBK->gTotDimAtBound( i ); std::cout << std::setw( 5 ) << actdims[ i ]; }
      std::cout                                                 << "\n";
      std::cout                                                 << "\n";
      std::cout << "   MaxM = ";
      for ( int inst = 0; inst < numInst; inst++ ) { MaxMs[ inst ] = scheme->get_D( inst ); std::cout << MaxMs[ inst ] << " "; }
      std::cout                                                 << "\n";
      std::cout << "   CutO = ";
      for ( int inst = 0; inst < numInst; inst++ ) { CutOs[ inst ] = scheme->get_cut_off( inst ); std::cout << CutOs[ inst ] << " "; }
      std::cout                                                 << "\n";
      std::cout << "   NSwes = ";
      for ( int inst = 0; inst < numInst; inst++ ) { NSwes[ inst ] = scheme->get_max_sweeps( inst ); std::cout << NSwes[ inst ] << " "; }
      std::cout                                                 << "\n";
      std::cout                                                 << "\n";
      std::cout << "   Norm      = " << normOfMPS               << "\n";
      std::cout << "   Energy    = " << energy                  << "\n";
      std::cout                                                 << "\n";
      std::cout << "  occupation numbers of molecular orbitals:\n";
      std::cout << "   ";
      for ( int i = 0; i < L; i++ ) { std::cout << std::setw( 20 ) << std::fixed << std::setprecision( 15 ) << oedmre[ i + L * i ]; }
      std::cout                                                 << "\n";

      char dataPointname[ 1024 ];
      sprintf( dataPointname, "/Output/DataPoint%.5f", t );
      const hid_t dataPointID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataPointname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
      const hsize_t Lposize = L + 1;
      hsize_t Lsq[ 2 ]; Lsq[ 0 ] = L; Lsq[ 1 ] = L;
      HDF5_MAKE_DATASET( dataPointID, "chrono",         1, &dimarray1, H5T_NATIVE_DOUBLE,  &elapsed    );
      HDF5_MAKE_DATASET( dataPointID, "t",              1, &dimarray1, H5T_NATIVE_DOUBLE,  &t          );
      HDF5_MAKE_DATASET( dataPointID, "Tmax",           1, &dimarray1, H5T_NATIVE_DOUBLE,  &time_final );
      HDF5_MAKE_DATASET( dataPointID, "dt",             1, &dimarray1, H5T_NATIVE_DOUBLE,  &time_step  );
      HDF5_MAKE_DATASET( dataPointID, "KryS",           1, &dimarray1, H5T_STD_I32LE,      &kry_size   );
      HDF5_MAKE_DATASET( dataPointID, "MPSDims",        1, &Lposize,   H5T_STD_I32LE,      actdims     );
      HDF5_MAKE_DATASET( dataPointID, "MaxMs",          1, &numInst,   H5T_STD_I32LE,      MaxMs       );
      HDF5_MAKE_DATASET( dataPointID, "CutOs",          1, &numInst,   H5T_STD_I32LE,      CutOs       );
      HDF5_MAKE_DATASET( dataPointID, "NSwes",          1, &numInst,   H5T_STD_I32LE,      NSwes       );
      HDF5_MAKE_DATASET( dataPointID, "Norm",           1, &dimarray1, H5T_NATIVE_DOUBLE,  &normOfMPS  );
      HDF5_MAKE_DATASET( dataPointID, "Energy",         1, &dimarray1, H5T_NATIVE_DOUBLE,  &energy     );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_REAL",      2, Lsq,        H5T_NATIVE_DOUBLE,  oedmre      );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_IMAG",      2, Lsq,        H5T_NATIVE_DOUBLE,  oedmim      );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_DMRG_REAL", 2, Lsq,        H5T_NATIVE_DOUBLE,  oedmdmrgre  );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_DMRG_IMAG", 2, Lsq,        H5T_NATIVE_DOUBLE,  oedmdmrgim  );

      delete[] actdims;
      delete[] MaxMs;
      delete[] CutOs;
      delete[] NSwes;
      delete[] oedmre;
      delete[] oedmim;
      delete[] oedmdmrgre;
      delete[] oedmdmrgim;

      delete theodm;
      
      if ( doDumpFCI ) {
         std::vector< std::vector< int > > alphasOut;
         std::vector< std::vector< int > > betasOut;
         std::vector< double > coefsRealOut;
         std::vector< double > coefsImagOut;
         const hsize_t Lsize = L;
         getFCITensor( prob, MPS, alphasOut, betasOut, coefsRealOut, coefsImagOut );

         char dataFCIName[ 1024 ];
         sprintf( dataFCIName, "%s/FCICOEF", dataPointname );
         const hid_t FCIID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataFCIName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;

         for ( int l = 0; l < alphasOut.size(); l++ ) {
            char dataFCINameN[ 1024 ];
            sprintf( dataFCINameN, "%s/FCICOEF/%i", dataPointname, l );
            const hid_t FCIID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataFCINameN, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
            HDF5_MAKE_DATASET( FCIID, "FCI_ALPHAS", 1, &Lsize,     H5T_NATIVE_INT,    &alphasOut   [ l ][ 0 ] );
            HDF5_MAKE_DATASET( FCIID, "FCI_BETAS",  1, &Lsize,     H5T_NATIVE_INT,    &betasOut    [ l ][ 0 ] );
            HDF5_MAKE_DATASET( FCIID, "FCI_REAL",   1, &dimarray1, H5T_NATIVE_DOUBLE, &coefsRealOut[ l ]      );
            HDF5_MAKE_DATASET( FCIID, "FCI_IMAG",   1, &dimarray1, H5T_NATIVE_DOUBLE, &coefsImagOut[ l ]      );
         }
      }

      if ( doDump2RDM ) {
         const hsize_t Lsize =  L * L * L * L;

         CTwoDM * thetdm = new CTwoDM( MPSBK, prob );
         CTwoDMBuilder * thetdmbuilder = new CTwoDMBuilder( prob, MPS, MPSBK );
         thetdmbuilder->Build2RDM( thetdm );

         double * tedm_real  = new double[ L * L * L * L ];
         double * tedm_imag  = new double[ L * L * L * L ];
         for( int idxA = 0; idxA < L; idxA++ ){
            for( int idxB = 0; idxB < L; idxB++ ){
               for( int idxC = 0; idxC < L; idxC++ ){
                  for( int idxD = 0; idxD < L; idxD++ ){
                     tedm_real[ idxA + L * ( idxB + L * ( idxC + L * idxD ) ) ] = std::real( thetdm->getTwoDMA_HAM( idxA, idxB, idxC, idxD ) );
                     tedm_imag[ idxA + L * ( idxB + L * ( idxC + L * idxD ) ) ] = (-1.0) * std::imag( thetdm->getTwoDMA_HAM( idxA, idxB, idxC, idxD ) );
                  }
               }
            }
         }

         HDF5_MAKE_DATASET( dataPointID, "TEDM_REAL", 1, &Lsize, H5T_NATIVE_DOUBLE, tedm_real );
         HDF5_MAKE_DATASET( dataPointID, "TEDM_IMAG", 1, &Lsize, H5T_NATIVE_DOUBLE, tedm_imag );

         delete[] tedm_real;
         delete[] tedm_imag;
         delete thetdmbuilder;
         delete thetdm;
      }      
      std::cout << "\n";

      if ( t + time_step < time_final ) {
         SyBookkeeper * MPSBKDT = new SyBookkeeper( prob, scheme->get_D( 0 ) );
         CTensorT ** MPSDT      = new CTensorT *[ L ];
         for ( int index = 0; index < L; index++ ) {
            MPSDT[ index ] = new CTensorT( index, MPSBKDT );
            MPSDT[ index ]->random();
         }
         double normDT = norm( MPSDT );
         MPSDT[ 0 ]->number_operator( 0.0, 1.0 / normDT );

         doStep_arnoldi( time_step, time_final, kry_size, -1.0 * first_energy, doImaginary, MPS, MPSBK, MPSDT, MPSBKDT );

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