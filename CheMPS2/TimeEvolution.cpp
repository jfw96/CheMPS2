
#include "TimeEvolution.h"
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <algorithm>
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

double CheMPS2::TimeEvolution::calcWieght( int nHoles, int nParticles, Problem * probState, CTensorT ** mpsState, SyBookkeeper * bkState, const int * hf_state ){

   const int L = prob->gL();
   double result = 0;

   int* myoccus = new int[ 2 * L ];
   for( int idx = 0; idx < 2 * L; idx++ ){
      if( idx < prob->gN() ){
         myoccus[ idx ] = 1;
      } else {
         myoccus[ idx ] = 0;
      }
   }

   std::sort( myoccus, myoccus + 2 * L );
   do {

      bool alphaEqualsBeta = true;

      int* alphas = &myoccus[ 0 ];
      int* betas = &myoccus[ L ];

      int iHoles = 0;
      int iParticles = 0;

      for( int idx = 0; idx < L; idx++ ) { 
         if ( alphas[ idx ] + betas[ idx ] < hf_state[ idx ] ){
            iHoles += hf_state[ idx ] - ( alphas[ idx ] + betas[ idx ] );
         }
         alphaEqualsBeta = alphaEqualsBeta && ( alphas[ idx ] == betas[ idx ] );
      }

      for( int idx = 0; idx < L; idx++ ) { 
         if ( alphas[ idx ] + betas[ idx ] > hf_state[ idx ] ){
            iParticles += ( alphas[ idx ] + betas[ idx ] ) - hf_state[ idx ];
         }
      }

      if( ( iHoles == nHoles ) && ( iParticles == nParticles ) ){
         result += std::pow( std::abs( getFCICoefficient( probState, mpsState, alphas, betas ) ), 2.0 );
      }

   } while ( std::next_permutation( myoccus, myoccus + 2 * L ) );

   delete[] myoccus;

   return result;
}


void CheMPS2::TimeEvolution::doStep_euler( const double time_step, const int kry_size, dcomplex offset, const bool backwards, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   dcomplex step = backwards ? -time_step : dcomplex( 0.0, -1.0 * time_step );

   HamiltonianOperator * op = new HamiltonianOperator( prob, 0.0*offset );

   SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D( 0 ) );
   CTensorT ** mpsTemp   = new CTensorT *[ L ];
   for ( int index = 0; index < L; index++ ) {
      mpsTemp[ index ] = new CTensorT( index, bkTemp );
      mpsTemp[ index ]->random();
   }
   normalize( L, mpsTemp );

   op->DSApply( mpsIn, bkIn, mpsTemp, bkTemp, scheme );
   scale( step, L, mpsTemp );

   CTensorT*** states = new CTensorT**[ 2 ];
   states[ 0 ] = mpsIn;
   states[ 1 ] = mpsTemp;
   SyBookkeeper ** bkers = new SyBookkeeper * [ 2 ];
   bkers[ 0 ] = bkIn; 
   bkers[ 1 ] = bkTemp;

   dcomplex coefs[] = { 1.0, 1.0 };
   op->DSSum( 2, &coefs[0], states, bkers, mpsOut, bkOut, scheme );

   for ( int site = 0; site < L; site++ ) {
      delete mpsTemp[ site ];
   }
   delete[] mpsTemp;
   delete bkTemp;
   delete[] states;
   delete[] bkers;
   delete op;

}

void CheMPS2::TimeEvolution::doStep_runge_kutta( const double time_step, const int kry_size, dcomplex offset, const bool backwards, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   dcomplex step = backwards ? dcomplex( 0.0, 1.0 * time_step ) : dcomplex( 0.0, -1.0 * time_step );

   HamiltonianOperator * op = new HamiltonianOperator( prob, offset );

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Generating Runge-Kutta vectors
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   CTensorT *** rungeKuttaVectors          = new CTensorT **[ 5 ];
   SyBookkeeper ** rungeKuttaSyBookkeepers = new SyBookkeeper *[ 5 ];

   // Inital state
   {
      rungeKuttaVectors[ 0 ] = mpsIn; 
      rungeKuttaSyBookkeepers[ 0 ] = bkIn;
   }

   // First vector
   {
      rungeKuttaSyBookkeepers[ 1 ] = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      rungeKuttaVectors[ 1 ]   = new CTensorT * [ L ];
      for ( int index = 0; index < L; index++ ) {
         rungeKuttaVectors[ 1 ][ index ] = new CTensorT( index, rungeKuttaSyBookkeepers[ 1 ] );
         rungeKuttaVectors[ 1 ][ index ]->random();
      }
      normalize( L, rungeKuttaVectors[ 1 ] );

      op->DSApply( mpsIn, bkIn, rungeKuttaVectors[ 1 ], rungeKuttaSyBookkeepers[ 1 ], scheme );
      scale( step, L, rungeKuttaVectors[ 1 ] );
   }

   // Second vector
   {

      SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }
      normalize( L, mpsTemp );

      dcomplex coefs[] = { 1.0, 0.5 };
      CTensorT*** states = new CTensorT**[ 2 ];
      states[ 0 ] = rungeKuttaVectors[ 0 ];
      states[ 1 ] = rungeKuttaVectors[ 1 ];
      SyBookkeeper ** bkers = new SyBookkeeper * [ 2 ];
      bkers[ 0 ] = rungeKuttaSyBookkeepers[ 0 ]; 
      bkers[ 1 ] = rungeKuttaSyBookkeepers[ 1 ];
      op->DSSum( 2, &coefs[0], states, bkers, mpsTemp, bkTemp, scheme );

      rungeKuttaSyBookkeepers[ 2 ] = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      rungeKuttaVectors[ 2 ]   = new CTensorT * [ L ];
      for ( int index = 0; index < L; index++ ) {
         rungeKuttaVectors[ 2 ][ index ] = new CTensorT( index, rungeKuttaSyBookkeepers[ 2 ] );
         rungeKuttaVectors[ 2 ][ index ]->random();
      }
      normalize( L, rungeKuttaVectors[ 2 ] );

      op->DSApply( mpsTemp, bkTemp, rungeKuttaVectors[ 2 ], rungeKuttaSyBookkeepers[ 2 ], scheme );
      scale( step, L, rungeKuttaVectors[ 2 ] );

      for ( int site = 0; site < L; site++ ) {
         delete mpsTemp[ site ];
      }
      delete[] mpsTemp;
      delete bkTemp;
      delete[] states;
      delete[] bkers;

   }

   // Third vector
   {

      SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }
      normalize( L, mpsTemp );

      dcomplex coefs[] = { 1.0, 0.5 };
      CTensorT*** states = new CTensorT**[ 3 ];
      states[ 0 ] = rungeKuttaVectors[ 0 ];
      states[ 1 ] = rungeKuttaVectors[ 2 ];
      SyBookkeeper ** bkers = new SyBookkeeper * [ 3 ];
      bkers[ 0 ] = rungeKuttaSyBookkeepers[ 0 ]; 
      bkers[ 1 ] = rungeKuttaSyBookkeepers[ 2 ];
      op->DSSum( 2, &coefs[0], states, bkers, mpsTemp, bkTemp, scheme );

      rungeKuttaSyBookkeepers[ 3 ] = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      rungeKuttaVectors[ 3 ]   = new CTensorT * [ L ];
      for ( int index = 0; index < L; index++ ) {
         rungeKuttaVectors[ 3 ][ index ] = new CTensorT( index, rungeKuttaSyBookkeepers[ 3 ] );
         rungeKuttaVectors[ 3 ][ index ]->random();
      }
      normalize( L, rungeKuttaVectors[ 3 ] );

      op->DSApply( mpsTemp, bkTemp, rungeKuttaVectors[ 3 ], rungeKuttaSyBookkeepers[ 3 ], scheme );
      scale( step, L, rungeKuttaVectors[ 3 ] );

      for ( int site = 0; site < L; site++ ) {
         delete mpsTemp[ site ];
      }
      delete[] mpsTemp;
      delete bkTemp;
      delete[] states;
      delete[] bkers;
   }

   // Fourth vector
   {

      SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }
      normalize( L, mpsTemp );

      dcomplex coefs[] = { 1.0, 1.0 };
      CTensorT*** states = new CTensorT**[ 2 ];
      states[ 0 ] = rungeKuttaVectors[ 0 ];
      states[ 1 ] = rungeKuttaVectors[ 3 ];
      SyBookkeeper ** bkers = new SyBookkeeper * [ 2 ];
      bkers[ 0 ] = rungeKuttaSyBookkeepers[ 0 ]; 
      bkers[ 1 ] = rungeKuttaSyBookkeepers[ 3 ];
      op->DSSum( 2, &coefs[0], states, bkers, mpsTemp, bkTemp, scheme );

      rungeKuttaSyBookkeepers[ 4 ] = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      rungeKuttaVectors[ 4 ]   = new CTensorT * [ L ];
      for ( int index = 0; index < L; index++ ) {
         rungeKuttaVectors[ 4 ][ index ] = new CTensorT( index, rungeKuttaSyBookkeepers[ 4 ] );
         rungeKuttaVectors[ 4 ][ index ]->random();
      }
      normalize( L, rungeKuttaVectors[ 4 ] );

      op->DSApply( mpsTemp, bkTemp, rungeKuttaVectors[ 4 ], rungeKuttaSyBookkeepers[ 4 ], scheme );
      scale( step, L, rungeKuttaVectors[ 4 ] );

      for ( int site = 0; site < L; site++ ) {
         delete mpsTemp[ site ];
      }
      delete[] mpsTemp;
      delete bkTemp;
      delete[] states;
      delete[] bkers;
   }

   // Add the whole thing
   dcomplex coefs[] = { 1.0, 1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0 };
   op->DSSum( 5, &coefs[0], rungeKuttaVectors, rungeKuttaSyBookkeepers, mpsOut, bkOut, scheme );

   for ( int cnt = 1; cnt < 5; cnt++ ) {
      for ( int site = 0; site < L; site++ ) {
         delete rungeKuttaVectors[ cnt ][ site ];
      }
   delete[] rungeKuttaVectors[ cnt ];
   delete rungeKuttaSyBookkeepers[ cnt ];
   }
   delete op;
}

void CheMPS2::TimeEvolution::doStep_arnoldi( const double time_step, const int kry_size, dcomplex offset, const bool backwards, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   int krylovSpaceDimension = kry_size;

   dcomplex step = backwards ? dcomplex( 0.0, 1.0 * time_step ) : dcomplex( 0.0, -1.0 * time_step );

   HamiltonianOperator * op = new HamiltonianOperator( prob, offset );

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Generating Krylov Space vectors
   ////
   ////////////////////////////////////////////////////////////////////////////////////////
   std::cout << "   Krylov space vectors:\n";

   CTensorT *** krylovBasisVectors          = new CTensorT **[ krylovSpaceDimension ];
   SyBookkeeper ** krylovBasisSyBookkeepers = new SyBookkeeper *[ krylovSpaceDimension ];

   // First vector is starting state
   krylovBasisVectors[ 0 ]       = mpsIn;
   krylovBasisSyBookkeepers[ 0 ] = bkIn;
   std::cout << "      i = " << 0 << " ";
   std::cout << "MPS dimensions:";
   for (int i = 0; i <= L; i++){
      std::cout << " " << krylovBasisSyBookkeepers[ 0 ]->gTotDimAtBound( i );
   }
   std::cout << std::endl;

   // Genereate remaining
   for ( int kry = 1; kry < krylovSpaceDimension; kry++ ) {

      struct timeval startVec, endVec;
      gettimeofday( &startVec, NULL );

      SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D( 0 ) );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }
      normalize( L, mpsTemp );

      op->DSApply( krylovBasisVectors[ kry - 1 ], krylovBasisSyBookkeepers[ kry - 1 ],
                   mpsTemp, bkTemp, scheme );

      normalize( L, mpsTemp );

      krylovBasisVectors[ kry ]       = mpsTemp;
      krylovBasisSyBookkeepers[ kry ] = bkTemp;

      gettimeofday( &endVec, NULL );
      const double elapsed = ( endVec.tv_sec - startVec.tv_sec ) + 1e-6 * ( endVec.tv_usec - startVec.tv_usec );

      std::cout << "      i = " << kry << " ";
      std::cout << "MPS dimensions:";
      for (int i = 0; i <= L; i++){
         std::cout << " " << krylovBasisSyBookkeepers[ kry ]->gTotDimAtBound( i );
      }
      std::cout << " time elapsed: " << elapsed << " seconds\n";
      
   }
   std::cout << "\n";

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Building S and H
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   dcomplex * krylovHamiltonian = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   dcomplex * overlaps          = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   for ( int irow = 0; irow < krylovSpaceDimension; irow++ ) {
      for( int icol = irow; icol < krylovSpaceDimension; icol++ ) {
         overlaps[ irow + icol * krylovSpaceDimension ]          = overlap( krylovBasisVectors[ irow ], krylovBasisVectors[ icol ] );
         overlaps[ icol + irow * krylovSpaceDimension ]          = std::conj( overlaps[ irow + icol * krylovSpaceDimension ] );
         krylovHamiltonian[ irow + icol * krylovSpaceDimension ] = op->Overlap( krylovBasisVectors[ irow ], krylovBasisSyBookkeepers[ irow ], krylovBasisVectors[ icol ], krylovBasisSyBookkeepers[ icol ] );
         krylovHamiltonian[ icol + irow * krylovSpaceDimension ] = std::conj( krylovHamiltonian[ irow + icol * krylovSpaceDimension ] );
      }
   }

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
   assert( info_lu == 0);

   dcomplex * work = new dcomplex[ krylovSpaceDimension ];
   int info_inve;
   zgetri_( &krylovSpaceDimension, overlaps_inv, &krylovSpaceDimension, piv, work, &krylovSpaceDimension, &info_inve );
   assert( info_inve == 0);

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Test the inverse
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   dcomplex * tobeiden = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   char notrans2     = 'N';
   dcomplex zeroC2   = 0.0;
   dcomplex oneC2    = 1.0;
   zgemm_( &notrans2, &notrans2, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension,
           &oneC2, overlaps, &krylovSpaceDimension, overlaps_inv, &krylovSpaceDimension, &zeroC2, tobeiden, &krylovSpaceDimension );

   for ( int irow = 0; irow < krylovSpaceDimension; irow++ ){
      for ( int icol = 0; icol < krylovSpaceDimension; icol++ ){
         std::cout << std::real( tobeiden[ irow +  icol * krylovSpaceDimension ] ) << " ";
      }
      std::cout << std::endl;
   }
   delete[] tobeiden;

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
   int deg        = 10;
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

void CheMPS2::TimeEvolution::Propagate( const char time_type, const double time_step_major, 
                                        const double time_step_minor, const double time_final, 
                                        CTensorT ** mpsIn, SyBookkeeper * bkIn, 
                                        const int kry_size,
                                        const bool backwards, const bool doDumpFCI, 
                                        const bool doDump2RDM, const int nWeights,
                                        const int * hfState ) {
   std::cout << "\n";
   std::cout << "   Starting to propagate MPS\n";
   std::cout << "\n";

   const hid_t outputID    = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, "/Output", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
   const hsize_t dimarray1 = 1;

   struct timeval start;
   gettimeofday( &start, NULL );

   HamiltonianOperator * hamOp = new HamiltonianOperator( prob );

   CheMPS2::SyBookkeeper * MPSBK  = new CheMPS2::SyBookkeeper( *bkIn );
   CheMPS2::CTensorT    ** MPS    = new CheMPS2::CTensorT *[ prob->gL() ];

   for ( int index = 0; index < prob->gL(); index++ ) {
      MPS[ index ] = new CheMPS2::CTensorT( mpsIn[ index ] );
   }

   double first_energy;
   int deltaN = 0;
   if( nWeights > 0 ){
      int nElecHF = 0; for ( int index = 0; index < prob->gL(); index++ ) { nElecHF += hfState[ index ]; }
      deltaN = nElecHF - prob->gN();
   }

   for ( double t = 0.0; t < time_final; t += time_step_major ) {

      int * actdims          = new int[ L + 1 ];
      const hsize_t numInst  = scheme->get_number();
      int * MaxMs            = new int[ numInst ];
      int * CutOs            = new int[ numInst ];
      int * NSwes            = new int[ numInst ];
      const double normOfMPS = norm( MPS );
      const double energy    = std::real( hamOp->ExpectationValue( MPS, MPSBK ) );
      const dcomplex oInit   = overlap( mpsIn, MPS );
      const double reoInit   = std::real( oInit );
      const double imoInit   = std::imag( oInit );
      COneDM * theodm        = new COneDM( MPS, MPSBK, prob );
      double * oedmre        = new double[ L * L ];
      double * oedmim        = new double[ L * L ];
      double * oedmdmrgre    = new double[ L * L ];
      double * oedmdmrgim    = new double[ L * L ];
      theodm->gOEDMReHamil( oedmre );
      theodm->gOEDMImHamil( oedmim );
      theodm->gOEDMReDMRG( oedmdmrgre );
      theodm->gOEDMImDMRG( oedmdmrgim );

      int*     nHoles = new int[ nWeights ];
      int* nParticles = new int[ nWeights ];
      double* weights = new double[ nWeights ];

      for( int iWeight = 0; iWeight < nWeights; iWeight++ ){
         nHoles[ iWeight ] = iWeight + deltaN;
         nParticles[ iWeight ] = iWeight;
         weights[ iWeight ] =  calcWieght( iWeight + deltaN, iWeight, prob, MPS, MPSBK, hfState );
      }

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
      std::cout << "   t        = " << t                            << "\n";
      std::cout << "   Tmax     = " << time_final                   << "\n";
      std::cout << "   dt major = " << time_step_major              << "\n";
      std::cout << "   dt minor = " << time_step_minor              << "\n";
      std::cout << "   KryS     = " << kry_size                     << "\n";
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
      std::cout << "   Re(OInit) = " << reoInit                 << "\n";
      std::cout << "   Im(OInit) = " << imoInit                 << "\n";
      std::cout                                                 << "\n";
      std::cout << "  occupation numbers of molecular orbitals:\n";
      std::cout << "   ";
      for ( int i = 0; i < L; i++ ) { std::cout << std::setw( 20 ) << std::fixed << std::setprecision( 15 ) << oedmre[ i + L * i ]; }
      std::cout                                                 << "\n";
      std::cout                                                 << "\n";

      for( int iWeight = 0; iWeight < nWeights; iWeight++ ){
         std::cout << "  " << nHoles[ iWeight ] <<  "h" << nParticles[ iWeight] << "p-weight  = " << weights[ iWeight ] << "\n";
      }
      std::cout                                                 << "\n";

      char dataPointname[ 1024 ];
      sprintf( dataPointname, "/Output/DataPoint%.5f", t );
      const hid_t dataPointID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataPointname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;
      const hsize_t Lposize = L + 1;
      hsize_t Lsq[ 2 ]; Lsq[ 0 ] = L; Lsq[ 1 ] = L;
      hsize_t weightSze = nWeights;
      HDF5_MAKE_DATASET( dataPointID, "chrono",         1, &dimarray1, H5T_NATIVE_DOUBLE,  &elapsed         );
      HDF5_MAKE_DATASET( dataPointID, "t",              1, &dimarray1, H5T_NATIVE_DOUBLE,  &t               );
      HDF5_MAKE_DATASET( dataPointID, "Tmax",           1, &dimarray1, H5T_NATIVE_DOUBLE,  &time_final      );
      HDF5_MAKE_DATASET( dataPointID, "dtmajor",        1, &dimarray1, H5T_NATIVE_DOUBLE,  &time_step_major );
      HDF5_MAKE_DATASET( dataPointID, "dtminor",        1, &dimarray1, H5T_NATIVE_DOUBLE,  &time_step_minor );
      HDF5_MAKE_DATASET( dataPointID, "KryS",           1, &dimarray1, H5T_STD_I32LE,      &kry_size        );
      HDF5_MAKE_DATASET( dataPointID, "MPSDims",        1, &Lposize,   H5T_STD_I32LE,      actdims          );
      HDF5_MAKE_DATASET( dataPointID, "MaxMs",          1, &numInst,   H5T_STD_I32LE,      MaxMs            );
      HDF5_MAKE_DATASET( dataPointID, "CutOs",          1, &numInst,   H5T_STD_I32LE,      CutOs            );
      HDF5_MAKE_DATASET( dataPointID, "NSwes",          1, &numInst,   H5T_STD_I32LE,      NSwes            );
      HDF5_MAKE_DATASET( dataPointID, "Norm",           1, &dimarray1, H5T_NATIVE_DOUBLE,  &normOfMPS       );
      HDF5_MAKE_DATASET( dataPointID, "Energy",         1, &dimarray1, H5T_NATIVE_DOUBLE,  &energy          );
      HDF5_MAKE_DATASET( dataPointID, "REOInit",        1, &dimarray1, H5T_NATIVE_DOUBLE,  &reoInit         );
      HDF5_MAKE_DATASET( dataPointID, "IMOInit",        1, &dimarray1, H5T_NATIVE_DOUBLE,  &imoInit         );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_REAL",      2, Lsq,        H5T_NATIVE_DOUBLE,  oedmre           );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_IMAG",      2, Lsq,        H5T_NATIVE_DOUBLE,  oedmim           );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_DMRG_REAL", 2, Lsq,        H5T_NATIVE_DOUBLE,  oedmdmrgre       );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_DMRG_IMAG", 2, Lsq,        H5T_NATIVE_DOUBLE,  oedmdmrgim       );
      HDF5_MAKE_DATASET( dataPointID, "nHoles",         1, &weightSze, H5T_STD_I32LE,      nHoles           );
      HDF5_MAKE_DATASET( dataPointID, "nParticles",     1, &weightSze, H5T_STD_I32LE,      nParticles       );
      HDF5_MAKE_DATASET( dataPointID, "weights",        1, &weightSze, H5T_NATIVE_DOUBLE,  weights          );

      delete[] actdims;
      delete[] MaxMs;
      delete[] CutOs;
      delete[] NSwes;
      delete[] oedmre;
      delete[] oedmim;
      delete[] oedmdmrgre;
      delete[] oedmdmrgim;
      delete[] nHoles;
      delete[] nParticles;
      delete[] weights;

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

      if ( t + time_step_major < time_final ) {
         for( double t_minor = 0.0; (time_step_major - t_minor) > 1e-6; t_minor+=time_step_minor ) {

            SyBookkeeper * MPSBKDT = new SyBookkeeper( prob, scheme->get_D( 0 ) );
            CTensorT ** MPSDT      = new CTensorT *[ L ];
            for ( int index = 0; index < L; index++ ) {
               MPSDT[ index ] = new CTensorT( index, MPSBKDT );
               MPSDT[ index ]->random();
            }
            normalize( L, MPSDT );

            if( time_type == 'K' ){
               doStep_arnoldi( time_step_minor, kry_size, -0.0 * first_energy, backwards, MPS, MPSBK, MPSDT, MPSBKDT );
            } else if ( time_type == 'R' ){
               doStep_runge_kutta( time_step_minor, kry_size, -1.0 * first_energy, backwards, MPS, MPSBK, MPSDT, MPSBKDT );
            } else if ( time_type == 'E' ){
               doStep_euler( time_step_minor, kry_size, -1.0 * first_energy, backwards, MPS, MPSBK, MPSDT, MPSBKDT );
            }

            for ( int site = 0; site < L; site++ ) {
               delete MPS[ site ];
            }
            delete[] MPS;
            delete MPSBK;

            MPS   = MPSDT;
            MPSBK = MPSBKDT;

            if ( backwards ) {
               normalize( L, MPS );
            }
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