
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

double CheMPS2::TimeEvolution::calc1h0p( Problem * probState, CTensorT ** mpsState, SyBookkeeper * bkState, const int * hf_state ){
   
   const int L = prob->gL();
   double result = 0;

// PARALLEL
#pragma omp parallel
   {
      int* alphas = new int[ L ];
      int* betas = new int[ L ];

      for( int idx = 0; idx < L; idx++ ){
         if( hf_state[ idx ] == 2 ){
            betas[ idx ] = 1;
            alphas[ idx ] = 1;
         } else if ( hf_state[ idx ] == 0 ) {
            betas[ idx ] = 0;
            alphas[ idx ] = 0;
         } else {
            std::cerr << "CheMPS2::TimeEvolution::calc1h0p is implemented for closed shell molecules only. Exiting..." << std::endl;
            abort();
         }
      }

#pragma omp for schedule( dynamic )
      for( int idx = 0; idx < L; idx++ ){
         if( alphas[ idx ] == 1 ){
            alphas[ idx ] = 0;
            result += 2.0 * std::pow( std::abs( getFCICoefficient( probState, mpsState, alphas, betas ) ), 2.0 );
            alphas[ idx ] = 1;
         }
      }
   
      delete[] alphas;
      delete[] betas;
   }

   return result;
}

double CheMPS2::TimeEvolution::calc2h1p( Problem * probState, CTensorT ** mpsState, SyBookkeeper * bkState, const int * hf_state ){
   
   const int L = prob->gL();
   double result = 0;

   int* alphas = new int[ L ];
   int* betas = new int[ L ];

   for( int idx = 0; idx < L; idx++ ){
      if( hf_state[ idx ] == 2 ){
         betas[ idx ] = 1;
         alphas[ idx ] = 1;
      } else if ( hf_state[ idx ] == 0 ) {
         betas[ idx ] = 0;
         alphas[ idx ] = 0;
      } else {
         std::cerr << "CheMPS2::TimeEvolution::calc2h1p is implemented for closed shell molecules only. Exiting..." << std::endl;
         abort();
      }
   }

   for( int a = 0; a < L; a++ ){
      for( int i = 0; i < L; i++ ){
         for( int j = 0; j < i; j++ ){
            for( int twoKappa = -1; twoKappa <= 1; twoKappa+=2 ){
               for( int twoTau = -1; twoTau <= 1; twoTau+=2 ){
                  for( int twoSigma = -1; twoSigma <= 1; twoSigma+=2 ){
                     int * toCreate = ( twoKappa == -1 ) ? alphas : betas;
                     int * toAnniA = ( twoTau == -1 ) ? alphas : betas;
                     int * toAnniB = ( twoSigma == -1 ) ? alphas : betas;

                     toCreate[ a ]++;
                     toAnniA[ i ]--;
                     toAnniB[ j ]--;

                     if( ( toAnniB[ j ] >= 0 ) && ( toAnniA[ i ] >= 0 ) && ( toCreate[ a ] <= 1 ) && ( i != j ) && ( i != a ) && ( j != a ) ){
                        result += std::pow( std::abs( getFCICoefficient( probState, mpsState, alphas, betas ) ), 2.0 );
                     }

                     toCreate[ a ]--;
                     toAnniA[ i ]++;
                     toAnniB[ j ]++;

                  }
               }
            }
         }
      }
   }

   delete[] alphas;
   delete[] betas;

   return result;
}


double CheMPS2::TimeEvolution::calcWieght( int nHoles, int nParticles, Problem * probState, CTensorT ** mpsState, SyBookkeeper * bkState, const int * hf_state ){

   if( ( nHoles == 1 ) && ( nParticles == 0 ) ){
      return calc1h0p( probState, mpsState, bkState, hf_state );
   }

   if( ( nHoles == 2 ) && ( nParticles == 1 ) ){
      return calc2h1p( probState, mpsState, bkState, hf_state );
   }


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

   dcomplex step = backwards ? dcomplex( 0.0, 1.0 * time_step ) : dcomplex( 0.0, -1.0 * time_step );

   HamiltonianOperator * op = new HamiltonianOperator( prob, offset );

   SyBookkeeper * bkTemp = new SyBookkeeper( prob, scheme->get_D( 0 ) );
   CTensorT ** mpsTemp   = new CTensorT *[ L ];
   for ( int index = 0; index < L; index++ ) {
      mpsTemp[ index ] = new CTensorT( index, bkTemp );
      mpsTemp[ index ]->random();
   }
   normalize( L, mpsTemp );

   op->DSApply( mpsIn, bkIn, mpsTemp, bkTemp, scheme );
   std::cout << overlap( mpsIn, mpsTemp ) << std::endl;
   CTensorT*** states = new CTensorT**[ 2 ];
   states[ 0 ] = mpsIn;
   states[ 1 ] = mpsTemp;
   SyBookkeeper ** bkers = new SyBookkeeper * [ 2 ];
   bkers[ 0 ] = bkIn; 
   bkers[ 1 ] = bkTemp;

   dcomplex * coefs = new dcomplex[ 2 ];
   coefs[ 0 ] = 1.0;
   coefs[ 1 ] = step;
   op->DSSum( 2, &coefs[0], states, bkers, mpsOut, bkOut, scheme );
   std::cout << overlap( mpsIn, mpsOut ) << std::endl;

   for ( int site = 0; site < L; site++ ) {
      delete mpsTemp[ site ];
   }
   delete[] mpsTemp;
   delete bkTemp;
   delete[] states;
   delete[] bkers;
   delete op;
   delete coefs;

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
      CTensorT*** states = new CTensorT**[ 2 ];
      states[ 0 ] = rungeKuttaVectors[ 0 ];
      states[ 1 ] = rungeKuttaVectors[ 2 ];
      SyBookkeeper ** bkers = new SyBookkeeper * [ 2 ];
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

void CheMPS2::TimeEvolution::doStep_arnoldi( const double time_step, 
                                             const int kry_size, 
                                             dcomplex offset, 
                                             const bool backwards, 
                                             const bool do_ortho,
                                             CTensorT ** mpsIn, 
                                             SyBookkeeper * bkIn, 
                                             CTensorT ** mpsOut, 
                                             SyBookkeeper * bkOut ) {

   int krylovSpaceDimension = kry_size;
   char notrans = 'N';
   char trans = 'C';

   dcomplex step = backwards ? dcomplex( 0.0, 1.0 * time_step ) : dcomplex( 0.0, -1.0 * time_step );

   HamiltonianOperator * op = new HamiltonianOperator( prob, offset );

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Generating Krylov Space vectors
   ////
   ////////////////////////////////////////////////////////////////////////////////////////
   std::cout << "  Krylov space vectors:\n";

   CTensorT *** krylovBasisVectors          = new CTensorT **[ krylovSpaceDimension ];
   SyBookkeeper ** krylovBasisSyBookkeepers = new SyBookkeeper *[ krylovSpaceDimension ];


   dcomplex * krylovHamiltonian = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   dcomplex * overlaps          = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   // First vector is starting state
   krylovBasisVectors[ 0 ]       = mpsIn;
   krylovBasisSyBookkeepers[ 0 ] = bkIn;
   std::cout << "      i = " << 0 << " ";
   std::cout << "MPS dimensions:";
   for (int i = 0; i <= L; i++){
      std::cout << " " << krylovBasisSyBookkeepers[ 0 ]->gTotDimAtBound( i );
   }
   std::cout << std::endl;

   krylovHamiltonian[ 0 + 0 * krylovSpaceDimension ] = op->Overlap( krylovBasisVectors[ 0 ], krylovBasisSyBookkeepers[ 0 ], krylovBasisVectors[ 0 ], krylovBasisSyBookkeepers[ 0 ] );
   overlaps[ 0 + 0 * krylovSpaceDimension ]          = overlap( krylovBasisVectors[ 0 ], krylovBasisVectors[ 0 ] );

   for ( int kry = 1; kry < krylovSpaceDimension; kry++ ) {

      struct timeval start, end;
      gettimeofday( &start, NULL );

      SyBookkeeper * bkTemp = new SyBookkeeper( *krylovBasisSyBookkeepers[ kry - 1 ] );
      CTensorT ** mpsTemp   = new CTensorT *[ L ];
      for ( int index = 0; index < L; index++ ) {
         mpsTemp[ index ] = new CTensorT( index, bkTemp );
         mpsTemp[ index ]->random();
      }
      normalize( L, mpsTemp );      

      if ( do_ortho ){
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

      } else {
         op->DSApply( krylovBasisVectors[ kry - 1 ], krylovBasisSyBookkeepers[ kry - 1 ], mpsTemp, bkTemp, scheme );
      }

      normalize( L, mpsTemp );
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
      std::cout << " time elapsed: " << elapsed << " seconds" << " vector norm: " << std::abs( overlaps[ kry + kry * krylovSpaceDimension ] ) << "\n";
      
   }
   std::cout << "\n";

   //////////////////////////////////////////////////////////////////////////////////////
   //
   // Building S and H
   //
   //////////////////////////////////////////////////////////////////////////////////////

   // dcomplex * krylovHamiltonian = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
   // dcomplex * overlaps          = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   // for ( int irow = 0; irow < krylovSpaceDimension; irow++ ) {
   //    for( int icol = irow; icol < krylovSpaceDimension; icol++ ) {
   //       overlaps[ irow + icol * krylovSpaceDimension ]          = overlap( krylovBasisVectors[ irow ], krylovBasisVectors[ icol ] );
   //       overlaps[ icol + irow * krylovSpaceDimension ]          = std::conj( overlaps[ irow + icol * krylovSpaceDimension ] );
   //       krylovHamiltonian[ irow + icol * krylovSpaceDimension ] = op->Overlap( krylovBasisVectors[ irow ], krylovBasisSyBookkeepers[ irow ], krylovBasisVectors[ icol ], krylovBasisSyBookkeepers[ icol ] );
   //       krylovHamiltonian[ icol + irow * krylovSpaceDimension ] = std::conj( krylovHamiltonian[ irow + icol * krylovSpaceDimension ] );
   //    }
   // }


   // // commented out for clarity in the terminal output
   // for ( int irow = 0; irow < krylovSpaceDimension; irow++ ){
   //    for ( int icol = 0; icol < krylovSpaceDimension; icol++ ){
   //       std::cout << krylovHamiltonian[ irow +  icol * krylovSpaceDimension ] << " ";
   //    }
   //    std::cout << std::endl;
   // }
   // // commented out for clarity in the terminal output
   // for ( int irow = 0; irow < krylovSpaceDimension; irow++ ){
   //    for ( int icol = 0; icol < krylovSpaceDimension; icol++ ){
   //       std::cout << overlaps[ irow +  icol * krylovSpaceDimension ] << " ";
   //    }
   //    std::cout << std::endl;
   // }

   //////////////////////////////////////////////////////////////////////////////////////
   //
   // Calculate S^-0.5
   //
   //////////////////////////////////////////////////////////////////////////////////////

   dcomplex* Oinvsqr = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   {
      char jobz = 'V';
      char uplo = 'U';
      int inc = 1;
      int N = krylovSpaceDimension * krylovSpaceDimension;
      int lwork = 2 * N - 1;
      int info;

      dcomplex * U = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
      dcomplex* work = new dcomplex[ lwork ];
      double* evals = new double[ krylovSpaceDimension ];
      double* rwork = new double[ 3 * N - 2 ];

      zcopy_( &N, overlaps, &inc, U, &inc );
      zheev_( &jobz, &uplo, &krylovSpaceDimension, U, &krylovSpaceDimension, evals, work, &lwork, rwork, &info );
      assert( info == 0 );

      dcomplex one = 1.0;
      dcomplex zero = 0.0;
      dcomplex* tmp = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];
      dcomplex* diag = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

      for( int i = 0; i < krylovSpaceDimension; i++ ){
         for(int j = 0; j < krylovSpaceDimension; j++) {
            diag[ i + j * krylovSpaceDimension ] = 0;   
         }
         diag[ i + i * krylovSpaceDimension ] = std::pow( evals[ i ], -0.5 );
      }

      delete[] work;
      delete[] evals;
      delete[] rwork;

      zgemm_( &notrans, &notrans, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension, &one, U, &krylovSpaceDimension, diag, &krylovSpaceDimension, &zero, tmp, &krylovSpaceDimension );
      zgemm_( &notrans, &trans, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension, &one, tmp, &krylovSpaceDimension, U, &krylovSpaceDimension, &zero, Oinvsqr, &krylovSpaceDimension );

      delete[] tmp;
      delete[] diag;
      delete[] U;
   }

   //////////////////////////////////////////////////////////////////////////////////////
   //
   // Calculate S^-0.5 H S^-0.5
   //
   //////////////////////////////////////////////////////////////////////////////////////

   dcomplex* Hortho = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   {
      dcomplex one = 1.0;
      dcomplex zero = 0.0;

      dcomplex* tmp3 = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];   

      zgemm_( &notrans, &notrans, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension, &one, Oinvsqr, &krylovSpaceDimension, krylovHamiltonian, &krylovSpaceDimension, &zero, tmp3, &krylovSpaceDimension );
      zgemm_( &notrans, &notrans, &krylovSpaceDimension, &krylovSpaceDimension, &krylovSpaceDimension, &step, tmp3, &krylovSpaceDimension, Oinvsqr, &krylovSpaceDimension, &zero, Hortho, &krylovSpaceDimension );

      delete[] tmp3;
   }

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Calculate the matrix exponential
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   dcomplex * theExp = new dcomplex[ krylovSpaceDimension * krylovSpaceDimension ];

   {
      int deg        = 10;
      double bla     = 1.0;
      int lwsp       = 4 * krylovSpaceDimension * krylovSpaceDimension + deg + 1;
      dcomplex * wsp = new dcomplex[ lwsp ];
      int * ipiv     = new int[ krylovSpaceDimension ];
      int iexph      = 0;
      int ns         = 0;
      int info;
      zgpadm_( &deg, &krylovSpaceDimension, &bla, Hortho, &krylovSpaceDimension,
               wsp, &lwsp, ipiv, &iexph, &ns, &info );
      assert( info == 0 );

      int inc = 1;
      int dim = krylovSpaceDimension * krylovSpaceDimension;
      zcopy_( &dim, &wsp[ iexph - 1 ], &inc, theExp, &inc );

      delete[] ipiv; 
      delete[] wsp;
   }

   ////////////////////////////////////////////////////////////////////////////////////////
   ////
   //// Test new coefficients and sum MPS
   ////
   ////////////////////////////////////////////////////////////////////////////////////////

   dcomplex * result = new dcomplex[ krylovSpaceDimension ];
   for ( int beta = 0; beta < krylovSpaceDimension; beta++ ) {
      result[ beta ] = 0.0;
      for( int i = 0; i < krylovSpaceDimension; i++ ){
         for( int j = 0; j < krylovSpaceDimension; j++ ){
            for( int a = 0; a < krylovSpaceDimension; a++ ){
               result[ beta ] += theExp[ i + krylovSpaceDimension * j ] * std::conj( Oinvsqr[ a + krylovSpaceDimension * j ] ) * Oinvsqr[ beta + krylovSpaceDimension * i ] * overlaps[ a + krylovSpaceDimension * 0 ];
            }
         }
      }
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
   delete[] theExp;
   delete[] overlaps;
   delete[] Oinvsqr;
   delete[] krylovHamiltonian;
   delete[] Hortho;

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
                                        const bool backwards, const double offset,
                                        const bool do_ortho, const bool doDumpFCI, 
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
      HDF5_MAKE_DATASET( dataPointID, "ReOInit",        1, &dimarray1, H5T_NATIVE_DOUBLE,  &reoInit         );
      HDF5_MAKE_DATASET( dataPointID, "ImOInit",        1, &dimarray1, H5T_NATIVE_DOUBLE,  &imoInit         );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_REAL",      2, Lsq,        H5T_NATIVE_DOUBLE,  oedmre           );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_IMAG",      2, Lsq,        H5T_NATIVE_DOUBLE,  oedmim           );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_DMRG_REAL", 2, Lsq,        H5T_NATIVE_DOUBLE,  oedmdmrgre       );
      HDF5_MAKE_DATASET( dataPointID, "OEDM_DMRG_IMAG", 2, Lsq,        H5T_NATIVE_DOUBLE,  oedmdmrgim       );

      delete[] actdims;
      delete[] MaxMs;
      delete[] CutOs;
      delete[] NSwes;
      delete[] oedmre;
      delete[] oedmim;
      delete[] oedmdmrgre;
      delete[] oedmdmrgim;

      delete theodm;

      if ( nWeights > 0 ){
         int deltaN  = 0;
         int nElecHF = 0; for ( int index = 0; index < prob->gL(); index++ ) { nElecHF += hfState[ index ]; }
         deltaN      = nElecHF - prob->gN();

         int*     nHoles =    new int[ nWeights ];
         int* nParticles =    new int[ nWeights ];
         double* weights = new double[ nWeights ];

         for( int iWeight = 0; iWeight < nWeights; iWeight++ ){
            nHoles[ iWeight ]     = iWeight + deltaN;
            nParticles[ iWeight ] = iWeight;
            weights[ iWeight ]    =  calcWieght( iWeight + deltaN, iWeight, prob, MPS, MPSBK, hfState );
         }

         std::cout << "  The lowest " << nWeights << " CI weights are:\n";
         for( int iWeight = 0; iWeight < nWeights; iWeight++ ){
            std::cout << "  " << nHoles[ iWeight ] <<  "h" << nParticles[ iWeight] << "p-weight  = " << weights[ iWeight ] << "\n";
         }
         std::cout                                                 << "\n";

         HDF5_MAKE_DATASET( dataPointID, "nHoles",     1, &weightSze, H5T_STD_I32LE,      nHoles     );
         HDF5_MAKE_DATASET( dataPointID, "nParticles", 1, &weightSze, H5T_STD_I32LE,      nParticles );
         HDF5_MAKE_DATASET( dataPointID, "weights",    1, &weightSze, H5T_NATIVE_DOUBLE,  weights    );

         delete[] nHoles;
         delete[] nParticles;
         delete[] weights;
      }

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

      if ( t + time_step_major < time_final ) {
         for( double t_minor = 0.0; (time_step_major - t_minor) > 1e-6; t_minor+=time_step_minor ) {

            SyBookkeeper * MPSBKDT = new SyBookkeeper( *MPSBK );
            CTensorT ** MPSDT      = new CTensorT *[ L ];
            for ( int index = 0; index < L; index++ ) {
               MPSDT[ index ] = new CTensorT( index, MPSBKDT );
               MPSDT[ index ]->random();
            }
            normalize( L, MPSDT );

            if( time_type == 'K' ){
               doStep_arnoldi( time_step_minor, kry_size, offset, backwards, do_ortho, MPS, MPSBK, MPSDT, MPSBKDT );
            } else if ( time_type == 'R' ){
               doStep_runge_kutta( time_step_minor, kry_size, offset, backwards, MPS, MPSBK, MPSDT, MPSBKDT );
            } else if ( time_type == 'E' ){
               doStep_euler( time_step_minor, kry_size, offset, backwards, MPS, MPSBK, MPSDT, MPSBKDT );
            }

            
            if ( prob->getApplyPulse()) {
               updateHamiltonian( t, time_step_minor );
            }
            
            //TODO: updateHamiltonian(amplitude, frequency, time) an dieser Stelle. Evtl: genauer schauen bei RK4: dort zwischen Hamiltonians

            for ( int site = 0; site < L; site++ ) {
               delete MPS[ site ];
            }
            delete[] MPS;
            delete MPSBK;

            MPS   = MPSDT;
            MPSBK = MPSBKDT;
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

void CheMPS2::TimeEvolution::updateHamiltonian( const double currentTime, const double timeStep ) {
   
   double nextTime = currentTime + timeStep;
   std::cout << "\nUpdate Hamiltonian\n"
             << " ( t = " << currentTime << " ) --> ( t = " << nextTime << " )\n" << std::endl;
   
   prob->construct_mxelem( nextTime );
}