
#include "TimeTaylor.h"
#include <assert.h>
#include <iomanip>
#include <iostream>

#include "CDensityMatrix.h"
#include "CHeffNS.h"
#include "CHeffNS_1S.h"
#include "COneDM.h"
#include "CSobject.h"
#include "CTwoDMBuilder.h"
#include "HamiltonianOperator.h"
#include "Lapack.h"
#include "Special.h"
#include "TwoDMBuilder.h"

CheMPS2::TimeTaylor::TimeTaylor( Problem * probIn, ConvergenceScheme * schemeIn, hid_t HDF5FILEIDIN )
    : prob( probIn ), scheme( schemeIn ), HDF5FILEID( HDF5FILEIDIN ), L( probIn->gL() ) {
   assert( probIn->checkConsistency() );

   prob->construct_mxelem();

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

CheMPS2::TimeTaylor::~TimeTaylor() {

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

void CheMPS2::TimeTaylor::HDF5_MAKE_DATASET( hid_t setID, const char * name, int rank, const hsize_t * dims, hid_t typeID, const void * data ) {
   if ( HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ) {
      H5LTmake_dataset( setID, name, rank, dims, typeID, data );
   }
}

void CheMPS2::TimeTaylor::updateMovingLeftSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {
   if ( isAllocated[ cnt ] == 1 ) {
      deleteTensors( cnt, true );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, false, bkUp, bkDown );
      isAllocated[ cnt ] = 2;
   }
   updateMovingLeft( cnt, mpsUp, bkUp, mpsDown, bkDown );
}

void CheMPS2::TimeTaylor::updateMovingRightSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {
   if ( isAllocated[ cnt ] == 2 ) {
      deleteTensors( cnt, false );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, true, bkUp, bkDown );
      isAllocated[ cnt ] = 1;
   }
   updateMovingRight( cnt, mpsUp, bkUp, mpsDown, bkDown );
}

void CheMPS2::TimeTaylor::deleteAllBoundaryOperators() {
   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      if ( isAllocated[ cnt ] == 1 ) {
         deleteTensors( cnt, true );
      }
      if ( isAllocated[ cnt ] == 2 ) {
         deleteTensors( cnt, false );
      }
      isAllocated[ cnt ] = 0;
   }
}

void CheMPS2::TimeTaylor::updateMovingLeft( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {

   const int dimL = std::max( bkUp->gMaxDimAtBound( index + 1 ), bkDown->gMaxDimAtBound( index + 1 ) );
   const int dimR = std::max( bkUp->gMaxDimAtBound( index + 2 ), bkDown->gMaxDimAtBound( index + 2 ) );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];
// Ltensors_MPSDT_MPS : all processes own all Ltensors_MPSDT_MPS
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == L - 2 ) {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            LtensorsT[ index ][ cnt2 ]->update( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
         }
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      const int k1          = L - 1 - index;
      const int upperbound1 = ( k1 * ( k1 + 1 ) ) / 2;
      int result[ 2 ];
// After this parallel region, WAIT because F0,F1,S0,S1[ index ][ cnt2 ][ cnt3
// == 0 ] is required for the complementary operators
#pragma omp for schedule( static )
      for ( int global = 0; global < upperbound1; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2 = k1 - 1 - result[ 1 ];
         const int cnt3 = result[ 0 ];
         if ( cnt3 == 0 ) {
            if ( cnt2 == 0 ) {
               if ( index == L - 2 ) {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
               }
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F0tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F1tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S0tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S1tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      const int k2          = index + 1;
      const int upperbound2 = ( k2 * ( k2 + 1 ) ) / 2;
#pragma omp for schedule( static ) nowait
      for ( int global = 0; global < upperbound2; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2       = k2 - 1 - result[ 1 ];
         const int cnt3       = result[ 0 ];
         const int siteindex1 = index - cnt3 - cnt2;
         const int siteindex2 = index - cnt3;
         const int irrep_prod = Irreps::directProd( bkUp->gIrrep( siteindex1 ), bkUp->gIrrep( siteindex2 ) );
         if ( index == L - 2 ) {
            Atensors[ index ][ cnt2 ][ cnt3 ]->clear();
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->clear();
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->clear();
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            Dtensors[ index ][ cnt2 ][ cnt3 ]->clear();
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
         } else {
            Atensors[ index ][ cnt2 ][ cnt3 ]->update( Atensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->update( AtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->update( Btensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->update( BtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->update( Ctensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->update( CtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            Dtensors[ index ][ cnt2 ][ cnt3 ]->update( Dtensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->update( DtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
         }
         for ( int num = 0; num < L - index - 1; num++ ) {
            if ( irrep_prod ==
                 S0tensorsT[ index ][ num ][ 0 ]->get_irrep() ) { // Then the matrix elements are not 0 due to symm.
               double alpha = prob->gMxElement( siteindex1, siteindex2, index + 1, index + 1 + num );
               if ( ( cnt2 == 0 ) && ( num == 0 ) )
                  alpha *= 0.5;
               if ( ( cnt2 > 0 ) && ( num > 0 ) )
                  alpha += prob->gMxElement( siteindex1, siteindex2, index + 1 + num, index + 1 );
               Atensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensors[ index ][ num ][ 0 ] );
               AtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensorsT[ index ][ num ][ 0 ] );

               if ( ( num > 0 ) && ( cnt2 > 0 ) ) {
                  alpha = prob->gMxElement( siteindex1, siteindex2, index + 1, index + 1 + num ) -
                          prob->gMxElement( siteindex1, siteindex2, index + 1 + num, index + 1 );
                  Btensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensors[ index ][ num ][ 0 ] );
                  BtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensorsT[ index ][ num ][ 0 ] );
               }
               alpha = 2 * prob->gMxElement( siteindex1, index + 1, siteindex2, index + 1 + num ) -
                       prob->gMxElement( siteindex1, index + 1, index + 1 + num, siteindex2 );
               Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensors[ index ][ num ][ 0 ] );
               CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensorsT[ index ][ num ][ 0 ] );

               alpha = -prob->gMxElement( siteindex1, index + 1, index + 1 + num, siteindex2 ); // Second line for CtensorsT
               Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensors[ index ][ num ][ 0 ] );
               DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensorsT[ index ][ num ][ 0 ] );

               if ( num > 0 ) {
                  alpha = 2 * prob->gMxElement( siteindex1, index + 1 + num, siteindex2, index + 1 ) -
                          prob->gMxElement( siteindex1, index + 1 + num, index + 1, siteindex2 );
                  Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F0tensorsT[ index ][ num ][ 0 ] );
                  CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F0tensors[ index ][ num ][ 0 ] );

                  alpha = -prob->gMxElement( siteindex1, index + 1 + num, index + 1, siteindex2 ); // Second line for Ctensors_MPS_mpsUp
                  Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F1tensorsT[ index ][ num ][ 0 ] );
                  DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F1tensors[ index ][ num ][ 0 ] );
               }
            }
         }
      }
// QQtensors  : certain processes own certain QQtensors  --- You don't want to
// locally parallellize when sending and receiving buffers!
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( index == L - 2 ) {
            Qtensors[ index ][ cnt2 ]->clear();
            QtensorsT[ index ][ cnt2 ]->clear();
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
         } else {
            dcomplex * workmemBIS = new dcomplex[ dimR * dimR ];
            Qtensors[ index ][ cnt2 ]->update( Qtensors[ index + 1 ][ cnt2 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->update( QtensorsT[ index + 1 ][ cnt2 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsL( Ltensors[ index + 1 ], LtensorsT[ index + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsL( Ltensors[ index + 1 ], LtensorsT[ index + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsAB( Atensors[ index + 1 ][ cnt2 + 1 ][ 0 ], Btensors[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsAB( AtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], BtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsCD( Ctensors[ index + 1 ][ cnt2 + 1 ][ 0 ], Dtensors[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsCD( CtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], DtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            delete[] workmemBIS;
         }
      }

      delete[] workmem;
   }
   // Xtensors
   if ( index == L - 2 ) {
      Xtensors[ index ]->update( mpsUp[ index + 1 ], mpsDown[ index + 1 ] );
   } else {
      Xtensors[ index ]->update( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ],
                                 Ltensors[ index + 1 ], LtensorsT[ index + 1 ], Xtensors[ index + 1 ],
                                 Qtensors[ index + 1 ][ 0 ], QtensorsT[ index + 1 ][ 0 ],
                                 Atensors[ index + 1 ][ 0 ][ 0 ], AtensorsT[ index + 1 ][ 0 ][ 0 ],
                                 CtensorsT[ index + 1 ][ 0 ][ 0 ], DtensorsT[ index + 1 ][ 0 ][ 0 ] );
   }

   // Otensors
   if ( index == L - 2 ) {
      Otensors[ index ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ] );
   } else {
      Otensors[ index ]->update_ownmem( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ] );
   }
}

void CheMPS2::TimeTaylor::updateMovingRight( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {

   const int dimL = std::max( bkUp->gMaxDimAtBound( index ), bkDown->gMaxDimAtBound( index ) );
   const int dimR = std::max( bkUp->gMaxDimAtBound( index + 1 ), bkDown->gMaxDimAtBound( index + 1 ) );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];

// Ltensors
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == 0 ) {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            LtensorsT[ index ][ cnt2 ]->update( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
         }
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      const int k1          = index + 1;
      const int upperbound1 = ( k1 * ( k1 + 1 ) ) / 2;
      int result[ 2 ];
// After this parallel region, WAIT because F0,F1,S0,S1[ index ][ cnt2 ][ cnt3
// == 0 ] is required for the complementary operators
#pragma omp for schedule( static )
      for ( int global = 0; global < upperbound1; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2 = index - result[ 1 ];
         const int cnt3 = result[ 0 ];
         if ( cnt3 == 0 ) { // Every MPI process owns the Operator[ index ][ cnt2 ][
            // cnt3 == 0 ]
            if ( cnt2 == 0 ) {
               if ( index == 0 ) {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
               }
               // // S1[ index ][ 0 ][ cnt3 ] doesn't exist
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F0tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F1tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S0tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S1tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      const int k2          = L - 1 - index;
      const int upperbound2 = ( k2 * ( k2 + 1 ) ) / 2;
#pragma omp for schedule( static ) nowait
      for ( int global = 0; global < upperbound2; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2       = k2 - 1 - result[ 1 ];
         const int cnt3       = result[ 0 ];
         const int siteindex1 = index + 1 + cnt3;
         const int siteindex2 = index + 1 + cnt2 + cnt3;
         const int irrep_prod = CheMPS2::Irreps::directProd( bkUp->gIrrep( siteindex1 ), bkUp->gIrrep( siteindex2 ) );
         if ( index == 0 ) {
            Atensors[ index ][ cnt2 ][ cnt3 ]->clear();
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->clear();
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->clear();
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            Dtensors[ index ][ cnt2 ][ cnt3 ]->clear();
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
         } else {
            Atensors[ index ][ cnt2 ][ cnt3 ]->update( Atensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->update( AtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->update( Btensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->update( BtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->update( Ctensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->update( CtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            Dtensors[ index ][ cnt2 ][ cnt3 ]->update( Dtensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->update( DtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
         }

         for ( int num = 0; num < index + 1; num++ ) {
            if ( irrep_prod == S0tensorsT[ index ][ num ][ 0 ]->get_irrep() ) { // Then the matrix elements are not 0 due to symm.
               double alpha = prob->gMxElement( index - num, index, siteindex1, siteindex2 );
               if ( ( cnt2 == 0 ) && ( num == 0 ) ) {
                  alpha *= 0.5;
               }
               if ( ( cnt2 > 0 ) && ( num > 0 ) ) {
                  alpha += prob->gMxElement( index - num, index, siteindex2, siteindex1 );
               }
               Atensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensors[ index ][ num ][ 0 ] );
               AtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensorsT[ index ][ num ][ 0 ] );

               if ( ( num > 0 ) && ( cnt2 > 0 ) ) {
                  alpha =
                      prob->gMxElement( index - num, index, siteindex1, siteindex2 ) -
                      prob->gMxElement( index - num, index, siteindex2, siteindex1 );
                  Btensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensors[ index ][ num ][ 0 ] );
                  BtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensorsT[ index ][ num ][ 0 ] );
               }

               alpha = 2 * prob->gMxElement( index - num, siteindex1, index, siteindex2 ) -
                       prob->gMxElement( index - num, siteindex1, siteindex2, index );
               Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensors[ index ][ num ][ 0 ] );
               CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensorsT[ index ][ num ][ 0 ] );

               alpha = -prob->gMxElement( index - num, siteindex1, siteindex2, index ); // Second line for CtensorsT
               Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensors[ index ][ num ][ 0 ] );
               DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensorsT[ index ][ num ][ 0 ] );
               if ( num > 0 ) {
                  alpha = 2 * prob->gMxElement( index - num, siteindex2, index, siteindex1 ) -
                          prob->gMxElement( index - num, siteindex2, siteindex1, index );

                  Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F0tensorsT[ index ][ num ][ 0 ] );
                  CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F0tensors[ index ][ num ][ 0 ] );

                  alpha = -prob->gMxElement( index - num, siteindex2, siteindex1, index ); // Second line for CtensorsT
                  Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F1tensorsT[ index ][ num ][ 0 ] );
                  DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F1tensors[ index ][ num ][ 0 ] );
               }
            }
         }
      }

// QQtensors_mpsUp_MPS : certain processes own certain QQtensors_mpsUp_MPS ---
// You don't want to locally parallellize when sending and receiving buffers!
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( index == 0 ) {
            Qtensors[ index ][ cnt2 ]->clear();
            QtensorsT[ index ][ cnt2 ]->clear();
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
         } else {
            dcomplex * workmemBIS = new dcomplex[ dimL * dimL ];
            Qtensors[ index ][ cnt2 ]->update( Qtensors[ index - 1 ][ cnt2 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            QtensorsT[ index ][ cnt2 ]->update( QtensorsT[ index - 1 ][ cnt2 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsL( Ltensors[ index - 1 ], LtensorsT[ index - 1 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsL( Ltensors[ index - 1 ], LtensorsT[ index - 1 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsAB( Atensors[ index - 1 ][ cnt2 + 1 ][ 0 ], Btensors[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsAB( AtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], BtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsCD( Ctensors[ index - 1 ][ cnt2 + 1 ][ 0 ], Dtensors[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsCD( CtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], DtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            delete[] workmemBIS;
         }
      }

      delete[] workmem;
   }

   // Xtensors
   if ( index == 0 ) {
      Xtensors[ index ]->update( mpsUp[ index ], mpsDown[ index ] );
   } else {
      Xtensors[ index ]->update( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], Ltensors[ index - 1 ],
                                 LtensorsT[ index - 1 ], Xtensors[ index - 1 ], Qtensors[ index - 1 ][ 0 ],
                                 QtensorsT[ index - 1 ][ 0 ], Atensors[ index - 1 ][ 0 ][ 0 ],
                                 AtensorsT[ index - 1 ][ 0 ][ 0 ], CtensorsT[ index - 1 ][ 0 ][ 0 ],
                                 DtensorsT[ index - 1 ][ 0 ][ 0 ] );
   }

   // Otensors
   if ( index == 0 ) {
      Otensors[ index ]->create( mpsUp[ index ], mpsDown[ index ] );
   } else {
      Otensors[ index ]->update_ownmem( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ] );
   }
}

void CheMPS2::TimeTaylor::allocateTensors( const int index, const bool movingRight, SyBookkeeper * bkUp, SyBookkeeper * bkDown ) {

   if ( movingRight ) {
      // Ltensors
      Ltensors[ index ]  = new CTensorL *[ index + 1 ];
      LtensorsT[ index ] = new CTensorLT *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Ltensors[ index ][ cnt2 ]  = new CTensorL( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown );
         LtensorsT[ index ][ cnt2 ] = new CTensorLT( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown );
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      // To right: F0tens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt-cnt3-cnt2
      // and cnt-cnt3; at boundary cnt+1
      F0tensors[ index ]  = new CTensorF0 **[ index + 1 ];
      F0tensorsT[ index ] = new CTensorF0T **[ index + 1 ];
      F1tensors[ index ]  = new CTensorF1 **[ index + 1 ];
      F1tensorsT[ index ] = new CTensorF1T **[ index + 1 ];
      S0tensors[ index ]  = new CTensorS0 **[ index + 1 ];
      S0tensorsT[ index ] = new CTensorS0T **[ index + 1 ];
      S1tensors[ index ]  = new CTensorS1 **[ index + 1 ];
      S1tensorsT[ index ] = new CTensorS1T **[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         F0tensors[ index ][ cnt2 ]  = new CTensorF0 *[ index - cnt2 + 1 ];
         F0tensorsT[ index ][ cnt2 ] = new CTensorF0T *[ index - cnt2 + 1 ];
         F1tensors[ index ][ cnt2 ]  = new CTensorF1 *[ index - cnt2 + 1 ];
         F1tensorsT[ index ][ cnt2 ] = new CTensorF1T *[ index - cnt2 + 1 ];
         S0tensors[ index ][ cnt2 ]  = new CTensorS0 *[ index - cnt2 + 1 ];
         S0tensorsT[ index ][ cnt2 ] = new CTensorS0T *[ index - cnt2 + 1 ];
         if ( cnt2 > 0 ) {
            S1tensors[ index ][ cnt2 ] = new CTensorS1 *[ index - cnt2 + 1 ];
         }
         if ( cnt2 > 0 ) {
            S1tensorsT[ index ][ cnt2 ] = new CTensorS1T *[ index - cnt2 + 1 ];
         }
         for ( int cnt3 = 0; cnt3 < index - cnt2 + 1; cnt3++ ) {
            const int Iprod = CheMPS2::Irreps::directProd( bkUp->gIrrep( index - cnt2 - cnt3 ), bkUp->gIrrep( index - cnt3 ) );

            F0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF0( index + 1, Iprod, movingRight, bkUp, bkDown );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF1( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS0( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS1( index + 1, Iprod, movingRight, bkUp, bkDown );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      // To right: Atens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt+1+cnt3 and
      // cnt+1+cnt2+cnt3; at boundary cnt+1
      Atensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      AtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      Btensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      BtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      Ctensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      CtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      Dtensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      DtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Atensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         AtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         if ( cnt2 > 0 ) {
            Btensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
            BtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         }
         Ctensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         CtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         Dtensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         DtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Idiff = CheMPS2::Irreps::directProd( bkUp->gIrrep( index + 1 + cnt2 + cnt3 ), bkUp->gIrrep( index + 1 + cnt3 ) );

            Atensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 2, Idiff, movingRight, true, false, bkUp, bkDown );
            AtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 2, Idiff, movingRight, true, false, bkUp, bkDown );
               BtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, true, false, bkUp, bkDown );
            CtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, false, false, bkUp, bkDown );
            Dtensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, movingRight, false, bkUp, bkDown );
            DtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, !movingRight, false, bkUp, bkDown );
         }
      }

      // QQtensors_MPSDT_MPS
      // To right: Qtens[ cnt][ cnt2 ] = operator on site cnt+1+cnt2; at boundary cnt+1
      Qtensors[ index ]  = new CTensorQ *[ L - 1 - index ];
      QtensorsT[ index ] = new CTensorQT *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Qtensors[ index ][ cnt2 ]  = new CTensorQ( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown, prob, index + 1 + cnt2 );
         QtensorsT[ index ][ cnt2 ] = new CTensorQT( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown, prob, index + 1 + cnt2 );
      }

      // Xtensors : a certain process owns the Xtensors
      Xtensors[ index ] = new CTensorX( index + 1, movingRight, bkUp, bkDown, prob );

      // Otensors :
      Otensors[ index ] = new CTensorO( index + 1, movingRight, bkUp, bkDown );
   } else {
      Ltensors[ index ]  = new CTensorL *[ L - 1 - index ];
      LtensorsT[ index ] = new CTensorLT *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Ltensors[ index ][ cnt2 ]  = new CTensorL( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown );
         LtensorsT[ index ][ cnt2 ] = new CTensorLT( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown );
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      // To left: F0tens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt+1+cnt3 and
      // cnt+1+cnt3+cnt2; at boundary cnt+1
      F0tensors[ index ]  = new CTensorF0 **[ L - 1 - index ];
      F0tensorsT[ index ] = new CTensorF0T **[ L - 1 - index ];
      F1tensors[ index ]  = new CTensorF1 **[ L - 1 - index ];
      F1tensorsT[ index ] = new CTensorF1T **[ L - 1 - index ];
      S0tensors[ index ]  = new CTensorS0 **[ L - 1 - index ];
      S0tensorsT[ index ] = new CTensorS0T **[ L - 1 - index ];
      S1tensors[ index ]  = new CTensorS1 **[ L - 1 - index ];
      S1tensorsT[ index ] = new CTensorS1T **[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         F0tensors[ index ][ cnt2 ]  = new CTensorF0 *[ L - 1 - index - cnt2 ];
         F0tensorsT[ index ][ cnt2 ] = new CTensorF0T *[ L - 1 - index - cnt2 ];
         F1tensors[ index ][ cnt2 ]  = new CTensorF1 *[ L - 1 - index - cnt2 ];
         F1tensorsT[ index ][ cnt2 ] = new CTensorF1T *[ L - 1 - index - cnt2 ];
         S0tensors[ index ][ cnt2 ]  = new CTensorS0 *[ L - 1 - index - cnt2 ];
         S0tensorsT[ index ][ cnt2 ] = new CTensorS0T *[ L - 1 - index - cnt2 ];
         if ( cnt2 > 0 ) {
            S1tensors[ index ][ cnt2 ]  = new CTensorS1 *[ L - 1 - index - cnt2 ];
            S1tensorsT[ index ][ cnt2 ] = new CTensorS1T *[ L - 1 - index - cnt2 ];
         }
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Iprod = Irreps::directProd( bkUp->gIrrep( index + 1 + cnt3 ), bkUp->gIrrep( index + 1 + cnt2 + cnt3 ) );

            F0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF0( index + 1, Iprod, movingRight, bkUp, bkDown );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF1( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS0( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS1( index + 1, Iprod, movingRight, bkUp, bkDown );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      // To left: Atens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt-cnt2-cnt3
      // and cnt-cnt3; at boundary cnt+1
      Atensors[ index ]  = new CTensorOperator **[ index + 1 ];
      AtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      Btensors[ index ]  = new CTensorOperator **[ index + 1 ];
      BtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      Ctensors[ index ]  = new CTensorOperator **[ index + 1 ];
      CtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      Dtensors[ index ]  = new CTensorOperator **[ index + 1 ];
      DtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Atensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         AtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         if ( cnt2 > 0 ) {
            Btensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
            BtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         }
         Ctensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         CtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         Dtensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         DtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         for ( int cnt3 = 0; cnt3 < index + 1 - cnt2; cnt3++ ) {
            const int Idiff = CheMPS2::Irreps::directProd( bkUp->gIrrep( index - cnt2 - cnt3 ), bkUp->gIrrep( index - cnt3 ) );

            Atensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 2, Idiff, movingRight, true, false, bkUp, bkDown );
            AtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 2, Idiff, movingRight, true, false, bkUp, bkDown );
               BtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, true, false, bkUp, bkDown );
            CtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, false, false, bkUp, bkDown );
            Dtensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, movingRight, false, bkUp, bkDown );
            DtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, !movingRight, false, bkUp, bkDown );
         }
      }

      // QQtensors  : certain processes own certain QQtensors
      // To left: Qtens[ cnt][ cnt2 ] = operator on site cnt-cnt2; at boundary
      // cnt+1
      Qtensors[ index ]  = new CTensorQ *[ index + 1 ];
      QtensorsT[ index ] = new CTensorQT *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Qtensors[ index ][ cnt2 ]  = new CTensorQ( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown, prob, index - cnt2 );
         QtensorsT[ index ][ cnt2 ] = new CTensorQT( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown, prob, index - cnt2 );
      }

      // Xtensors : a certain process owns the Xtensors
      Xtensors[ index ] = new CTensorX( index + 1, movingRight, bkUp, bkDown, prob );

      // Otensors :
      Otensors[ index ] = new CTensorO( index + 1, movingRight, bkUp, bkDown );
   }
}

void CheMPS2::TimeTaylor::deleteTensors( const int index, const bool movingRight ) {
   const int Nbound = movingRight ? index + 1 : L - 1 - index;
   const int Cbound = movingRight ? L - 1 - index : index + 1;

   // Ltensors  : all processes own all Ltensors_MPSDT_MPS
   for ( int cnt2 = 0; cnt2 < Nbound; cnt2++ ) {
      delete Ltensors[ index ][ cnt2 ];
      delete LtensorsT[ index ][ cnt2 ];
   }
   delete[] Ltensors[ index ];
   delete[] LtensorsT[ index ];

   // Two-operator tensors : certain processes own certain two-operator tensors
   for ( int cnt2 = 0; cnt2 < Nbound; cnt2++ ) {
      for ( int cnt3 = 0; cnt3 < Nbound - cnt2; cnt3++ ) {
         delete F0tensors[ index ][ cnt2 ][ cnt3 ];
         delete F0tensorsT[ index ][ cnt2 ][ cnt3 ];
         delete F1tensors[ index ][ cnt2 ][ cnt3 ];
         delete F1tensorsT[ index ][ cnt2 ][ cnt3 ];
         delete S0tensors[ index ][ cnt2 ][ cnt3 ];
         delete S0tensorsT[ index ][ cnt2 ][ cnt3 ];
         if ( cnt2 > 0 ) {
            delete S1tensors[ index ][ cnt2 ][ cnt3 ];
            delete S1tensorsT[ index ][ cnt2 ][ cnt3 ];
         }
      }
      delete[] F0tensors[ index ][ cnt2 ];
      delete[] F0tensorsT[ index ][ cnt2 ];
      delete[] F1tensors[ index ][ cnt2 ];
      delete[] F1tensorsT[ index ][ cnt2 ];
      delete[] S0tensors[ index ][ cnt2 ];
      delete[] S0tensorsT[ index ][ cnt2 ];
      if ( cnt2 > 0 ) {
         delete[] S1tensors[ index ][ cnt2 ];
         delete[] S1tensorsT[ index ][ cnt2 ];
      }
   }
   delete[] F0tensors[ index ];
   delete[] F0tensorsT[ index ];
   delete[] F1tensors[ index ];
   delete[] F1tensorsT[ index ];
   delete[] S0tensors[ index ];
   delete[] S0tensorsT[ index ];
   delete[] S1tensors[ index ];
   delete[] S1tensorsT[ index ];

   // Complementary two-operator tensors : certain processes own certain complementary two-operator tensors
   for ( int cnt2 = 0; cnt2 < Cbound; cnt2++ ) {
      for ( int cnt3 = 0; cnt3 < Cbound - cnt2; cnt3++ ) {
         delete Atensors[ index ][ cnt2 ][ cnt3 ];
         delete AtensorsT[ index ][ cnt2 ][ cnt3 ];
         if ( cnt2 > 0 ) {
            delete Btensors[ index ][ cnt2 ][ cnt3 ];
            delete BtensorsT[ index ][ cnt2 ][ cnt3 ];
         }
         delete Ctensors[ index ][ cnt2 ][ cnt3 ];
         delete CtensorsT[ index ][ cnt2 ][ cnt3 ];
         delete Dtensors[ index ][ cnt2 ][ cnt3 ];
         delete DtensorsT[ index ][ cnt2 ][ cnt3 ];
      }
      delete[] Atensors[ index ][ cnt2 ];
      delete[] AtensorsT[ index ][ cnt2 ];
      if ( cnt2 > 0 ) {
         delete[] Btensors[ index ][ cnt2 ];
         delete[] BtensorsT[ index ][ cnt2 ];
      }
      delete[] Ctensors[ index ][ cnt2 ];
      delete[] CtensorsT[ index ][ cnt2 ];
      delete[] Dtensors[ index ][ cnt2 ];
      delete[] DtensorsT[ index ][ cnt2 ];
   }
   delete[] Atensors[ index ];
   delete[] AtensorsT[ index ];
   delete[] Btensors[ index ];
   delete[] BtensorsT[ index ];
   delete[] Ctensors[ index ];
   delete[] CtensorsT[ index ];
   delete[] Dtensors[ index ];
   delete[] DtensorsT[ index ];

   // QQtensors_MPSDT_MPS : certain processes own certain QQtensors_MPSDT_MPS
   for ( int cnt2 = 0; cnt2 < Cbound; cnt2++ ) {
      delete Qtensors[ index ][ cnt2 ];
      delete QtensorsT[ index ][ cnt2 ];
   }
   delete[] Qtensors[ index ];
   delete[] QtensorsT[ index ];

   // Xtensors
   delete Xtensors[ index ];

   // Otensors
   delete Otensors[ index ];
}

void CheMPS2::TimeTaylor::fitApplyH_1site( CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut, const int nSweeps ) {

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsOut, bkOut, mpsIn, bkIn );
   }

   for ( int i = 0; i < nSweeps; ++i ) {
      for ( int site = L - 1; site > 0; site-- ) {

         CHeffNS_1S * heff = new CHeffNS_1S( bkOut, bkIn, prob );
         heff->Apply( mpsIn[ site ], mpsOut[ site ],
                      Ltensors, LtensorsT,
                      Atensors, AtensorsT,
                      Btensors, BtensorsT,
                      Ctensors, CtensorsT,
                      Dtensors, DtensorsT,
                      S0tensors, S0tensorsT,
                      S1tensors, S1tensorsT,
                      F0tensors, F0tensorsT,
                      F1tensors, F1tensorsT,
                      Qtensors, QtensorsT,
                      Xtensors, Otensors, false );
         delete heff;
         right_normalize( mpsOut[ site - 1 ], mpsOut[ site ] );
         updateMovingLeftSafe( site - 1, mpsOut, bkOut, mpsIn, bkIn );
      }

      for ( int site = 0; site < L - 1; site++ ) {

         CHeffNS_1S * heff = new CHeffNS_1S( bkOut, bkIn, prob );
         heff->Apply( mpsIn[ site ], mpsOut[ site ],
                      Ltensors, LtensorsT,
                      Atensors, AtensorsT,
                      Btensors, BtensorsT,
                      Ctensors, CtensorsT,
                      Dtensors, DtensorsT,
                      S0tensors, S0tensorsT,
                      S1tensors, S1tensorsT,
                      F0tensors, F0tensorsT,
                      F1tensors, F1tensorsT,
                      Qtensors, QtensorsT,
                      Xtensors, Otensors, true );
         delete heff;

         left_normalize( mpsOut[ site ], mpsOut[ site + 1 ] );
         updateMovingRightSafe( site, mpsOut, bkOut, mpsIn, bkIn );
      }
   }
}

void CheMPS2::TimeTaylor::fitApplyH( dcomplex factor, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut, const int nSweeps, const int D, const double cut_off ) {

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsOut, bkOut, mpsIn, bkIn );
   }
   for ( int i = 0; i < nSweeps; ++i ) {
      for ( int site = L - 2; site > 0; site-- ) {

         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;

         CSobject * denSB = new CSobject( site, bkIn );
         denSB->Join( mpsIn[ site ], mpsIn[ site + 1 ] );

         CSobject * denPB = new CSobject( site, bkOut );

         CHeffNS * heff = new CHeffNS( bkOut, bkIn, prob, offset );
         heff->Apply( denSB, denPB, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         double disc = denPB->Split( mpsOut[ site ], mpsOut[ site + 1 ], D, cut_off, false, false );
         delete heff;
         delete denPB;
         delete denSB;

         updateMovingLeftSafe( site, mpsOut, bkOut, mpsIn, bkIn );
      }

      for ( int site = 0; site < L - 2; site++ ) {
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;

         CSobject * denSB = new CSobject( site, bkIn );
         denSB->Join( mpsIn[ site ], mpsIn[ site + 1 ] );

         CSobject * denPB = new CSobject( site, bkOut );

         CHeffNS * heff = new CHeffNS( bkOut, bkIn, prob, offset );
         heff->Apply( denSB, denPB, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         double disc = denPB->Split( mpsOut[ site ], mpsOut[ site + 1 ], D, cut_off, true, false );
         delete heff;
         delete denPB;
         delete denSB;

         updateMovingRightSafe( site, mpsOut, bkOut, mpsIn, bkIn );
      }
   }
}

void CheMPS2::TimeTaylor::fitAddMPS( dcomplex factor,
                                     CTensorT ** mpsA, SyBookkeeper * bkA,
                                     CTensorT ** mpsB, SyBookkeeper * bkB,
                                     CTensorT ** mpsOut, SyBookkeeper * bkOut,
                                     const int nSweeps, const int D, const double cut_off ) {

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   CTensorO ** OtensorsA = new CTensorO *[ L - 1 ];
   CTensorO ** OtensorsB = new CTensorO *[ L - 1 ];

   for ( int index = 0; index < L - 1; index++ ) {
      OtensorsA[ index ] = new CTensorO( index + 1, true, bkOut, bkA );
      OtensorsB[ index ] = new CTensorO( index + 1, true, bkOut, bkB );

      // Otensors
      if ( index == 0 ) {
         OtensorsA[ index ]->create( mpsOut[ index ], mpsA[ index ] );
         OtensorsB[ index ]->create( mpsOut[ index ], mpsB[ index ] );
      } else {
         OtensorsA[ index ]->update_ownmem( mpsOut[ index ], mpsA[ index ], OtensorsA[ index - 1 ] );
         OtensorsB[ index ]->update_ownmem( mpsOut[ index ], mpsB[ index ], OtensorsB[ index - 1 ] );
      }
   }

   for ( int i = 0; i < nSweeps; ++i ) {
      for ( int site = L - 2; site > 0; site-- ) {

         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? OtensorsA[ site - 1 ] : NULL;
         CTensorO * leftOverlapB  = ( site - 1 ) >= 0 ? OtensorsB[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? OtensorsA[ site + 1 ] : NULL;
         CTensorO * rightOverlapB = ( site + 2 ) < L ? OtensorsB[ site + 1 ] : NULL;

         CSobject * denSA = new CSobject( site, bkOut );
         denSA->Join( leftOverlapA, mpsA[ site ], mpsA[ site + 1 ], rightOverlapA );

         CSobject * denSB = new CSobject( site, bkOut );
         denSA->Join( leftOverlapB, mpsB[ site ], mpsB[ site + 1 ], rightOverlapB );

         denSA->Add( factor, denSB );

         double disc = denSA->Split( mpsOut[ site ], mpsOut[ site + 1 ], D, cut_off, false, true );

         delete denSA;
         delete denSB;

         delete OtensorsA[ site ];
         delete OtensorsB[ site ];
         OtensorsA[ site ] = new CTensorO( site + 1, false, bkOut, bkA );
         OtensorsB[ site ] = new CTensorO( site + 1, false, bkOut, bkB );

         // Otensors
         if ( site == L - 2 ) {
            OtensorsA[ site ]->create( mpsOut[ site + 1 ], mpsA[ site + 1 ] );
            OtensorsB[ site ]->create( mpsOut[ site + 1 ], mpsB[ site + 1 ] );
         } else {
            OtensorsA[ site ]->update_ownmem( mpsOut[ site + 1 ], mpsA[ site + 1 ], OtensorsA[ site + 1 ] );
            OtensorsB[ site ]->update_ownmem( mpsOut[ site + 1 ], mpsB[ site + 1 ], OtensorsB[ site + 1 ] );
         }
      }

      for ( int site = 0; site < L - 2; site++ ) {
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? OtensorsA[ site - 1 ] : NULL;
         CTensorO * leftOverlapB  = ( site - 1 ) >= 0 ? OtensorsB[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? OtensorsA[ site + 1 ] : NULL;
         CTensorO * rightOverlapB = ( site + 2 ) < L ? OtensorsB[ site + 1 ] : NULL;

         CSobject * denSA = new CSobject( site, bkOut );
         denSA->Join( leftOverlapA, mpsA[ site ], mpsA[ site + 1 ], rightOverlapA );

         CSobject * denSB = new CSobject( site, bkOut );
         denSB->Join( leftOverlapB, mpsB[ site ], mpsB[ site + 1 ], rightOverlapB );

         denSA->Add( factor, denSB );
         double disc = denSA->Split( mpsOut[ site ], mpsOut[ site + 1 ], D, cut_off, true, true );

         delete denSA;
         delete denSB;

         delete OtensorsA[ site ];
         delete OtensorsB[ site ];
         OtensorsA[ site ] = new CTensorO( site + 1, true, bkOut, bkA );
         OtensorsB[ site ] = new CTensorO( site + 1, true, bkOut, bkB );

         // Otensors
         if ( site == 0 ) {
            OtensorsA[ site ]->create( mpsOut[ site ], mpsA[ site ] );
            OtensorsB[ site ]->create( mpsOut[ site ], mpsB[ site ] );
         } else {
            OtensorsA[ site ]->update_ownmem( mpsOut[ site ], mpsA[ site ], OtensorsA[ site - 1 ] );
            OtensorsB[ site ]->update_ownmem( mpsOut[ site ], mpsB[ site ], OtensorsB[ site - 1 ] );
         }
      }
   }

   for ( int index = 0; index < L - 1; index++ ) {
      delete OtensorsA[ index ];
      delete OtensorsB[ index ];
   }
   delete[] OtensorsA;
   delete[] OtensorsB;
}

void CheMPS2::TimeTaylor::doStep_rk_4( const int currentInstruction, const bool doImaginary, const double offset ) {

   abort();
}

int CheMPS2::TimeTaylor::doStep_arnoldi( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   int krylovSpaceDimension = scheme->get_krylov_dimension( currentInstruction );

   dcomplex step = doImaginary ? -scheme->get_time_step( currentInstruction ) : dcomplex( 0.0, -1.0 * scheme->get_time_step( currentInstruction ) );

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
                         scheme->get_max_sweeps( currentInstruction ),
                         scheme->get_D( currentInstruction ),
                         scheme->get_cut_off( currentInstruction ) );

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
              scheme->get_max_sweeps( currentInstruction ),
              scheme->get_D( currentInstruction ),
              scheme->get_cut_off( currentInstruction ) );

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
   return krylovSpaceDimension;
}

void CheMPS2::TimeTaylor::doStep_krylov( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

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
}

void CheMPS2::TimeTaylor::doStep_euler_g( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   CheMPS2::SyBookkeeper * denHPsi = new CheMPS2::SyBookkeeper( prob, scheme->get_D( currentInstruction ) );
   CheMPS2::CTensorT ** HPsi       = new CheMPS2::CTensorT *[ prob->gL() ];
   for ( int index = 0; index < prob->gL(); index++ ) {
      HPsi[ index ] = new CheMPS2::CTensorT( index, denHPsi );
      HPsi[ index ]->random();
   }
   dcomplex step = doImaginary ? -scheme->get_time_step( currentInstruction ) : dcomplex( 0.0, -1.0 * -scheme->get_time_step( currentInstruction ) );

   // fitApplyH( step, offset, mpsIn, bkIn, HPsi, denHPsi );
   fitApplyH_1site( mpsIn, bkIn, HPsi, denHPsi, 5 );
   // fitAddMPS( step, mpsIn, bkIn, HPsi, denHPsi, mpsOut, bkOut );

   for ( int idx = 0; idx < prob->gL(); idx++ ) {
      delete HPsi[ idx ];
   }
   delete[] HPsi;
   delete denHPsi;
}

void CheMPS2::TimeTaylor::doStep_taylor_1site( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   dcomplex step = doImaginary ? -scheme->get_time_step( currentInstruction ) : dcomplex( 0.0, -1.0 * -scheme->get_time_step( currentInstruction ) );

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsOut, bkOut, mpsIn, bkIn );
   }

   for ( int i = 0; i < scheme->get_max_sweeps( currentInstruction ); ++i ) {
      for ( int site = L - 1; site > 0; site-- ) {

         CTensorT * linear        = new CTensorT( site, bkOut );
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 1 ) < L ? Otensors[ site ] : NULL;
         linear->Join( leftOverlapA, mpsIn[ site ], rightOverlapA );

         CTensorT * perturb = new CTensorT( mpsOut[ site ] );
         CHeffNS_1S * heff  = new CHeffNS_1S( bkOut, bkIn, prob );
         heff->Apply( mpsIn[ site ], perturb,
                      Ltensors, LtensorsT,
                      Atensors, AtensorsT,
                      Btensors, BtensorsT,
                      Ctensors, CtensorsT,
                      Dtensors, DtensorsT,
                      S0tensors, S0tensorsT,
                      S1tensors, S1tensorsT,
                      F0tensors, F0tensorsT,
                      F1tensors, F1tensorsT,
                      Qtensors, QtensorsT,
                      Xtensors, Otensors, false );
         delete heff;

         perturb->zaxpy( step, linear );
         linear->zcopy( mpsOut[ site ] );
         delete perturb;
         delete linear;

         right_normalize( mpsOut[ site - 1 ], mpsOut[ site ] );
         updateMovingLeftSafe( site - 1, mpsOut, bkOut, mpsIn, bkIn );
      }

      for ( int site = 0; site < L - 1; site++ ) {
         CTensorT * linear        = new CTensorT( site, bkOut );
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 1 ) < L ? Otensors[ site ] : NULL;
         linear->Join( leftOverlapA, mpsIn[ site ], rightOverlapA );

         CTensorT * perturb = new CTensorT( mpsOut[ site ] );
         CHeffNS_1S * heff  = new CHeffNS_1S( bkOut, bkIn, prob );
         heff->Apply( mpsIn[ site ], perturb,
                      Ltensors, LtensorsT,
                      Atensors, AtensorsT,
                      Btensors, BtensorsT,
                      Ctensors, CtensorsT,
                      Dtensors, DtensorsT,
                      S0tensors, S0tensorsT,
                      S1tensors, S1tensorsT,
                      F0tensors, F0tensorsT,
                      F1tensors, F1tensorsT,
                      Qtensors, QtensorsT,
                      Xtensors, Otensors, true );
         delete heff;

         perturb->zaxpy( step, linear );
         linear->zcopy( mpsOut[ site ] );
         delete perturb;
         delete linear;

         left_normalize( mpsOut[ site ], mpsOut[ site + 1 ] );
         updateMovingRightSafe( site, mpsOut, bkOut, mpsIn, bkIn );
      }
   }
}
void CheMPS2::TimeTaylor::doStep_taylor_1( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut ) {

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   const dcomplex step = doImaginary ? -scheme->get_time_step( currentInstruction ) : dcomplex( 0.0, -1.0 * -scheme->get_time_step( currentInstruction ) );
   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsOut, bkOut, mpsIn, bkIn );
   }

   for ( int i = 0; i < scheme->get_max_sweeps( currentInstruction ); ++i ) {
      for ( int site = L - 2; site > 0; site-- ) {
         CSobject * denSB = new CSobject( site, bkIn );
         denSB->Join( mpsIn[ site ], mpsIn[ site + 1 ] );

         CSobject * denPA         = new CSobject( site, bkOut );
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;
         denPA->Join( leftOverlapA, denSB, rightOverlapA );

         CSobject * denPB = new CSobject( site, bkOut );

         CHeffNS * heff = new CHeffNS( bkOut, bkIn, prob, offset );
         heff->Apply( denSB, denPB, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         denPA->Add( step, denPB );
         double disc = denPA->Split( mpsOut[ site ], mpsOut[ site + 1 ], scheme->get_D( currentInstruction ), scheme->get_cut_off( currentInstruction ), false, true );

         delete heff;
         delete denPA;
         delete denPB;
         delete denSB;

         updateMovingLeftSafe( site, mpsOut, bkOut, mpsIn, bkIn );
      }

      for ( int site = 0; site < L - 2; site++ ) {
         CSobject * denSB = new CSobject( site, bkIn );
         denSB->Join( mpsIn[ site ], mpsIn[ site + 1 ] );

         CSobject * denPA         = new CSobject( site, bkOut );
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;
         denPA->Join( leftOverlapA, denSB, rightOverlapA );

         CSobject * denPB = new CSobject( site, bkOut );

         CHeffNS * heff = new CHeffNS( bkOut, bkIn, prob, offset );
         heff->Apply( denSB, denPB, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         denPA->Add( step, denPB );
         double disc = denPA->Split( mpsOut[ site ], mpsOut[ site + 1 ], scheme->get_D( currentInstruction ), scheme->get_cut_off( currentInstruction ), true, true );

         delete heff;
         delete denPA;
         delete denPB;
         delete denSB;

         updateMovingRightSafe( site, mpsOut, bkOut, mpsIn, bkIn );
      }
   }
}

void CheMPS2::TimeTaylor::Propagate( SyBookkeeper * initBK, CTensorT ** initMPS, const bool doImaginary, const bool doDumpFCI ) {

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

   double t                 = 0.0;
   double firstEnergy       = 0;
   int krylovSpaceDimension = 0;

   for ( int inst = 0; inst < scheme->get_number(); inst++ ) {

      for ( ; t < scheme->get_max_time( inst ); t += scheme->get_time_step( inst ) ) {
         char dataPointname[ 1024 ];
         sprintf( dataPointname, "/Output/DataPoint%.5f", t );
         hid_t dataPointID = HDF5FILEID != H5_CHEMPS2_TIME_NO_H5OUT ? H5Gcreate( HDF5FILEID, dataPointname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) : H5_CHEMPS2_TIME_NO_H5OUT;

         std::cout << hashline;
         std::cout << "\n";
         std::cout << "   t = " << t << "\n";
         HDF5_MAKE_DATASET( dataPointID, "t", 1, &dimarray1, H5T_NATIVE_DOUBLE, &t );

         double Tmax = scheme->get_max_time( inst );
         std::cout << "   Tmax = " << Tmax << "\n";
         HDF5_MAKE_DATASET( dataPointID, "Tmax", 1, &dimarray1, H5T_NATIVE_DOUBLE, &Tmax );

         double dt = scheme->get_time_step( inst );
         std::cout << "   dt = " << dt << "\n";
         HDF5_MAKE_DATASET( dataPointID, "dt", 1, &dimarray1, H5T_NATIVE_DOUBLE, &dt );

         std::cout << "   KryS = " << krylovSpaceDimension << "\n";
         HDF5_MAKE_DATASET( dataPointID, "KryS", 1, &dimarray1, H5T_STD_I32LE, &krylovSpaceDimension );

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

         int MaxM = scheme->get_D( inst );
         std::cout << "   MaxM = " << MaxM << "\n";
         HDF5_MAKE_DATASET( dataPointID, "MaxM", 1, &dimarray1, H5T_STD_I32LE, &MaxM );

         double CutO = scheme->get_cut_off( inst );
         std::cout << "   CutO = " << CutO << "\n";
         HDF5_MAKE_DATASET( dataPointID, "CutO", 1, &dimarray1, H5T_NATIVE_DOUBLE, &CutO );

         int NSwe = scheme->get_max_sweeps( inst );
         std::cout << "   NSwe = " << NSwe << "\n";
         HDF5_MAKE_DATASET( dataPointID, "NSwe", 1, &dimarray1, H5T_STD_I32LE, &NSwe );

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

         if ( t + scheme->get_time_step( inst ) < scheme->get_max_time( inst ) ) {
            SyBookkeeper * MPSBKDT = new SyBookkeeper( *MPSBK );
            CTensorT ** MPSDT      = new CTensorT *[ L ];
            for ( int index = 0; index < L; index++ ) {
               MPSDT[ index ] = new CTensorT( index, MPSBKDT );
               MPSDT[ index ]->random();
            }

            // doStep_krylov( inst, doImaginary, firstEnergy, MPS, MPSBK, MPSDT, MPSBKDT );
            krylovSpaceDimension = doStep_arnoldi( inst, doImaginary, firstEnergy, MPS, MPSBK, MPSDT, MPSBKDT );

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
   }

   for ( int site = 0; site < L; site++ ) {
      delete MPS[ site ];
   }
   delete[] MPS;
   delete MPSBK;
   delete hamOp;
}