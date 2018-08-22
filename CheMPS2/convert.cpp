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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <assert.h>
#include <sstream>

#include "Initialize.h"
#include "COneDM.h"
#include "CTwoDMBuilder.h"
#include "CTwoDM.h"
#include "CASSCF.h"
#include "Molden.h"
#include "MPIchemps2.h"
#include "EdmistonRuedenberg.h"
#include "TimeEvolution.h"
#include "CFCI.h"
#include "CTensorT.h"
#include "TensorT.h"
#include "CSobject.h"
#include "Lapack.h"
#include "HamiltonianOperator.h"

#include "Irreps.h"

using namespace std;


void saveMPS(const std::string name, CheMPS2::CTensorT ** MPSlocation, CheMPS2::SyBookkeeper * BKlocation) {
 
   //The hdf5 file
   hid_t file_id = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   //The current virtual dimensions
   for( int bound = 0; bound <= BKlocation->gL(); bound++ ){
      for( int N = BKlocation->gNmin( bound ); N <= BKlocation->gNmax( bound ); N++ ){
         for( int TwoS = BKlocation->gTwoSmin( bound, N ); TwoS <= BKlocation->gTwoSmax( bound, N ); TwoS += 2 ){
            for( int Irrep = 0; Irrep < BKlocation->getNumberOfIrreps(); Irrep++ ){
   
               std::stringstream sstream;
               sstream << "/VirtDim_" << bound << "_" << N << "_" << TwoS << "_" << Irrep;
               hid_t group_id2 = H5Gcreate( file_id, sstream.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
               
               hsize_t dimarray2     = 1; //One integer
               hid_t dataspace_id2   = H5Screate_simple( 1, &dimarray2, NULL );
               hid_t dataset_id2     = H5Dcreate( group_id2, "Value", H5T_STD_I32LE, dataspace_id2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
               int toWrite2 = BKlocation->gCurrentDim( bound, N, TwoS, Irrep );
               H5Dwrite( dataset_id2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &toWrite2 );
            
               H5Dclose( dataset_id2 );
               H5Sclose( dataspace_id2 );

               H5Gclose( group_id2 );
               
            }
         }
      }
   }
      
   //The MPS
   for (int site = 0; site < BKlocation->gL(); site++ ){
      
      std::stringstream sstream;
      sstream << "/MPS_" << site;
      hid_t group_id3 = H5Gcreate( file_id, sstream.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
      
      hsize_t dimarray3     = 2 * MPSlocation[ site ]->gKappa2index( MPSlocation[ site ]->gNKappa() ); //An array of doubles
      hid_t dataspace_id3   = H5Screate_simple( 1, &dimarray3, NULL );
      hid_t dataset_id3     = H5Dcreate( group_id3, "Values", H5T_IEEE_F64LE, dataspace_id3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
      H5Dwrite( dataset_id3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, MPSlocation[ site ]->gStorage() );
   
      H5Dclose( dataset_id3 );
      H5Sclose( dataspace_id3 );

      H5Gclose( group_id3 );
      
   }
      
   H5Fclose( file_id );

}

void loadDIM(const std::string name, CheMPS2::SyBookkeeper * BKlocation){

   //The hdf5 file
   hid_t file_id = H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      
      //The current virtual dimensions
      for (int bound=0; bound<=BKlocation->gL(); bound++){
         for (int N=BKlocation->gNmin(bound); N<=BKlocation->gNmax(bound); N++){
            for (int TwoS=BKlocation->gTwoSmin(bound,N); TwoS<=BKlocation->gTwoSmax(bound,N); TwoS+=2){
               for (int Irrep=0; Irrep<BKlocation->getNumberOfIrreps(); Irrep++){
     
                  std::stringstream sstream;
                  sstream << "/VirtDim_" << bound << "_" << N << "_" << TwoS << "_" << Irrep;
                  hid_t group_id2 = H5Gopen(file_id, sstream.str().c_str(), H5P_DEFAULT);
                  
                     hid_t dataset_id2 = H5Dopen(group_id2, "Value", H5P_DEFAULT);
                     int toRead;
                     H5Dread(dataset_id2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &toRead);
                     BKlocation->SetDim(bound, N, TwoS, Irrep, toRead);
                     H5Dclose(dataset_id2);
                  
                  H5Gclose(group_id2);
                  
               }
            }
         }
      }
      
   H5Fclose(file_id);

}

void loadMPS( const std::string name, const int L, CheMPS2::CTensorT ** MPSlocation ){

   //The hdf5 file
   hid_t file_id = H5Fopen( name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
      
   //The MPS
   for (int site = 0; site < L; site++){
      
      std::stringstream sstream;
      sstream << "/MPS_" << site;
      hid_t group_id3 = H5Gopen( file_id, sstream.str().c_str(), H5P_DEFAULT );
      
      hid_t dataset_id3     = H5Dopen( group_id3, "Values", H5P_DEFAULT );
      H5Dread( dataset_id3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, MPSlocation[ site ]->gStorage() );
      H5Dclose( dataset_id3 );

      H5Gclose( group_id3 );
      
   }
      
   H5Fclose( file_id );

}


void loadMPS( const std::string name, const int L, CheMPS2::TensorT ** MPSlocation ){

   //The hdf5 file
   hid_t file_id = H5Fopen( name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
      
   //The MPS
   for (int site = 0; site < L; site++){
      
      std::stringstream sstream;
      sstream << "/MPS_" << site;
      hid_t group_id3 = H5Gopen( file_id, sstream.str().c_str(), H5P_DEFAULT );
      
      hid_t dataset_id3     = H5Dopen( group_id3, "Values", H5P_DEFAULT );
      H5Dread( dataset_id3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, MPSlocation[ site ]->gStorage() );
      H5Dclose( dataset_id3 );

      H5Gclose( group_id3 );
      
   }
      
   H5Fclose( file_id );

}


void fetch_ints( const string rawdata, vector<int> & result, int num ){

   int pos  = 0;
   int pos2 = 0;
   for ( int no = 0; no < num; no++ ){
      pos2 = rawdata.find( ",", pos );
      if ( pos2 == string::npos ){ pos2 = rawdata.length(); }
      result.push_back( atoi( rawdata.substr( pos, pos2-pos ).c_str() ) );
      pos = pos2 + 1;
   }

}

void fetch_doubles( const string rawdata, double * result, const int num ){

   int pos  = 0;
   int pos2 = 0;
   for ( int no = 0; no < num; no++ ){
      pos2 = rawdata.find( ",", pos );
      if ( pos2 == string::npos ){ pos2 = rawdata.length(); }
      result[ no ] = atof( rawdata.substr( pos, pos2-pos ).c_str() );
      pos = pos2 + 1;
   }

}

bool file_exists( const string filename, const string tag ){

   #ifdef CHEMPS2_MPI_COMPILATION
      const bool am_i_master = ( CheMPS2::MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER );
   #else
      const bool am_i_master = true;
   #endif

   struct stat file_info;
   const bool on_disk = (( filename.length() > 0 ) && ( stat( filename.c_str(), &file_info ) == 0 ));
   if (( on_disk == false ) && ( am_i_master )){
      cerr << "Unable to retrieve file " << filename << "!" << endl;
      cerr << "Invalid option for " << tag << "!" << endl;
   }
   return on_disk;

}

bool find_integer( int * result, const string line, const string tag, const bool lower_bound, const int val_lower, const bool upper_bound, const int val_upper ){

   #ifdef CHEMPS2_MPI_COMPILATION
      const bool am_i_master = ( CheMPS2::MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER );
   #else
      const bool am_i_master = true;
   #endif

   if ( line.find( tag ) != string::npos ){

      const int pos = line.find( "=" ) + 1;
      result[ 0 ] = atoi( line.substr( pos, line.length() - pos ).c_str() );

      const bool lower_ok = (( lower_bound == false ) || ( result[ 0 ] >= val_lower ));
      const bool upper_ok = (( upper_bound == false ) || ( result[ 0 ] <= val_upper ));

      if (( lower_ok == false ) || ( upper_ok == false )){
         if ( am_i_master ){
            cerr << line << endl;
            cerr << "Invalid option for " << tag << "!" << endl;
         }
         return false;
      }
   }

   return true;

}

bool find_double( double * result, const string line, const string tag, const bool lower_bound, const double val_lower ){

   #ifdef CHEMPS2_MPI_COMPILATION
      const bool am_i_master = ( CheMPS2::MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER );
   #else
      const bool am_i_master = true;
   #endif

   if ( line.find( tag ) != string::npos ){

      const int pos = line.find( "=" ) + 1;
      result[ 0 ] = atof( line.substr( pos, line.length() - pos ).c_str() );

      const bool lower_ok = (( lower_bound == false ) || ( result[ 0 ] >= val_lower ));

      if ( lower_ok == false ){
         if ( am_i_master ){
            cerr << line << endl;
            cerr << "Invalid option for " << tag << "!" << endl;
         }
         return false;
      }
   }

   return true;

}

bool find_character( char * result, const string line, const string tag, char * options, const int num_options ){

   #ifdef CHEMPS2_MPI_COMPILATION
      const bool am_i_master = ( CheMPS2::MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER );
   #else
      const bool am_i_master = true;
   #endif

   if ( line.find( tag ) != string::npos ){

      const int pos = line.find( "=" ) + 1;
      string temp = line.substr( pos, line.length() - pos );
      temp.erase( std::remove( temp.begin(), temp.end(), ' ' ), temp.end() );
      result[ 0 ] = temp.c_str()[ 0 ];

      bool encountered = false;
      for ( int cnt = 0; cnt < num_options; cnt++ ){
         if ( options[ cnt ] == result[ 0 ] ){ encountered = true; }
      }

      if ( encountered == false ){
         if ( am_i_master ){
            cerr << line << endl;
            cerr << "Invalid option for " << tag << "!" << endl;
         }
         return false;
      }
   }

   return true;

}

bool find_string( string& result, const string line, const string tag, vector<string> options ){

   if ( line.find( tag ) != string::npos ){

      const int pos = line.find( "=" ) + 1;
      string temp = line.substr( pos, line.length() - pos );
      temp.erase( std::remove( temp.begin(), temp.end(), ' ' ), temp.end() );
      result = temp;

      bool encountered = false;
      for ( int cnt = 0; cnt < options.size(); cnt++ ){
         if ( options[ cnt ].compare( result ) == 0 ){ encountered = true; }
      }

      if ( encountered == false ){
         cerr << line << endl;
         cerr << "Invalid option for " << tag << "!" << endl;
         return false;
      }
   }

   return true;

}


bool find_boolean( bool * result, const string line, const string tag ){

   #ifdef CHEMPS2_MPI_COMPILATION
      const bool am_i_master = ( CheMPS2::MPIchemps2::mpi_rank() == MPI_CHEMPS2_MASTER );
   #else
      const bool am_i_master = true;
   #endif

   if ( line.find( tag ) != string::npos ){

      const int pos       = line.find( "=" ) + 1;
      const int pos_true  = line.substr( pos, line.length() - pos ).find( "TRUE" );
      const int pos_false = line.substr( pos, line.length() - pos ).find( "FALSE" );
      result[ 0 ] = ( pos_true != string::npos );

      if (( pos_true == string::npos ) && ( pos_false == string::npos )){
         if ( am_i_master ){
            cerr << line << endl;
            cerr << "Invalid option for " << tag << "!" << endl;
         }
         return false;
      }
   }

   return true;

}

void applyAnnihilator( const int pos, CheMPS2::CTensorT** mpsIn, CheMPS2::SyBookkeeper* bkIn, CheMPS2::CTensorT** mpsOut, CheMPS2::SyBookkeeper *bkOut, CheMPS2::Problem *probOut ){

   const int L = bkIn->gL();
   assert( pos < L );
   assert( ( L + 1 ) == bkOut->gL() );

   CheMPS2::SyBookkeeper* bkTemp = new CheMPS2::SyBookkeeper( *bkOut );

   for( int site = 0; site < L; site++ ){
      for ( int NL = bkIn->gNmin( site ); NL <= bkIn->gNmax( site ); NL++ ) {
         for ( int TwoSL = bkIn->gTwoSmin( site, NL ); TwoSL <= bkIn->gTwoSmax( site, NL ); TwoSL += 2 ) {
            for ( int IL = 0; IL < bkIn->getNumberOfIrreps(); IL++ ) {
               const int dimL = bkIn->gCurrentDim( site, NL, TwoSL, IL );
               bkTemp->SetDim( site, NL, TwoSL, IL, dimL );
               if ( dimL > 0 ) {
                  for ( int NR = NL; NR <= NL + 2; NR++ ) {
                     const int TwoJ = ( ( NR == NL + 1 ) ? 1 : 0 );
                     for ( int TwoSR = TwoSL - TwoJ; TwoSR <= TwoSL + TwoJ; TwoSR += 2 ) {
                        if ( TwoSR >= 0 ) {
                           int IR         = ( ( NR == NL + 1 ) ? CheMPS2::Irreps::directProd( IL, bkIn->gIrrep( site ) ) : IL );
                           const int dimR = bkIn->gCurrentDim( site + 1, NR, TwoSR, IR );
                           bkTemp->SetDim( site + 1, NR, TwoSR, IR, dimR );
                        }
                     }
                  }
               }
            }
         }
      }
   }

   CheMPS2::CTensorT ** mpsTemp = new CheMPS2::CTensorT *[ bkTemp->gL() ];
   for ( int index = 0; index < L + 1; index++ ) {
      mpsTemp[ index ] = new CheMPS2::CTensorT( index, bkTemp );
   }

  for( int site = 0; site < L; site++ ){
      for( int ikappa = 0; ikappa < mpsIn[ site ]->gNKappa(); ikappa++ ){
         const int NL    = mpsIn[ site ]->gNL( ikappa );
         const int TwoSL = mpsIn[ site ]->gTwoSL( ikappa );
         const int IL    = mpsIn[ site ]->gIL( ikappa );

         const int NR    = mpsIn[ site ]->gNR( ikappa );
         const int TwoSR = mpsIn[ site ]->gTwoSR( ikappa );
         const int IR    = mpsIn[ site ]->gIR( ikappa );

         int dimL = bkIn->gCurrentDim( site,     NL, TwoSL, IL );
         int dimR = bkIn->gCurrentDim( site + 1, NR, TwoSR, IR );

         int dimLxdimR = dimL * dimR;
         int inc = 1;
         zcopy_( &dimLxdimR, mpsIn[ site ]->gStorage( NL, TwoSL, IL, NR, TwoSR, IR ), &inc, mpsTemp[ site ]->gStorage( NL, TwoSL, IL, NR, TwoSR, IR ), &inc );
      }
   }
   assert( mpsTemp[ L ]->gNKappa() == 1 );
   mpsTemp[ L ]->gStorage( )[ 0 ] = 1.0;


   CheMPS2::ConvergenceScheme * opt_scheme = new CheMPS2::ConvergenceScheme( 1 );
   opt_scheme->set_instruction( 0, 100, 0.0, 5, 0.0 );

   CheMPS2::HamiltonianOperator * op = new CheMPS2::HamiltonianOperator( probOut, 0.0 );
   op->DSApply( mpsTemp, bkTemp, mpsOut, bkOut, opt_scheme );

   delete op;
   delete opt_scheme;

   for ( int site = 0; site < L + 1; site++ ) {
      delete mpsTemp[ site ];
   }
   delete[] mpsTemp;
   delete bkTemp;

}


void print_help(){

cout << "\n"
"CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum chemistry\n"
"Copyright (C) 2013-2017 Sebastian Wouters\n"
"\n"
"Usage: chemps2 [OPTIONS]\n"
"\n"

/**************************************************
* The following is copied directly from chemps2.1 *
**************************************************/

"   SYMMETRY\n"
"       Conventions for the symmetry group and irrep numbers (same as psi4):\n"
"\n"
"                        |  0    1    2    3    4    5    6    7\n"
"               ---------|-----------------------------------------\n"
"               0 : c1   |  A\n"
"               1 : ci   |  Ag   Au\n"
"               2 : c2   |  A    B\n"
"               3 : cs   |  Ap   App\n"
"               4 : d2   |  A    B1   B2   B3\n"
"               5 : c2v  |  A1   A2   B1   B2\n"
"               6 : c2h  |  Ag   Bg   Au   Bu\n"
"               7 : d2h  |  Ag   B1g  B2g  B3g  Au   B1u  B2u  B3u\n"
"\n"
"   ARGUMENTS\n"
"       -f, --file=inputfile\n"
"              Specify the input file.\n"
"\n"
"       -v, --version\n"
"              Print the version of chemps2.\n"
"\n"
"       -h, --help\n"
"              Display this help.\n"
" " << endl;

}

int main( int argc, char ** argv ){

   /************************
   *  Read in the options  *
   *************************/

   string  inputfile = "";
   string    fcifile = "";
   string outputfile = "";

   int group = -1;

   struct option long_options[] =
   {
      {"input",   required_argument, 0, 'i'},
      {"fci",     required_argument, 0, 'f'},
      {"output",  required_argument, 0, 'o'},
      {"group",   required_argument, 0, 'g'},
      {"version", no_argument,       0, 'v'},
      {"help",    no_argument,       0, 'h'},
      {0, 0, 0, 0}
   };

   int option_index = 0;
   int c;
   while (( c = getopt_long( argc, argv, "ifovh:", long_options, &option_index )) != -1 ){
      switch( c ){
         case 'h':
         case '?':
            print_help();
            return 0;
            break;
         case 'v':
            cout << "chemps2 version " << CHEMPS2_VERSION << endl;
            return 0;
            break;
         case 'i':
            inputfile = optarg;
            if ( file_exists( inputfile, "--input" ) == false ){ return -1; }
            break;
         case 'f':
            fcifile = optarg;
            if ( file_exists( fcifile,     "--fci" ) == false ){ return -1; }
            break;
         case 'o':
            outputfile = optarg;
            break;
         case 'g':
            group = atoi( optarg );
            break;
      }
   }

   CheMPS2::Irreps Symmhelper( group );
   const int num_irreps = Symmhelper.getNumberOfIrreps();

   int fcidump_norb    = -1;
   int fcidump_nelec   = -1;
   int fcidump_two_s   = -1;
   int fcidump_irrep   = -1;
   {
      ifstream thefcidump( fcifile.c_str() );
      string line;
      int pos, pos2;
      getline( thefcidump, line ); // &FCI NORB= X,NELEC= Y,MS2= Z,
      pos = line.find( "FCI" );
      if ( pos == string::npos ){
         cerr << "The file " << fcifile << " is not a fcidump file!" << endl; 
         return -1;
      }
      pos = line.find( "NORB"  ); pos = line.find( "=", pos ); pos2 = line.find( ",", pos );
      fcidump_norb = atoi( line.substr( pos+1, pos2-pos-1 ).c_str() );
      pos = line.find( "NELEC" ); pos = line.find( "=", pos ); pos2 = line.find( ",", pos );
      fcidump_nelec = atoi( line.substr( pos+1, pos2-pos-1 ).c_str() );
      pos = line.find( "MS2"   ); pos = line.find( "=", pos ); pos2 = line.find( ",", pos );
      fcidump_two_s = atoi( line.substr( pos+1, pos2-pos-1 ).c_str() );
      do { getline( thefcidump, line ); } while ( line.find( "ISYM" ) == string::npos );
      pos = line.find( "ISYM"  ); pos = line.find( "=", pos ); pos2 = line.find( ",", pos );
      const int molpro_wfn_irrep = atoi( line.substr( pos+1, pos2-pos-1 ).c_str() );
      thefcidump.close();

      int * psi2molpro = new int[ num_irreps ];
      Symmhelper.symm_psi2molpro( psi2molpro );
      for ( int cnt = 0; cnt < num_irreps; cnt++ ){
         if ( molpro_wfn_irrep == psi2molpro[ cnt ] ){ fcidump_irrep = cnt; }
      }
      if ( fcidump_irrep == -1 ){
         cerr << "Could not find the molpro wavefunction symmetry (ISYM) in the fcidump file!" << endl; 
         return -1;
      }
      delete [] psi2molpro;
   }

   CheMPS2::Initialize::Init();
   CheMPS2::Hamiltonian * ham = new CheMPS2::Hamiltonian( fcifile, group );
   CheMPS2::Problem * prob    = new CheMPS2::Problem( ham, fcidump_two_s, fcidump_nelec, fcidump_irrep );

   /********************
   *  Load the HF-MPS  *
   ********************/

   CheMPS2::SyBookkeeper * bkIn  = new CheMPS2::SyBookkeeper( prob, 1 );
   loadDIM( inputfile, bkIn );

   CheMPS2::TensorT  ** mpsIn    = new  CheMPS2::TensorT *[ fcidump_norb ];
   CheMPS2::CTensorT ** mpsOut   = new CheMPS2::CTensorT *[ fcidump_norb ];
   for ( int index = 0; index < fcidump_norb; index++ ) {
       mpsIn[ index ] =  new CheMPS2::TensorT( index, bkIn );
      mpsOut[ index ] = new CheMPS2::CTensorT( index, bkIn );
      mpsOut[ index ]->Clear();
   }
   loadMPS( inputfile, fcidump_norb, mpsIn );

   for( int site = 0; site < fcidump_norb; site++ ){
      assert( mpsIn[ site ]->gNKappa() == mpsOut[ site ]->gNKappa() );
      assert( mpsIn[ site ]->gKappa2index( mpsIn[ site ]->gNKappa() ) == mpsOut[ site ]->gKappa2index( mpsOut[ site ]->gNKappa() ) );

      for( int i = 0; i < mpsIn[ site ]->gKappa2index( mpsIn[ site ]->gNKappa() ); i++ ){
         mpsOut[ site ]->gStorage()[ i ] = mpsIn[ site ]->gStorage()[ i ];
      }
   }

   saveMPS( outputfile, mpsOut, bkIn );
   cout << "The converted state has been successfully stored to " << outputfile << " .\n";

   for ( int site = 0; site < fcidump_norb; site++ ) {
      delete mpsIn[ site ];
      delete mpsOut[ site ];
   }
   delete[] mpsIn;
   delete[] mpsOut;
   delete bkIn;

   delete prob;
   delete ham;

   return 0;

}