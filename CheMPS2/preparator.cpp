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
#include "CTensorF.h"
#include "CTensorLF.h"
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
"\n"
"   INPUT FILE\n"
"       FCIDUMP = /path/to/fcidump\n"
"              Note that orbital irreps in the FCIDUMP file follow molpro convention!\n"
"\n"
"       GROUP = int\n"
"              Set the psi4 symmetry group number [0-7] which corresponds to the FCIDUMP file.\n"
"\n"
"       MULTIPLICITY = int\n"
"              Overwrite the spin multiplicity [2S+1] of the FCIDUMP file.\n"
"\n"
"       NELECTRONS = int\n"
"              Overwrite the number of electrons of the FCIDUMP file.\n"
"\n"
"       IRREP = int\n"
"              Overwrite the target wavefunction irrep [0-7] of the FCIDUMP file (psi4 convention).\n"
"\n"
"       SWEEP_STATES = int, int, int\n"
"              Set the number of reduced renormalized basis states for the successive sweep instructions (positive integers).\n"
"\n"
"       SWEEP_MAX_SWEEPS = int, int, int\n"
"              Set the maximum number of sweeps for the successive sweep instructions (positive integers).\n"
"\n"
"       SWEEP_NOISE_PREFAC = flt, flt, flt\n"
"              Set the noise prefactors for the successive sweep instructions (floats).\n"
"\n"
"       SWEEP_CUTOFF = flt, flt, flt\n"
"              Set the cut off parameter for the Krylov space generation (positive integers).\n"
"\n"
"       REORDER_FIEDLER = bool\n"
"              When all orbitals are active orbitals, switch on orbital reordering based on the Fiedler vector of the exchange matrix (TRUE or FALSE; default FALSE).\n"
"\n"
"       REORDER_ORDER = int, int, int, int\n"
"              When all orbitals are active orbitals, provide a custom orbital reordering (default unspecified). When specified, this option takes precedence over REORDER_ORDER.\n"
"\n"
"       TIME_TYPE = char\n"
"              Set the type of time evolution calculation to be performed. Options are (K) for Krylov (default), (R) for Runge-Kutta, (E) for Euler, and (F) for FCI.\n"
"\n"
"       TIME_STEP_MAJOR = flt\n"
"              Set the time step (DT) for wave function anlysis (positive float).\n"
"\n"
"       TIME_STEP_MINOR = flt\n"
"              Set the time step (DT) for the time evolution calculation (positive float).\n"
"\n"
"       TIME_FINAL = flt\n"
"              Set the final time for the time evolution calculation (positive float). \n"
"\n"
"       TIME_NINIT = int, int, int\n"
"              Set the occupation numbers for the inital state. Ordered as in the FCIDUMP file. (positive integers).\n"
"\n"
"       TIME_HF_STATE = int, int, int\n"
"              Set the occupation numbers for the corresponding Hartree-Fock state inital state.\n"
"\n"
"       TIME_N_WEIGHTS = int\n"
"              Set the numbers of CI-weights to be calculated (default 0).\n"
"\n"
"       TIME_KRYSIZE = int\n"
"              Set the maximum Krylov space dimension of a time propagation step. Required when TIME_EVOLU = TRUE (positive integer).\n"
"\n"
"       TIME_HDF5OUTPUT = /path/to/hdf5/destination\n"
"              Set the file path for the HDF5 output when specified (default unspecified).\n"
"\n"
"       TIME_DUMPFCI = bool\n"
"              Set if the FCI coefficients are dumped into the HDF5 file. Only has affect if TIME_EVOLU = TRUE and TIME_HDF5OUTPUT is specified (TRUE or FALSE; default FALSE).\n"
"\n"
"       TIME_DUMP2RDM = bool\n"
"              Set if the 2RDM is dumped into the HDF5 file. Only has affect if TIME_EVOLU = TRUE and TIME_HDF5OUTPUT is specified (TRUE or FALSE; default FALSE).\n"
"\n"
" " << endl;

}

int main( int argc, char ** argv ){

   /************************
   *  Read in the options  *
   *************************/

   string inputfile = "";
   string task = "";

   struct option long_options[] =
   {
      {"file",    required_argument, 0, 'f'},
      {"version", no_argument,       0, 'v'},
      {"help",    no_argument,       0, 'h'},
      {0, 0, 0, 0}
   };

   int option_index = 0;
   int c;
   while (( c = getopt_long( argc, argv, "hvf:", long_options, &option_index )) != -1 ){
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
         case 'f':
            inputfile = optarg;
            if ( file_exists( inputfile, "--file" ) == false ){ return -1; }
            break;
      }
   }

   if ( inputfile.length() == 0 ){
      cerr << "The input file should be specified!" << endl;
      return -1;
   }


   ifstream input( inputfile.c_str() );
   string line;
   while ( input.eof() == false ){

      getline( input, line );

      vector<string> taskoptions = { "HF_STATE", "IONIZE" };
      if ( !find_string( task, line, "PREP_TASK", taskoptions ) ){ return -1; }
   }

   input.clear();
   input.seekg(0, ios::beg);

   cout << "\nRunning chemps2prp version " << CHEMPS2_VERSION << "\n" << endl;


   /**********************
   *  Hartree-Fock State *
   ***********************/

   if( task.compare( "HF_STATE" ) == 0 ){

      /************************
      *  Declaring variables  *
      ************************/

      int norbitals          = -1; 
      int group              = -1;
      int multiplicity       = -1;
      int nelectrons         = -1;
      int targetirrep        = -1;
      string irreps          = "";
      string hf_occupation   = "";
      string output_file     = "";
      vector<int> irreps_parsed;
      vector<int> hf_occupation_parsed;

      /***************
      *  Parse file  *
      ***************/

      while ( input.eof() == false ){
         getline( input, line );

         if ( line.find( "IRREPS" ) != string::npos ){
            const int pos = line.find( "=" ) + 1;
            irreps = line.substr( pos, line.length() - pos );
         }

         if ( line.find( "HF_OCCUPATION" ) != string::npos ){
            const int pos = line.find( "=" ) + 1;
            hf_occupation = line.substr( pos, line.length() - pos );
         }

         if ( line.find( "OUTPUT_FILE" ) != string::npos ){
            const int pos   = line.find( "=" ) + 1;
            output_file = line.substr( pos, line.length() - pos );
            output_file.erase( remove( output_file.begin(), output_file.end(), ' ' ), output_file.end() );
         }

         if ( find_integer( &norbitals,    line, "ORBITALS",     true, 1, false, -1 ) == false ){ return -1; }
         if ( find_integer( &group,        line, "GROUP",        true, 0, true,   7 ) == false ){ return -1; }
         if ( find_integer( &multiplicity, line, "MULTIPLICITY", true, 1, false, -1 ) == false ){ return -1; }
         if ( find_integer( &nelectrons,   line, "NELECTRONS",   true, 2, false, -1 ) == false ){ return -1; }
         if ( find_integer( &targetirrep,  line, "TARGETIRREP",  true, 0, true,   7 ) == false ){ return -1; }

      }

      /*******************************
      *  Check the parsed variables  *
      *******************************/

      if( norbitals < 1 )                              { cerr << " The number of ORBITALS must be given and > 1 .\n";                    return -1; }
      if( ( group < 0 ) || ( group > 7 ) )             { cerr << " The molecular GROUP number of must be given and between 0 and 7 .\n"; return -1; }
      if( ( multiplicity < 0 ) )                       { cerr << " The multiplicity must be given and >= 0 .\n";                         return -1; }
      if( ( nelectrons < 0 ) )                         { cerr << " The number of electrons must be given and >1 0 .\n";                  return -1; }
      if( ( targetirrep < 0 ) || ( targetirrep > 7 ) ) { cerr << " The target IRREP must be given and between 0 and 7 .\n";              return -1; }
      if( output_file.length() == 0 )                  { cerr << " The OUTPUT_FILE must be given .\n";                                   return -1; }


      CheMPS2::Irreps Symmhelper( group );
      const int num_irreps = Symmhelper.getNumberOfIrreps();

      const int irreps_n  = count( irreps.begin(), irreps.end(), ',' ) + 1;
      const bool irreps_ok = ( norbitals == irreps_n );

      if ( !irreps_ok ){
         cerr << "There should be " << norbitals << " numbers in IRREPS !" << endl;
         return -1;
      }

      fetch_ints( irreps, irreps_parsed, norbitals );

      for ( int cnt = 0; cnt < norbitals; cnt ++ ){
         if ( ( irreps_parsed[ cnt ] < 1 ) || ( irreps_parsed[ cnt ] > num_irreps ) ){
            cerr << "The irrep number in IRREPS has to be between 1 and " << num_irreps << " !" << endl;
            return -1;
         }
      }

      const int hf_occupation_n  = count( hf_occupation.begin(), hf_occupation.end(), ',' ) + 1;
      const bool hf_occupation_ok = ( norbitals == hf_occupation_n );

      if ( !hf_occupation_ok ){
         cerr << "There should be " << norbitals << " numbers in HF_OCCUPATION !" << endl;
         return -1;
      }

      fetch_ints( hf_occupation, hf_occupation_parsed, norbitals );

      for ( int cnt = 0; cnt < norbitals; cnt ++ ){
         if ( ( hf_occupation_parsed[ cnt ] < 0 ) || ( hf_occupation_parsed[ cnt ] > 2 ) ){
            cerr << "The irrep number in HF_OCCUPATION has to be between 0 and 2 !" << endl;
            return -1;
         }
      }

      int elec_sum = 0; for ( int cnt = 0; cnt < norbitals; cnt++ ) { elec_sum += hf_occupation_parsed[ cnt ];  }
      if ( elec_sum != nelectrons ){
         cerr << "There should be " << nelectrons << " distributed over the molecular orbitals in TIME_NINIT !" << endl;
         return -1;
      }

      /********************************
      *  Output the parsed varaibles  *
      ********************************/

      cout << "   Preparing a Hartree-Fock ^ with the following options:\n";
      cout << "   GROUP              = " << Symmhelper.getGroupName() << endl;
      cout << "   MULTIPLICITY       = " << multiplicity << endl;
      cout << "   NELECTRONS         = " << nelectrons << endl;
      cout << "   IRREP              = " << Symmhelper.getIrrepName( targetirrep ) << endl;
      cout << "   IRREPS             = [ " << irreps_parsed[ 0 ]; for ( int cnt = 1; cnt < norbitals; cnt++ ){ cout << " ; " << irreps_parsed[ cnt ]; } cout << " ]" << endl;
      cout << "   HF_OCCUPATION      = [ " << hf_occupation_parsed[ 0 ]; for ( int cnt = 1; cnt < norbitals; cnt++ ){ cout << " ; " << hf_occupation_parsed[ cnt ]; } cout << " ]" << endl;
      cout << "   OUTPUT_FILE        = " << output_file << endl;
      cout << "\n";
      /*****************************
      *  Create necessary objects  *
      *****************************/

      CheMPS2::Initialize::Init();
      CheMPS2::Hamiltonian ham( norbitals, group, &irreps_parsed[ 0 ] );
      CheMPS2::Problem prob( &ham, multiplicity - 1, nelectrons, targetirrep );

      /*******************
      *  Create the MPS  *
      *******************/

      CheMPS2::SyBookkeeper * MPSBK  = new CheMPS2::SyBookkeeper( &prob, &hf_occupation_parsed[ 0 ] );
      CheMPS2::CTensorT    ** MPS    = new CheMPS2::CTensorT *[ norbitals ];

      for ( int index = 0; index < norbitals; index++ ) {
         MPS[ index ] = new CheMPS2::CTensorT( index, MPSBK );
         assert( MPS[ index ]->gNKappa() == 1 );
         MPS[ index ]->gStorage()[ 0 ] = 1.0;
      }
      normalize( norbitals, MPS );

      /*******************************
      *  Store the MPS to HDF5 file  *
      *******************************/

      saveMPS( output_file, MPS, MPSBK );
      cout << "   The Hartree-Fock state has been successfully stored to " << output_file << " .\n";

      /*******************************
      *  Delete the allocated stuff  *
      *******************************/

      for ( int site = 0; site < norbitals; site++ ) {
         delete MPS[ site ];
      }
      delete[] MPS;
      delete MPSBK;

   }

   input.clear();
   input.seekg(0, ios::beg);

   /*******************************
   *  Excite electron to contimum *
   *******************************/

   if( task.compare( "IONIZE" ) == 0 ){

      /************************
      *  Declaring variables  *
      ************************/

      int norbitals          = -1; 
      int group              = -1;
      int multiplicity       = -1;
      int nelectrons         = -1;
      int targetirrep        = -1;
      int ionized_orbital    = -1;
      string irreps          = "";
      string input_file      = "";
      string output_file     = "";
      vector<int> irreps_parsed;

      /***************
      *  Parse file  *
      ***************/

      while ( input.eof() == false ){
         getline( input, line );

         if ( line.find( "IRREPS" ) != string::npos ){
            const int pos = line.find( "=" ) + 1;
            irreps = line.substr( pos, line.length() - pos );
         }
         
         if ( line.find( "INPUT_FILE" ) != string::npos ){
            const int pos   = line.find( "=" ) + 1;
            input_file = line.substr( pos, line.length() - pos );
            input_file.erase( remove( input_file.begin(), input_file.end(), ' ' ), input_file.end() );
         }

         if ( line.find( "OUTPUT_FILE" ) != string::npos ){
            const int pos   = line.find( "=" ) + 1;
            output_file = line.substr( pos, line.length() - pos );
            output_file.erase( remove( output_file.begin(), output_file.end(), ' ' ), output_file.end() );
         }

         if ( find_integer( &norbitals,        line, "ORBITALS",         true, 1, false, -1 ) == false ){ return -1; }
         if ( find_integer( &group,            line, "GROUP",            true, 0, true,   7 ) == false ){ return -1; }
         if ( find_integer( &multiplicity,     line, "MULTIPLICITY",     true, 1, false, -1 ) == false ){ return -1; }
         if ( find_integer( &nelectrons,       line, "NELECTRONS",       true, 2, false, -1 ) == false ){ return -1; }
         if ( find_integer( &targetirrep,      line, "TARGETIRREP",      true, 0, true,   7 ) == false ){ return -1; }
         if ( find_integer( &ionized_orbital,  line, "IONIZED_ORBITAL",  true, 0, false, -1 ) == false ){ return -1; }

      }

      /*******************************
      *  Check the parsed variables  *
      *******************************/

      if( norbitals < 1 )                              { cerr << " The number of ORBITALS must be given and > 1 .\n";                    return -1; }
      if( ( group < 0 ) || ( group > 7 ) )             { cerr << " The molecular GROUP number of must be given and between 0 and 7 .\n"; return -1; }
      if( ( multiplicity < 0 ) )                       { cerr << " The multiplicity must be given and >= 0 .\n";                         return -1; }
      if( ( nelectrons < 0 ) )                         { cerr << " The number of electrons must be given and >1 0 .\n";                  return -1; }
      if( ( targetirrep < 0 ) || ( targetirrep > 7 ) ) { cerr << " The target IRREP must be given and between 0 and 7 .\n";              return -1; }
      if( ionized_orbital < 0 )                        { cerr << " The IONIZED_ORBITAL must be given and >=0 .\n";                       return -1; }
      if( input_file.length() == 0 )                   { cerr << " The INPUT_FILE must be given .\n";                                    return -1; }
      if( output_file.length() == 0 )                  { cerr << " The OUTPUT_FILE must be given .\n";                                   return -1; }


      CheMPS2::Irreps Symmhelper( group );
      const int num_irreps = Symmhelper.getNumberOfIrreps();

      const int irreps_n  = count( irreps.begin(), irreps.end(), ',' ) + 1;
      const bool irreps_ok = ( norbitals == irreps_n );

      if ( !irreps_ok ){
         cerr << "There should be " << norbitals << " numbers in IRREPS !" << endl;
         return -1;
      }

      fetch_ints( irreps, irreps_parsed, norbitals );

      for ( int cnt = 0; cnt < norbitals; cnt ++ ){
         if ( ( irreps_parsed[ cnt ] < 0 ) || ( irreps_parsed[ cnt ] > num_irreps ) ){
            cerr << "The irrep number in IRREPS has to be between 1 and " << num_irreps << " !" << endl;
            return -1;
         }
      }

      /********************************
      *  Output the parsed varaibles  *
      ********************************/

      cout << "   Preparing a Hartree-Fock ^ with the following options:\n";
      cout << "   GROUP              = " << Symmhelper.getGroupName() << endl;
      cout << "   MULTIPLICITY       = " << multiplicity << endl;
      cout << "   NELECTRONS         = " << nelectrons << endl;
      cout << "   IRREP              = " << Symmhelper.getIrrepName( targetirrep ) << endl;
      cout << "   IRREPS             = [ " << irreps_parsed[ 0 ]; for ( int cnt = 1; cnt < norbitals; cnt++ ){ cout << " ; " << irreps_parsed[ cnt ]; } cout << " ]" << endl;
      cout << "   IONIZED_ORBITAL    = " << ionized_orbital << endl;
      cout << "   INPUT_FILE         = " << input_file << endl;
      cout << "   OUTPUT_FILE        = " << output_file << endl;
      cout << "\n";

      /*****************************
      *  Create necessary objects  *
      *****************************/

      CheMPS2::Initialize::Init();
      CheMPS2::Hamiltonian hamNON( norbitals, group, &irreps_parsed[ 0 ] );
      CheMPS2::Problem probNON( &hamNON, multiplicity - 1, nelectrons, targetirrep );

      // /********************
      // *  Load the HF-MPS  *
      // ********************/

      // CheMPS2::SyBookkeeper * MPSBK  = new CheMPS2::SyBookkeeper( &probNON, 1 );
      // loadDIM( input_file, MPSBK );
      // CheMPS2::CTensorT    ** MPS    = new CheMPS2::CTensorT *[ norbitals ];

      // for ( int index = 0; index < norbitals; index++ ) {
      //    MPS[ index ] = new CheMPS2::CTensorT( index, MPSBK );
      // }
      // loadMPS( input_file, norbitals, MPS );

      /********************
      *  Load the HF-MPS  *
      ********************/

      CheMPS2::SyBookkeeper * MPSBK  = new CheMPS2::SyBookkeeper( &probNON, 100 );
      CheMPS2::CTensorT    ** MPS    = new CheMPS2::CTensorT *[ norbitals ];

      for ( int index = 0; index < norbitals; index++ ) {
         MPS[ index ] = new CheMPS2::CTensorT( index, MPSBK );
         MPS[ index ]->random();
      }

      /*****************************************
      *  Prepare the ionized state and save it *
      *****************************************/
      irreps_parsed.push_back( irreps_parsed[ ionized_orbital ] );

      CheMPS2::Hamiltonian hamION( norbitals + 1, group, &irreps_parsed[ 0 ] );
      hamION.setTmat( ionized_orbital, norbitals, 1.0 );
      CheMPS2::Problem probION( &hamION, multiplicity - 1, nelectrons, CheMPS2::Irreps::directProd( targetirrep, irreps_parsed[ ionized_orbital ] ) );
      probION.construct_mxelem();

      CheMPS2::SyBookkeeper * MPSBKION  = new CheMPS2::SyBookkeeper( &probION, 100 );
      CheMPS2::CTensorT    ** MPSION    = new CheMPS2::CTensorT *[ norbitals + 1 ];

      for ( int index = 0; index < norbitals + 1; index++ ) {
         MPSION[ index ] = new CheMPS2::CTensorT( index, MPSBKION );
         MPSION[ index ]->random();
      }

      applyAnnihilator( ionized_orbital, MPS, MPSBK, MPSION, MPSBKION, &probION );

      saveMPS( output_file, MPSION, MPSBKION );
      cout << "   The ionized state has been successfully stored to " << output_file << " .\n";

      /*******************************
      *  Delete the allocated stuff  *
      *******************************/

      for ( int site = 0; site < norbitals; site++ ) {
         delete MPS[ site ];
      }
      for ( int site = 0; site < norbitals + 1; site++ ) {
         delete MPSION[ site ];
      }
      delete[] MPS;
      delete[] MPSION;
      delete MPSBK;
      delete MPSBKION;

   }

   input.clear();
   input.seekg(0, ios::beg);

   input.close();

   return 0;

}