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
#include "CASSCF.h"
#include "Molden.h"
#include "MPIchemps2.h"
#include "EdmistonRuedenberg.h"
#include "TimeEvolution.h"
#include "CFCI.h"
#include "Irreps.h"

using namespace std;

void fetch_ints( const string rawdata, int * result, const int num ){

   int pos  = 0;
   int pos2 = 0;
   for ( int no = 0; no < num; no++ ){
      pos2 = rawdata.find( ",", pos );
      if ( pos2 == string::npos ){ pos2 = rawdata.length(); }
      result[ no ] = atoi( rawdata.substr( pos, pos2-pos ).c_str() );
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

int clean_exit( const int return_code ){

   #ifdef CHEMPS2_MPI_COMPILATION
   CheMPS2::MPIchemps2::mpi_finalize();
   #endif

   return return_code;

}

bool print_molcas_reorder( int * dmrg2ham, const int L, const string filename, const bool read ){

   bool on_disk = false;

   if ( read ){
      struct stat file_info;
      on_disk = (( filename.length() > 0 ) && ( stat( filename.c_str(), &file_info ) == 0 ));
      if ( on_disk ){
         ifstream input( filename.c_str() );
         string line;
         getline( input, line );
         const int num = count( line.begin(), line.end(), ',' ) + 1;
         assert( num == L );
         fetch_ints( line, dmrg2ham, L );
         input.close();
         cout << "Read orbital reordering = [ ";
         for ( int orb = 0; orb < L - 1; orb++ ){ cout << dmrg2ham[ orb ] << ", "; }
         cout << dmrg2ham[ L - 1 ] << " ]." << endl;
      }
   } else { // write
      FILE * capturing;
      capturing = fopen( filename.c_str(), "w" ); // "w" with fopen means truncate file
      for ( int orb = 0; orb < L - 1; orb++ ){
         fprintf( capturing, "%d, ", dmrg2ham[ orb ] );
      }
      fprintf( capturing, "%d \n", dmrg2ham[ L - 1 ] );
      fclose( capturing );
      cout << "Orbital reordering written to " << filename << "." << endl;
      on_disk = true;
   }

   return on_disk;

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
"       TIME_STEP_MAJOR = flt\n"
"              Set the time step (DT) for wave function anlysis. Required when TIME_EVOLU = TRUE (positive float).\n"
"\n"
"       TIME_STEP_MINOR = flt\n"
"              Set the time step (DT) for the time evolution calculation. Required when TIME_EVOLU = TRUE (positive float).\n"
"\n"
"       TIME_FINAL = flt\n"
"              Set the final time for the time evolution calculation. Required when TIME_EVOLU = TRUE (positive float). \n"
"\n"
"       TIME_NINIT = int, int, int\n"
"              Set the occupation numbers for the inital state. Required when TIME_EVOLU = TRUE. Ordered as in the FCIDUMP file. (positive integers).\n"
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
   string fcidump   = "";

   int group        = -1;
   int multiplicity = -1;
   int nelectrons   = -1;
   int irrep        = -1;

   char   time_type       = 'K';
   double time_step_major = 0.0;
   double time_step_minor = 0.0;
   double time_final      = 0.0;
   string time_ninit      = "";
   string time_hf_state   = "";
   string time_hdf5output = "";
   int    time_n_weights  = 0; 
   int    time_krysize    = 0;
   bool   time_dumpfci    = false;
   bool   time_dump2rdm   = false;

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

      if ( line.find( "FCIDUMP" ) != string::npos ){
         const int pos = line.find( "=" ) + 1;
         fcidump = line.substr( pos, line.length() - pos );
         fcidump.erase( remove( fcidump.begin(), fcidump.end(), ' ' ), fcidump.end() );
         if ( file_exists( fcidump, "FCIDUMP" ) == false ){ return -1; }
      }

      if ( line.find( "TIME_HDF5OUTPUT" ) != string::npos ){
         const int pos   = line.find( "=" ) + 1;
         time_hdf5output = line.substr( pos, line.length() - pos );
         time_hdf5output.erase( remove( time_hdf5output.begin(), time_hdf5output.end(), ' ' ), time_hdf5output.end() );
      }

      if ( find_integer( &group,        line, "GROUP",        true, 0, true,   7 ) == false ){ return -1; }
      if ( find_integer( &multiplicity, line, "MULTIPLICITY", true, 1, false, -1 ) == false ){ return -1; }
      if ( find_integer( &nelectrons,   line, "NELECTRONS",   true, 2, false, -1 ) == false ){ return -1; }
      if ( find_integer( &irrep,        line, "IRREP",        true, 0, true,   7 ) == false ){ return -1; }

      char options1[] = { 'K', 'R', 'E', 'F' };
      if ( find_character( &time_type,        line, "TIME_TYPE",        options1, 4 ) == false ){ return -1; }

      if ( find_boolean( &time_dumpfci,     line, "TIME_DUMPFCI"     ) == false ){ return -1; }
      if ( find_boolean( &time_dump2rdm,    line, "TIME_DUMP2RDM"    ) == false ){ return -1; }

      if ( line.find( "TIME_STEP_MAJOR" ) != string::npos ){
         find_double( &time_step_major, line, "TIME_STEP_MAJOR", true, 0.0 );
      }

      if ( line.find( "TIME_STEP_MINOR" ) != string::npos ){
         find_double( &time_step_minor, line, "TIME_STEP_MINOR", true, 0.0 );
      }

      if ( line.find( "TIME_FINAL" ) != string::npos ){
         find_double( &time_final, line, "TIME_FINAL", true, 0.0 );
      }

      if ( line.find( "TIME_NINIT" ) != string::npos ){
         const int pos = line.find( "=" ) + 1;
         time_ninit = line.substr( pos, line.length() - pos );
      }

      if ( line.find( "TIME_HF_STATE" ) != string::npos ){
         const int pos = line.find( "=" ) + 1;
         time_hf_state = line.substr( pos, line.length() - pos );
      }

      if ( line.find( "TIME_KRYSIZE" ) != string::npos ){
         find_integer( &time_krysize, line, "TIME_KRYSIZE", true, 1, false, -1 );
      }

      if ( line.find( "TIME_N_WEIGHTS" ) != string::npos ){
         find_integer( &time_n_weights, line, "TIME_N_WEIGHTS", true, 1, false, -1 );
      }
   }
   input.close();

  /*******************************
   *  Check the target symmetry  *
   *******************************/

   if ( group == -1 ){
      cerr << "GROUP is a mandatory option!" << endl; 
      return -1;
   }
   CheMPS2::Irreps Symmhelper( group );
   const int num_irreps = Symmhelper.getNumberOfIrreps();

   int fcidump_norb  = -1;
   int fcidump_nelec = -1;
   int fcidump_two_s = -1;
   int fcidump_irrep = -1;
   {
      ifstream thefcidump( fcidump.c_str() );
      string line;
      int pos, pos2;
      getline( thefcidump, line ); // &FCI NORB= X,NELEC= Y,MS2= Z,
      pos = line.find( "FCI" );
      if ( pos == string::npos ){
         cerr << "The file " << fcidump << " is not a fcidump file!" << endl; 
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
   if ( multiplicity == -1 ){ multiplicity = fcidump_two_s + 1; }
   if ( nelectrons   == -1 ){   nelectrons = fcidump_nelec;     }
   if ( irrep        == -1 ){        irrep = fcidump_irrep;     }

   /************************
   *  Parse time evolution *
   ************************/

   int * time_ninit_parsed    = new int[ fcidump_norb ];
   int * time_hf_state_parsed = NULL;


   if ( time_step_major <= 0.0 ){
      cerr << "TIME_STEP_MAJOR should be greater than zero !" << endl;
      return -1;
   }

   if ( time_step_minor <= 0.0 ){
      cerr << "TIME_STEP_MINOR should be greater than zero !" << endl;
      return -1;
   }

   if( std::abs( ( time_step_major / time_step_minor ) - round( time_step_major / time_step_minor ) ) > 1e-6 ){
      cerr << "TIME_STEP_MAJOR must be N*TIME_STEP_MINOR !" << endl;
      return -1;
   }

   if ( time_final <= 0 ){
      cerr << "TIME_FINAL should be greater than zero !" << endl;
      return -1;
   }

   if ( time_ninit.length() == 0 ){
      cerr << "TIME_NINIT is mandatory options when TIME_EVOLU = TRUE !" << endl;
      return -1;
   }

   if ( time_type == 'K' && time_krysize <= 0 ){
      cerr << "TIME_KRYSIZE should be greater than zero if TIME_TYPE = K!" << endl;
      return -1;
   }

   const int ni_ini  = count( time_ninit.begin(), time_ninit.end(), ',' ) + 1;
   const bool init_ok = ( fcidump_norb == ni_ini );

   if ( init_ok == false ){
      cerr << "There should be " << fcidump_norb << " numbers in TIME_NINIT when TIME_EVOLU = TRUE !" << endl;
      return -1;
   }

   fetch_ints( time_ninit, time_ninit_parsed, fcidump_norb );

   for ( int cnt = 0; cnt < fcidump_norb; cnt ++ ){
      if ( ( time_ninit_parsed[ cnt ] < 0 ) || ( time_ninit_parsed[ cnt ] > 2 ) ){
         cerr << "The occupation number in TIME_NINIT has to be 0, 1 or 2 !" << endl;
         return -1;
      }
   }

   int elec_sum = 0; for ( int cnt = 0; cnt < fcidump_norb; cnt++ ) { elec_sum += time_ninit_parsed[ cnt ];  }
   if ( elec_sum != nelectrons ){
      cerr << "There should be " << nelectrons << " distributed over the molecular orbitals in TIME_NINIT !" << endl;
      return -1;
   }

   if( time_n_weights > 0 ){
      const int hf_ini  = count( time_hf_state.begin(), time_hf_state.end(), ',' ) + 1;
      const bool hf_ok = ( fcidump_norb == hf_ini );

      if ( hf_ok == false ){
         cerr << "There should be " << fcidump_norb << " numbers in TIME_HF_STATE  !" << endl;
         return -1;
      }

      time_hf_state_parsed = new int[ fcidump_norb ];
      fetch_ints( time_hf_state, time_hf_state_parsed, fcidump_norb );

      for ( int cnt = 0; cnt < fcidump_norb; cnt ++ ){
         if ( !(time_hf_state_parsed[ cnt ] == 0 || time_hf_state_parsed[ cnt ] == 2 ) ){
            cerr << "The occupation number in TIME_HF_STATE has to be 0 or 2 (closed shell) !" << endl;
            return -1;
         }
      }
   }

   /**********************
   *  Print the options  *
   ***********************/

   cout << "\nRunning chemps2 version " << CHEMPS2_VERSION << " with the following options:\n" << endl;
   cout << "   FCIDUMP            = " << fcidump << endl;
   cout << "   GROUP              = " << Symmhelper.getGroupName() << endl;
   cout << "   MULTIPLICITY       = " << multiplicity << endl;
   cout << "   NELECTRONS         = " << nelectrons << endl;
   cout << "   IRREP              = " << Symmhelper.getIrrepName( irrep ) << endl;
   cout << "   TIME_TYPE          = " << time_type << endl;
   cout << "   TIME_STEP_MAJOR    = " << time_step_major << endl;
   cout << "   TIME_STEP_MINOR    = " << time_step_minor << endl;
   cout << "   TIME_FINAL         = " << time_final << endl;
   cout << "   TIME_NINIT         = [ " << time_ninit_parsed[ 0 ]; for ( int cnt = 1; cnt < fcidump_norb; cnt++ ){ cout << " ; " << time_ninit_parsed[ cnt ]; } cout << " ]" << endl;
   if( time_n_weights > 0 ){
      cout << "   TIME_HF_STATE      = [ " << time_hf_state_parsed[ 0 ]; for ( int cnt = 1; cnt < fcidump_norb; cnt++ ){ cout << " ; " << time_hf_state_parsed[ cnt ]; } cout << " ]" << endl; 
      cout << "   TIME_N_WEIGHTS     = [ " << time_n_weights << endl; 
   }
   cout << "   TIME_KRYSIZE       = " << time_krysize << endl;
   cout << "   TIME_HDF5OUTPUT    = " << time_hdf5output << endl;
   cout << "   TIME_DUMPFCI       = " << (( time_dumpfci    ) ? "TRUE" : "FALSE" ) << endl;
   cout << "   TIME_DUMP2RDM      = " << (( time_dump2rdm   ) ? "TRUE" : "FALSE" ) << endl;
   cout << " " << endl;

   /********************************
   *  Running the DMRG calculation *
   ********************************/

   CheMPS2::Initialize::Init();
   CheMPS2::Hamiltonian * ham = new CheMPS2::Hamiltonian( fcidump, group );
   CheMPS2::Problem * prob = new CheMPS2::Problem( ham, multiplicity - 1, nelectrons, irrep );

   int sum_up      = 0;
   int sum_down    = 0;
   int * bits_up   = new int[ prob->gL() ];
   int * bits_down = new int[ prob->gL() ];

   for( int orb = 0; orb < prob->gL(); orb++ ){
      if( time_ninit_parsed[ orb ] == 2 ){
         bits_up[ orb ] = 1;
         bits_down[ orb ] = 1;
         sum_up++;
         sum_down++;
      } else if ( time_ninit_parsed[ orb ] == 1 ) {
         bits_up[ orb ] = 1;
         bits_down[ orb ] = 0;
         sum_up++;
      } else {
         bits_up[ orb ] = 0;
         bits_down[ orb ] = 0;
      }
   }

   hid_t fileID = H5_CHEMPS2_TIME_NO_H5OUT;
   if ( time_hdf5output.length() > 0){ fileID = H5Fcreate( time_hdf5output.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT ); }

   CheMPS2::CFCI * fcisolver = new CheMPS2::CFCI( ham, sum_up, sum_down, irrep, 100.0, 2, fileID );

   int length = fcisolver->getVecLength( 0 );
   dcomplex * start = new dcomplex [ length ];
   fcisolver->ClearVector( length, start );
   fcisolver->setFCIcoeff( bits_up, bits_down, 1.0, start );

   fcisolver->TimeEvolution( time_step_major, time_step_minor, time_final, start, time_krysize, time_dumpfci, time_dump2rdm, time_n_weights, time_hf_state_parsed );

   delete[] start;

   delete[] bits_up;
   delete[] bits_down;
   delete fcisolver;

   delete prob;
   delete ham;
   delete [] time_ninit_parsed;
   delete[] time_hf_state_parsed;

   return 0;

}