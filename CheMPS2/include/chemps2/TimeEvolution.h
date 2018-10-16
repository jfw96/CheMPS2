#ifndef TIMEEVOLUTION_CHEMPS2_H
#define TIMEEVOLUTION_CHEMPS2_H

#include "CTensorT.h"
#include "ConvergenceScheme.h"
#include "Logger.h"
#include "MyHDF5.h"
#include "Problem.h"
#include "SyBookkeeper.h"
#include "hdf5_hl.h"
#include <ctime>

namespace CheMPS2 {

   class TimeEvolution {
      public:
      //! Constructor
      /** \param Problem to problem to be solved*/
      TimeEvolution( Problem * probIn, ConvergenceScheme * schemeIn, hid_t HDF5FILEIDIN );

      ~TimeEvolution();

      void Propagate( const char time_type, const double time_step_major, 
                      const double time_step_minor, const double time_final, 
                      CTensorT ** mpsIn, SyBookkeeper * bkIn, 
                      const int kry_size, 
                      const bool backwards, const double offset,
                      const bool do_ortho, const bool doDumpFCI, 
                      const bool doDump2RDM, const int nWeights = 0,
                      const int * hfState = NULL );

      private:
      void HDF5_MAKE_DATASET( hid_t setID, const char * name, int rank,
                              const hsize_t * dims, hid_t typeID, const void * data );

      double calcWieght( int nHoles, int nParticles, Problem * probState, CTensorT ** mpsState, SyBookkeeper * bkState, const int * hf_state );

      void doStep_euler( const double time_step, const int kry_size, 
                         dcomplex offset, const bool backwards, 
                         CTensorT ** mpsIn, SyBookkeeper * bkIn, 
                         CTensorT ** mpsOut, SyBookkeeper * bkOut );

      void doStep_arnoldi( const double time_step, 
                           const int kry_size, 
                           dcomplex offset, 
                           const bool backwards, 
                           const bool do_ortho,
                           CTensorT ** mpsIn, 
                           SyBookkeeper * bkIn, 
                           CTensorT ** mpsOut, 
                           SyBookkeeper * bkOut );

      void doStep_runge_kutta( const double time_step, const int kry_size, 
                               dcomplex offset, const bool backwards, 
                               CTensorT ** mpsIn, SyBookkeeper * bkIn, 
                               CTensorT ** mpsOut, SyBookkeeper * bkOut );

      const int L;

      Problem * prob;

      ConvergenceScheme * scheme;

      hid_t HDF5FILEID;

      std::time_t start;
   };
} // namespace CheMPS2

#endif
