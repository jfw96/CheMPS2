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

      void Propagate( const char time_type, SyBookkeeper * initBK, CTensorT ** initMPS,
                      const double time_step_major, const double time_step_minor, 
                      const double time_final, const int kry_size, 
                      const bool doImaginary, const bool doDumpFCI, 
                      const bool doDump2RDM );

      private:
      void HDF5_MAKE_DATASET( hid_t setID, const char * name, int rank,
                              const hsize_t * dims, hid_t typeID, const void * data );

      // void doStep_krylov( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut );

      void doStep_arnoldi( const double time_step, const int kry_size,
                           dcomplex offset, const bool doImaginary, 
                           CTensorT ** mpsIn, SyBookkeeper * bkIn,
                           CTensorT ** mpsOut, SyBookkeeper * bkOut );

      void doStep_runge_kutta( const double time_step, const int kry_size, 
                               dcomplex offset, const bool doImaginary, 
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
