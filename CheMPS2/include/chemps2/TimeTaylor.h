#ifndef TIMETAYLOR_CHEMPS2_H
#define TIMETAYLOR_CHEMPS2_H

#include "CTensorF0.h"
#include "CTensorF0T.h"
#include "CTensorF1.h"
#include "CTensorF1T.h"
#include "CTensorL.h"
#include "CTensorLT.h"
#include "CTensorQ.h"
#include "CTensorQT.h"
#include "CTensorS0.h"
#include "CTensorS0T.h"
#include "CTensorS1.h"
#include "CTensorS1T.h"
#include "CTensorT.h"
#include "CTensorX.h"
#include "ConvergenceScheme.h"
#include "Logger.h"
#include "MyHDF5.h"
#include "Problem.h"
#include "SyBookkeeper.h"
#include "hdf5_hl.h"

namespace CheMPS2 {

   class TimeTaylor {
      public:
      //! Constructor
      /** \param Problem to problem to be solved*/
      TimeTaylor( Problem * probIn, ConvergenceScheme * schemeIn, hid_t HDF5FILEIDIN );

      ~TimeTaylor();

      void Propagate( SyBookkeeper * initBK, CTensorT ** initMPS, const bool doImaginary = false, const bool doDumpFCI = false );

      void fitApplyH( dcomplex factor, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut, const int nSweeps = 5, const int D = 100, const double cut_off = 1e-10 );

      void fitApplyH_1site( CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut, const int nSweeps );

      void fitAddMPS( dcomplex factor,
                      CTensorT ** mpsA, SyBookkeeper * bkA,
                      CTensorT ** mpsB, SyBookkeeper * bkB,
                      CTensorT ** mpsOut, SyBookkeeper * bkOut,
                      const int nSweeps = 5, const int D = 100, const double cut_off = 1e-10 );

      private:
      const int L;

      Problem * prob;

      ConvergenceScheme * scheme;

      hid_t HDF5FILEID;

      std::time_t start;
      // CTensorT ** MPS;
      // CTensorT ** MPSDT;

      // SyBookkeeper * denBK;
      // SyBookkeeper * denBKDT;

      // Whether or not allocated
      int * isAllocated;

      // TensorL's
      CTensorL *** Ltensors;
      CTensorLT *** LtensorsT;

      // TensorX's
      CTensorX ** Xtensors;

      // TensorF0F0's
      CTensorF0 **** F0tensors;
      CTensorF0T **** F0tensorsT;

      // TensorF1F1's
      CTensorF1 **** F1tensors;
      CTensorF1T **** F1tensorsT;

      // TensorS0S0's
      CTensorS0 **** S0tensors;
      CTensorS0T **** S0tensorsT;

      // TensorS1S1's
      CTensorS1 **** S1tensors;
      CTensorS1T **** S1tensorsT;

      // ABCD-tensors
      CTensorOperator **** Atensors;
      CTensorOperator **** AtensorsT;
      CTensorOperator **** Btensors;
      CTensorOperator **** BtensorsT;
      CTensorOperator **** Ctensors;
      CTensorOperator **** CtensorsT;
      CTensorOperator **** Dtensors;
      CTensorOperator **** DtensorsT;

      // TensorQQ's
      CTensorQ *** Qtensors;
      CTensorQT *** QtensorsT;

      // TensorO's
      CTensorO ** Otensors;

      void HDF5_MAKE_DATASET(hid_t setID, const char * name, int rank, const hsize_t *dims, hid_t typeID, const void * data );

      void updateMovingRightSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void updateMovingRight( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void updateMovingLeftSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void updateMovingLeft( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void allocateTensors( const int index, const bool movingRight, SyBookkeeper * bkUp, SyBookkeeper * bkDown );

      void deleteAllBoundaryOperators();

      void deleteTensors( const int index, const bool movingRight );

      void doStep_euler_g( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut );
      void doStep_taylor_1( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut );
      void doStep_taylor_1site( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut );
      void doStep_rk_4( const int currentInstruction, const bool doImaginary, const double offset );
      void doStep_krylov( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut );
      int doStep_arnoldi( const int currentInstruction, const bool doImaginary, const double offset, CTensorT ** mpsIn, SyBookkeeper * bkIn, CTensorT ** mpsOut, SyBookkeeper * bkOut );
   };
} // namespace CheMPS2

#endif
