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
#include "Problem.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {

   class TimeTaylor {
      public:
      //! Constructor
      /** \param Problem to problem to be solved*/
      TimeTaylor( Problem * probIn, ConvergenceScheme * schemeIn, Logger * loggerIn );

      ~TimeTaylor();

      void Propagate();

      double Energy();

      private:
      const int L;

      Problem * prob;

      ConvergenceScheme * scheme;

      Logger * logger;

      CTensorT ** MPS;
      CTensorT ** MPSDT;

      SyBookkeeper * denBK;
      SyBookkeeper * denBKDT;

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

      void updateMovingRightSafe( const int cnt );

      void updateMovingRight( const int index );

      void updateMovingLeftSafe( const int cnt );

      void updateMovingLeft( const int index );

      void allocateTensors( const int index, const bool movingRight );

      void deleteAllBoundaryOperators();

      void deleteTensors( const int index, const bool movingRight );

      void doStep( const int currentInstruction );
   };
}

#endif
