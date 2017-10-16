#ifndef TIMETAYLOR_CHEMPS2_H
#define TIMETAYLOR_CHEMPS2_H

#include "Problem.h"
#include "Logger.h"
#include "SyBookkeeper.h"
// #include "TensorF0.h"
// #include "TensorF0T.h"
// #include "TensorF1.h"
// #include "TensorF1T.h"
// #include "TensorL.h"
// #include "TensorLT.h"
// #include "TensorQ.h"
// #include "TensorQT.h"
// #include "TensorS0.h"
// #include "TensorS0T.h"
// #include "TensorS1.h"
// #include "TensorS1T.h"
// #include "TensorT.h"
// #include "TensorX.h"

namespace CheMPS2 {
   class TimeTaylor {
      public:
      
          //! Constructor
          /** \param Problem to problem to be solved*/
      TimeTaylor(Problem* probIn, Logger* loggerIn);

          ~TimeTaylor();

          // void Propagate();

      private:
          const int L;

          Problem* prob;

          Logger* logger;

          // TensorT** MPS;
          // TensorT** MPSDT;

      
          CheMPS2::SyBookkeeper* denBK;
          CheMPS2::SyBookkeeper* denBKDT;

  // // Whether or not allocated
  // int* isAllocated;

  // // TensorL's
  // // Ltensors_**upperMPS**_**lowerMPS**
  // TensorL*** Ltensors;
  // TensorLT*** LtensorsT;

  // // TensorX's
  // TensorX** Xtensors;

  // // TensorF0F0's
  // TensorF0**** F0tensors;
  // TensorF0T**** F0tensorsT;

  // // TensorF1F1's
  // TensorF1**** F1tensors;
  // TensorF1T**** F1tensorsT;

  // // TensorS0S0's
  // TensorS0**** S0tensors;
  // TensorS0T**** S0tensorsT;

  // // TensorS1S1's
  // TensorS1**** S1tensors;
  // TensorS1T**** S1tensorsT;

  // // ABCD-tensors
  // TensorOperator**** Atensors;
  // TensorOperator**** AtensorsT;
  // TensorOperator**** Btensors;
  // TensorOperator**** BtensorsT;
  // TensorOperator**** Ctensors;
  // TensorOperator**** CtensorsT;
  // TensorOperator**** Dtensors;
  // TensorOperator**** DtensorsT;

  // // TensorQQ's
  // TensorQ*** Qtensors;
  // TensorQT*** QtensorsT;

  // // TensorO's
  // TensorO** Otensors;

  // void deleteAllBoundaryOperators();

  // void deleteTensors(const int index, const bool movingRight);

  // void updateMovingRightSafe(const int cnt);

  // void updateMovingRight(const int index);

  // void allocateTensors(const int index, const bool movingRight);

  // void doStep();

  // void updateMovingLeftSafe(const int cnt);

  // void updateMovingLeft(const int index);

  // double calcEnergy();
};
}

#endif
