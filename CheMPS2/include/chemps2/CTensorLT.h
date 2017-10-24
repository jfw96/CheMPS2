
#ifndef CTENSORLT_CHEMPS2_H
#define CTENSORLT_CHEMPS2_H

#include "CTensorO.h"
#include "CTensorT.h"

namespace CheMPS2 {
   /** CTensorLT class.
    \author Lars-Hendrik Frahm
    \date September 14, 2017

    The CTensorLT class is a storage and manipulation class for a single
   contracted creator/annihilitor. */
   class CTensorLT : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff          The irrep of the one creator ( sandwiched if
     CTensorLT ; to sandwich if TensorQ )
      \param moving_right   If true: sweep from left to right. If false: sweep
     from right to left
      \param book_up        Symmetry bookkeeper of the upper MPS
      \param book_down      Symmetry bookkeeper of the lower MPS */
      CTensorLT( const int boundary_index, const int Idiff, const bool moving_right, const CheMPS2::SyBookkeeper * book_up, const CheMPS2::SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorLT();

      //! Create a new CTensorLT
      /** \param mps_tensor_up   Upper TensorT from which the new CTensorLT should be
     made
      \param mps_tensor_down Lower TensorT from which the new CTensorLT should be
     made
      \param previous        Overlap matrix on the previous edge
      \param workmem         Work memory of size max(dimLup,down) *
     max(dimRup,down) */
      void create( CTensorT * mps_tensor_up, CTensorT * mps_tensor_down, CTensorO * previous, dcomplex * workmem );

      private:
      //! Make new when moving_right == true
      void create_right( const int ikappa, CTensorT * mps_tensor_up, CTensorT * mps_tensor_down, CTensorO * previous, dcomplex * workmem );

      //! Make new when moving_right == false
      void create_left( const int ikappa, CTensorT * mps_tensor_up, CTensorT * mps_tensor_down, CTensorO * previous, dcomplex * workmem );
   };
}

#endif
