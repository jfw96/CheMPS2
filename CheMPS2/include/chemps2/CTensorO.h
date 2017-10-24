
#ifndef TENSORO_CHEMPS2_H
#define TENSORO_CHEMPS2_H

#include "CTensorOperator.h"
#include "CTensorT.h"
#include "Problem.h"

namespace CheMPS2 {
   /** TensorO class.
    \author Lars-Hendrik Frahm
    \date September 13, 2017

    The TensorO class is a storage class for overlaps between different MPSs. */
   class CTensorO : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param book_up   The symmetry bookkeeper with the upper symmetry sector
     virtual dimensions
      \param book_down The symmetry bookkeeper with the lower symmetry sector
     virtual dimensions */
      CTensorO( const int boundary_index, const bool moving_right, const CheMPS2::SyBookkeeper * book_up, const CheMPS2::SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorO();

      //! Clear and add the relevant terms to the CTensorO
      /** \param mps_tensor_up   Upper MPS tensor from which the new CTensorO should
     be made
      \param mps_tensor_down Lower MPS tensor from which the new CTensorO should
     be made */
      void create( CTensorT * mps_tensor_up, CTensorT * mps_tensor_down );

      //! Update the previous CTensorO
      /** \param mps_tensor_up   Upper MPS tensor from which the
      update should be made
      \param mps_tensor_down Lower MPS tensor from which the
      update should be made
      \param previous        Previous CTensorO from which the
      update should be made */
      void update_ownmem( CTensorT * mps_tensor_up, CTensorT * mps_tensor_down, CTensorO * previous );

      private:
      // helper functions
      void create_right( const int ikappa, CTensorT * mps_tensor_up, CTensorT * mps_tensor_down );

      void create_left( const int ikappa, CTensorT * mps_tensor_up, CTensorT * mps_tensor_down );
   };
}

#endif
