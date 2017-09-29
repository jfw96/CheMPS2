
#ifndef TENSOR_CHEMPS2_H
#define TENSOR_CHEMPS2_H

#include <complex>
#include "SyBookkeeper.h"
typedef std::complex<double> dcomplex;

namespace CheMPS2 {
/** Pure virtual CTensor class.
    \author Lars-Hendrik Frahm
    \date September 12, 2017

    The CTensor class defines parameters and functions which all CTensors must have. */
class CTensor {
 public:
  //! Get the number of tensor blocks
  /** return The number of tensor blocks */
  virtual int gNKappa() const = 0;

  //! Get the pointer to the storage
  /** return pointer to the storage */
  virtual dcomplex* gStorage() = 0;

  //! Get the pointer to the constant storage
  /** return pointer to the storage */
  virtual const dcomplex* gCStorage() const = 0;

  //! Get the index corresponding to a certain tensor block
  /** \param N1 The left or up particle number sector
      \param TwoS1 The left or up spin symmetry sector
      \param I1 The left or up irrep sector
      \param N2 The right or down particle number sector
      \param TwoS2 The right or down spin symmetry sector
      \param I2 The right or down irrep sector
      \return The kappa corresponding to the input parameters; -1 means no
      such block */
  virtual int gKappa(const int N1, const int TwoS1, const int I1, const int N2,
                     const int TwoS2, const int I2) const = 0;

  //! Get the storage jump corresponding to a certain tensor block
  /** \param kappa The symmetry block
      \return kappa2index[ kappa ], the memory jumper to a certain block */
  virtual int gKappa2index(const int kappa) const = 0;

  //! Get the pointer to the storage of a certain tensor block
  /** \param N1 The left or up particle number sector
      \param TwoS1 The left or up spin symmetry sector
      \param I1 The left or up irrep sector
      \param N2 The right or down particle number sector
      \param TwoS2 The right or down spin symmetry sector
      \param I2 The right or down irrep sector
      \return Pointer to the storage of the specified tensor block; NULL
      means no such block */
  virtual dcomplex* gStorage(const int N1, const int TwoS1, const int I1,
                             const int N2, const int TwoS2, const int I2) = 0;

  //! Get the pointer to the storage of a certain tensor block
  /** \param N1 The left or up particle number sector
      \param TwoS1 The left or up spin symmetry sector
      \param I1 The left or up irrep sector
      \param N2 The right or down particle number sector
      \param TwoS2 The right or down spin symmetry sector
      \param I2 The right or down irrep sector
      \return Pointer to the storage of the specified tensor block; NULL
      means no such block */
  virtual const dcomplex* gCStorage(const int N1, const int TwoS1, const int I1,
                                    const int N2, const int TwoS2,
                                    const int I2) const = 0;

  //! Get the location index
  /** \return the index */
  virtual int gIndex() const = 0;

protected:
  //! Index of the CTensor object. For CTensorT: a site index; for other tensors:
  //! a boundary index
  int index;

  //! Number of tensor blocks.
  int nKappa;

  //! The actual variables. CTensor block kappa begins at
  //! storage+kappa2index[kappa] and ends at storage+kappa2index[kappa+1].
  dcomplex* storage;

  //! kappa2index[kappa] indicates the start of tensor block kappa in storage.
  //! kappa2index[nKappa] gives the size of storage.
  int* kappa2index;
};
}

#endif
