#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"
#include <omp.h>

namespace {

// original loop
// void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
//                    const Matrix& A, const Matrix& B, Matrix& C) {
//   for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
//     for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
//       for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
//         C(i, j) += A(i, k) * B(k, j);
// }

// loop with best time
// void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
//                    const Matrix& A, const Matrix& B, Matrix& C) {
  
//   for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
//     for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
//       for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
//         C(i, j) += A(i, k) * B(k, j);
// }

// parallel loop
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  // #pragma omp parallel for
  for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
      for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
        C(i, j) += A(i, k) * B(k, j);
}

// const int szBlock = 16;
// const int szBlock = 32;
// const int szBlock = 64;
// const int szBlock = 128;
const int szBlock = 256;
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  // prodSubBlocks(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  std::cout << "Size block: " << szBlock << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < A.nbRows; i += szBlock)
    for (int j = 0; j < B.nbCols; j += szBlock)
      for (int k = 0; k < A.nbCols; k += szBlock)
        prodSubBlocks(i, j, k, szBlock, A, B, C); 


  return C;
}
