#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move Q to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K matrix by concatenating keys[0..i]
    Matrix* K = matrix_memory_allocator.Allocate("K");
    gpu_sim.MoveMatrixToSharedMem(keys[0]);
    gpu_sim.Copy(keys[0], K, Position::kInSharedMemory);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix* temp = matrix_memory_allocator.Allocate("K_temp");
      gpu_sim.Concat(K, keys[j], temp, 0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(K);
      K = temp;
    }

    // Build V matrix by concatenating values[0..i]
    Matrix* V = matrix_memory_allocator.Allocate("V");
    gpu_sim.MoveMatrixToSharedMem(values[0]);
    gpu_sim.Copy(values[0], V, Position::kInSharedMemory);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      Matrix* temp = matrix_memory_allocator.Allocate("V_temp");
      gpu_sim.Concat(V, values[j], temp, 0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(V);
      V = temp;
    }

    // Compute Q * K^T
    Matrix* K_copy = matrix_memory_allocator.Allocate("K_copy");
    gpu_sim.Copy(K, K_copy, Position::kInSharedMemory);
    gpu_sim.Transpose(K_copy, Position::kInSharedMemory);

    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_copy, QK);

    // Compute exp(QK)
    Matrix* exp_QK = matrix_memory_allocator.Allocate("exp_QK");
    gpu_sim.MatExp(QK, exp_QK);

    // Compute row sums of exp_QK for softmax denominator
    Matrix* S = nullptr;

    for (size_t row = 0; row <= i; ++row) {
      Matrix* row_mat = matrix_memory_allocator.Allocate("row_mat");
      gpu_sim.GetRow(exp_QK, row, row_mat, Position::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_mat, row_sum);

      if (row == 0) {
        S = row_sum;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("S_temp");
        gpu_sim.Concat(S, row_sum, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(S);
        gpu_sim.ReleaseMatrix(row_sum);
        S = temp;
      }
      gpu_sim.ReleaseMatrix(row_mat);
    }

    // Compute softmax: divide each row of exp_QK by corresponding S[r]
    Matrix* softmax_result = nullptr;

    for (size_t row = 0; row <= i; ++row) {
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.GetRow(exp_QK, row, exp_row, Position::kInSharedMemory);

      Matrix* s_scalar = matrix_memory_allocator.Allocate("s_scalar");
      gpu_sim.GetRow(S, row, s_scalar, Position::kInSharedMemory);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, s_scalar, softmax_row);

      if (row == 0) {
        softmax_result = softmax_row;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("softmax_temp");
        gpu_sim.Concat(softmax_result, softmax_row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax_result = temp;
      }

      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(s_scalar);
    }

    // Compute final result: softmax_result * V
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_result, V, result);

    // Move result to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(result);

    // Run simulator and commit answer
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);

    // Release matrices to free memory
    gpu_sim.ReleaseMatrix(K);
    gpu_sim.ReleaseMatrix(K_copy);
    gpu_sim.ReleaseMatrix(QK);
    gpu_sim.ReleaseMatrix(exp_QK);
    gpu_sim.ReleaseMatrix(S);
    gpu_sim.ReleaseMatrix(softmax_result);
    gpu_sim.ReleaseMatrix(V);
    // result is released automatically after CommitAnswer
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu