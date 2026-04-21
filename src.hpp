#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // We need to compute attention: softmax(Q * K^T) * V
    // Where:
    // - Q: current_query with shape [i+1, 512]
    // - K: concatenation of keys[0..i], each shape [1, 512] -> shape [i+1, 512] (i+1 groups)
    // - V: concatenation of values[0..i], each shape [1, 512] -> shape [i+1, 512]
    // Output shape: [i+1, 512]
    // Note: i is 0-based, round = i+1 uses first i+1 groups

    // First, move Q to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Concatenate all keys[0..i] into K matrix
    // Start by moving first key to SRAM
    gpu_sim.MoveMatrixToSharedMem(keys[0]);
    Matrix* K = matrix_memory_allocator.Allocate("K");
    gpu_sim.Copy(keys[0], K, Position::kInSharedMemory);

    // Concatenate remaining keys up to i
    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix* temp = matrix_memory_allocator.Allocate("K_temp");
      gpu_sim.Concat(K, keys[j], temp, 0, Position::kInSharedMemory); // axis=0 for vertical concat
      gpu_sim.ReleaseMatrix(K);
      K = temp;
    }

    // Concatenate all values[0..i] into V matrix
    gpu_sim.MoveMatrixToSharedMem(values[0]);
    Matrix* V = matrix_memory_allocator.Allocate("V");
    gpu_sim.Copy(values[0], V, Position::kInSharedMemory);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      Matrix* temp = matrix_memory_allocator.Allocate("V_temp");
      gpu_sim.Concat(V, values[j], temp, 0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(V);
      V = temp;
    }

    // Compute Q * K^T
    // First transpose K (need a copy since Transpose is in-place)
    Matrix* K_copy = matrix_memory_allocator.Allocate("K_copy");
    gpu_sim.Copy(K, K_copy, Position::kInSharedMemory);
    gpu_sim.Transpose(K_copy, Position::kInSharedMemory);

    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_copy, QK); // shape [i+1, i+1]

    // Apply softmax row-wise: softmax(x) = exp(x) / sum(exp(x)) per row
    // Compute exp(QK)
    Matrix* exp_QK = matrix_memory_allocator.Allocate("exp_QK");
    gpu_sim.MatExp(QK, exp_QK); // shape [i+1, i+1]

    // Compute row sums of exp_QK to get denominator for softmax
    // We need column vector S where S[r] = sum_{c} exp_QK[r,c]
    // Build S by concatenating row sums
    Matrix* S = nullptr; // column vector [i+1, 1]

    for (size_t row = 0; row <= i; ++row) {
      Matrix* row_mat = matrix_memory_allocator.Allocate("row_" + std::to_string(row));
      gpu_sim.GetRow(exp_QK, row, row_mat, Position::kInSharedMemory);
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row));
      gpu_sim.Sum(row_mat, row_sum); // 1x1 matrix

      if (row == 0) {
        S = row_sum;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("S_temp");
        gpu_sim.Concat(S, row_sum, temp, 0, Position::kInSharedMemory); // vertical concat
        gpu_sim.ReleaseMatrix(S);
        S = temp;
      }
      gpu_sim.ReleaseMatrix(row_mat);
    }

    // Now compute softmax: exp_QK[r,c] / S[r]
    // We need to divide each row of exp_QK by corresponding S[r]
    // Create matrix where each row is S repeated i times
    // Actually, MatDiv divides by a scalar (1x1 matrix) element-wise
    // But we need row-wise division
    // For now, implement naive: process each row separately
    Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result");

    for (size_t row = 0; row <= i; ++row) {
      // Get row of exp_QK
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(row));
      gpu_sim.GetRow(exp_QK, row, exp_row, Position::kInSharedMemory);

      // Get S[r] (scalar)
      Matrix* s_row = matrix_memory_allocator.Allocate("s_row_" + std::to_string(row));
      gpu_sim.GetRow(S, row, s_row, Position::kInSharedMemory); // S is [i+1, 1], so row is 1x1

      // Divide exp_row by s_row (element-wise)
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, s_row, softmax_row); // shape [1, i]

      // Concatenate rows to build softmax_result
      if (row == 0) {
        softmax_result = softmax_row;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("softmax_temp");
        gpu_sim.Concat(softmax_result, softmax_row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        softmax_result = temp;
      }

      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(s_row);
    }

    // Multiply softmax_result * V
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_result, V, result); // shape [i+1, 512]

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu