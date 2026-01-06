#ifndef _TRACKED_BICGSTAB_INCLUDE
#define _TRACKED_BICGSTAB_INCLUDE

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <chrono>

namespace CustomSolvers {

// Helper function to solve with progress tracking
template<typename SolverType, typename MatrixType, typename VectorType>
VectorType solveWithTracking(SolverType& solver,
                              const MatrixType& A,
                              const VectorType& b,
                              int trackInterval = 100)
{
    using Scalar = typename VectorType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    // Initial guess
    VectorType x = VectorType::Zero(b.size());
    VectorType x_prev = x;

    // Track time
    auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << "  Starting iterations (tracking every " << trackInterval << " iterations)..." << std::endl;

    // We'll use multiple solver calls with limited iterations
    int maxIterPerRound = trackInterval;
    int totalMaxIter = solver.maxIterations();
    RealScalar tolerance = solver.tolerance();
    RealScalar rhsNorm = b.norm();

    int totalIterations = 0;
    bool converged = false;

    for(int round = 0; round < (totalMaxIter / maxIterPerRound) + 1; round++)
    {
        // Configure solver for this round
        solver.setMaxIterations(maxIterPerRound);
        solver.setTolerance(tolerance);

        // Solve with current guess
        x = solver.solveWithGuess(b, x);

        totalIterations += solver.iterations();
        RealScalar currentError = solver.error();

        // Check convergence
        VectorType residual = b - A * x;
        RealScalar residualNorm = residual.norm();
        RealScalar relativeError = residualNorm / rhsNorm;

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);

        std::cout << "  Iteration " << totalIterations
                  << ": relative residual = " << relativeError
                  << ", solver error = " << currentError
                  << ", time = " << elapsed.count() / 1000.0 << "s" << std::endl;

        if(solver.info() == Eigen::Success || relativeError < tolerance)
        {
            converged = true;
            break;
        }

        if(totalIterations >= totalMaxIter)
        {
            break;
        }

        // Check if we're making progress
        RealScalar change = (x - x_prev).norm();
        if(change < Eigen::NumTraits<RealScalar>::epsilon() * x.norm())
        {
            std::cout << "  Warning: solver is not making progress (change = " << change << ")" << std::endl;
            break;
        }

        x_prev = x;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    VectorType finalResidual = b - A * x;
    RealScalar finalError = finalResidual.norm() / rhsNorm;

    std::cout << "  Final: " << totalIterations
              << " iterations, relative residual = " << finalError
              << ", total time = " << totalTime.count() / 1000.0 << "s"
              << ", status = " << (converged ? "CONVERGED" : "NOT CONVERGED") << std::endl;

    return x;
}


} // namespace CustomSolvers

#endif // _TRACKED_BICGSTAB_INCLUDE

