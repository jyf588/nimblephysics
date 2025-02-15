#include "dart/realtime/SSID.hpp"

#include <thread>

#include "dart/realtime/Millis.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

#include "signal.h"

namespace dart {

using namespace trajectory;

namespace realtime {

SSID::SSID(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<trajectory::LossFn> loss,
    int planningHistoryMillis,
    int sensorDim)
  : mRunning(false),
    mWorld(world),
    mLoss(loss),
    mPlanningHistoryMillis(planningHistoryMillis),
    mSensorDim(sensorDim),
    mSensorLog(VectorLog(sensorDim)),
    mControlLog(VectorLog(world->getNumDofs()))
{
  int dofs = world->getNumDofs();
  mInitialPosEstimator
      = [dofs](Eigen::MatrixXs /* sensors */, long /* time */) {
          return Eigen::VectorXs::Zero(dofs);
        };

  std::shared_ptr<IPOptOptimizer> ipoptOptimizer
      = std::make_shared<IPOptOptimizer>();
  ipoptOptimizer->setCheckDerivatives(false);
  ipoptOptimizer->setSuppressOutput(true);
  ipoptOptimizer->setTolerance(1e-9);
  ipoptOptimizer->setIterationLimit(20);
  ipoptOptimizer->setRecordFullDebugInfo(false);
  ipoptOptimizer->setRecordIterations(false);
  ipoptOptimizer->setLBFGSHistoryLength(5);
  ipoptOptimizer->setSilenceOutput(true);
  mOptimizer = ipoptOptimizer;
}

/// This updates the loss function that we're going to move in real time to
/// minimize. This can happen quite frequently, for example if our loss
/// function is to track a mouse pointer in a simulated environment, we may
/// reset the loss function every time the mouse moves.
void SSID::setLoss(std::shared_ptr<trajectory::LossFn> loss)
{
  mLoss = loss;
}

/// This sets the optimizer that MPC will use. This will override the default
/// optimizer. This should be called before start().
void SSID::setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer)
{
  mOptimizer = optimizer;
}

/// This returns the current optimizer that MPC is using
std::shared_ptr<trajectory::Optimizer> SSID::getOptimizer()
{
  return mOptimizer;
}

/// This sets the problem that MPC will use. This will override the default
/// problem. This should be called before start().
void SSID::setProblem(std::shared_ptr<trajectory::Problem> problem)
{
  mProblem = problem;
}

/// This registers a function that can be used to estimate the initial state
/// for the inference system from recent sensor history and the timestamp
void SSID::setInitialPosEstimator(
    std::function<Eigen::VectorXs(Eigen::MatrixXs, long)> initialPosEstimator)
{
  mInitialPosEstimator = initialPosEstimator;
}

/// This returns the current problem definition that MPC is using
std::shared_ptr<trajectory::Problem> SSID::getProblem()
{
  return mProblem;
}

/// This logs that the sensor output is a specific vector now
void SSID::registerSensorsNow(Eigen::VectorXs sensors)
{
  return registerSensors(timeSinceEpochMillis(), sensors);
}

/// This logs that the controls are a specific vector now
void SSID::registerControlsNow(Eigen::VectorXs controls)
{
  return registerControls(timeSinceEpochMillis(), controls);
}

/// This logs that the sensor output was a specific vector at a specific
/// moment
void SSID::registerSensors(long now, Eigen::VectorXs sensors)
{
  mSensorLog.record(now, sensors);
}

/// This logs that our controls were this value at this time
void SSID::registerControls(long now, Eigen::VectorXs controls)
{
  mControlLog.record(now, controls);
}

/// This starts our main thread and begins running optimizations
void SSID::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mOptimizationThread = std::thread(&SSID::optimizationThreadLoop, this);
}

/// This stops our main thread, waits for it to finish, and then returns
void SSID::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

/// This runs inference to find mutable values, starting at `startTime`
void SSID::runInference(long startTime)
{
  long startComputeWallTime = timeSinceEpochMillis();

  int millisPerStep = static_cast<int>(ceil(mWorld->getTimeStep() * 1000.0));
  int steps = static_cast<int>(
      ceil(static_cast<s_t>(mPlanningHistoryMillis) / millisPerStep));

  if (!mProblem)
  {
    std::shared_ptr<MultiShot> multishot
        = std::make_shared<MultiShot>(mWorld, *mLoss.get(), steps, 10, true);
    multishot->setParallelOperationsEnabled(true);
    mProblem = multishot;
  }

  // Every turn, we need to pin all the forces

  Eigen::MatrixXs forceHistory = mControlLog.getValues(
      startTime - mPlanningHistoryMillis, steps, millisPerStep);
  for (int i = 0; i < steps; i++)
  {
    mProblem->pinForce(i, forceHistory.col(i));
  }

  // We also need to set all the sensor history into metadata

  Eigen::MatrixXs sensorHistory = mSensorLog.getValues(
      startTime - mPlanningHistoryMillis, steps, millisPerStep);
  mProblem->setMetadata("forces", forceHistory);
  mProblem->setMetadata("sensors", sensorHistory);

  mProblem->setStartPos(mInitialPosEstimator(sensorHistory, startTime));

  // Then actually run the optimization

  mSolution = mOptimizer->optimize(mProblem.get());

  long computeDurationWallTime = timeSinceEpochMillis() - startComputeWallTime;

  const trajectory::TrajectoryRollout* cache
      = mProblem->getRolloutCache(mWorld);

  Eigen::VectorXs pos = cache->getPosesConst().col(steps - 1);
  Eigen::VectorXs vel = cache->getVelsConst().col(steps - 1);
  Eigen::VectorXs mass = mWorld->getMasses();

  for (auto listener : mInferListeners)
  {
    listener(startTime, pos, vel, mass, computeDurationWallTime);
  }
}

/// This registers a listener to get called when we finish replanning
void SSID::registerInferListener(
    std::function<
        void(long, Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs, long)>
        inferListener)
{
  mInferListeners.push_back(inferListener);
}

/// This is the function for the optimization thread to run when we're live
void SSID::optimizationThreadLoop()
{
  // block signals in this thread and subsequently
  // spawned threads, so they're guaranteed to go to the server thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

  while (mRunning)
  {
    long startTime = timeSinceEpochMillis();
    if (mControlLog.availableHistoryBefore(startTime) > mPlanningHistoryMillis)
    {
      std::cout << "Running inference" << std::endl;
      runInference(startTime);
    }
    // long endTime = timeSinceEpochMillis();
  }
}

} // namespace realtime
} // namespace dart