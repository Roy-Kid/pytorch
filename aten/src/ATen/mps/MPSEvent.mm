//  Copyright © 2023 Apple Inc.

#include <ATen/mps/MPSEvent.h>

namespace at::mps {

MPSEvent::MPSEvent(id_t ID, MPSStream* stream, bool enable_timing)
    : m_id(ID), m_enable_timing(enable_timing), m_stream(stream), m_event([stream->device() newSharedEvent]) {}

MPSEvent::~MPSEvent() {
  if (m_event) {
    [m_event release];
    m_event = nil;
  }
}

void MPSEvent::recordLocked(bool syncEvent) {
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  ++m_signalCounter;
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  if (m_enable_timing) {
    uint64_t timingGeneration;
    {
      std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
      timingGeneration = ++m_timing_generation;
    }
    // Public timing events commit immediately after the signal, making the
    // command buffer's GPU end time the timestamp for this event boundary.
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      notifyCpuSync(timingGeneration, cb.GPUEndTime);
    }];
  }
  [commandBuffer encodeSignalEvent:m_event value:m_signalCounter];
  m_recorded.store(true);
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
}

bool MPSEvent::waitLocked(bool syncEvent) {
  // check if event is not recorded yet
  if (m_event.signaledValue >= m_signalCounter) {
    return false;
  }
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  [commandBuffer encodeWaitForEvent:m_event value:m_signalCounter];
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
  return true;
}

void MPSEvent::record(bool needsLock, bool syncEvent) {
  if (!needsLock) {
    recordLocked(syncEvent);
    return;
  }
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      recordLocked(syncEvent);
    }
  });
}

bool MPSEvent::wait(bool needsLock, bool syncEvent) {
  __block bool waited = false;
  if (!needsLock) {
    return waitLocked(syncEvent);
  }
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      waited = waitLocked(syncEvent);
    }
  });
  return waited;
}

void MPSEvent::notifyCpuSync(uint64_t timingGeneration, double completionTime) {
  std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
  if (timingGeneration >= m_timing_completed) {
    m_completion_time = completionTime;
    m_timing_completed = timingGeneration;
  }
  m_cpu_sync_cv.notify_all();
}

double MPSEvent::waitForCpuSync() {
  std::unique_lock<std::mutex> lock(m_cpu_sync_mutex);
  m_cpu_sync_cv.wait(lock, [&] { return m_timing_completed >= m_timing_generation; });
  return m_completion_time;
}

bool MPSEvent::synchronize() {
  if (!m_recorded.load() || m_event.signaledValue >= m_signalCounter) {
    return false;
  }
  TORCH_CHECK([m_event waitUntilSignaledValue:m_signalCounter timeoutMS:UINT64_MAX], "Failed to synchronize MPS event");
  return true;
}

bool MPSEvent::query() const {
  // return false if not recorded or signaled yet
  return m_recorded.load() && (m_event.signaledValue >= m_signalCounter);
}

void MPSEvent::reset(MPSStream* stream, bool enable_timing) {
  if (stream != m_stream) {
    m_signalCounter = 0;
    m_event.signaledValue = 0;
    m_stream = stream;
  }
  {
    std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
    m_completion_time = 0.0;
  }
  m_enable_timing = enable_timing;
  m_recorded.store(false);
};

//-----------------------------------------------------------------
//  MPSEventPtrTarget
//-----------------------------------------------------------------

MPSEventPtrTarget::~MPSEventPtrTarget() {
  m_pool->returnEventToPool(m_event);
}

//-----------------------------------------------------------------
//  MPSEventPool
//-----------------------------------------------------------------

MPSEventPool::MPSEventPool(MPSStream* default_stream) : m_default_stream(default_stream) {}

void MPSEventPool::returnEventToPool(MPSEvent* event) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  m_pool.push(std::unique_ptr<MPSEvent>(event));
}

MPSEventPool::~MPSEventPool() {
  emptyCache();
}

MPSEventPtr MPSEventPool::acquireEvent(bool enable_timing, MPSStream* stream) {
  if (!stream) {
    stream = m_default_stream;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_pool.empty()) {
      auto event = m_pool.top().release();
      m_pool.pop();
      event->reset(stream, enable_timing);
      return c10::make_intrusive<MPSEventPtrTarget>(event, this);
    }
  }
  auto new_event = std::make_unique<MPSEvent>(++m_event_counter, stream, enable_timing);
  return c10::make_intrusive<MPSEventPtrTarget>(new_event.release(), this);
}

void MPSEventPool::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  while (!m_pool.empty()) {
    m_pool.pop();
  }
}

id_t MPSEventPool::acquireEvent(bool enable_timing) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEventPtr event = acquireEvent(enable_timing, nullptr);
  TORCH_INTERNAL_ASSERT(event);
  id_t event_id = event->getID();
  m_in_use_events.emplace(event_id, std::move(event));
  return event_id;
}

void MPSEventPool::releaseEvent(id_t event_id) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  // returns the event back to the MPSEventPool
  m_in_use_events.erase(event_id);
}

void MPSEventPool::recordEvent(id_t event_id, bool syncEvent) {
  MPSEvent* event = getInUseEvent(event_id);
  event->record(/*needsLock*/ true, syncEvent);
}

void MPSEventPool::waitForEvent(id_t event_id, bool syncEvent) {
  MPSEvent* event = getInUseEvent(event_id);
  event->wait(/*needsLock*/ true, syncEvent);
}

void MPSEventPool::synchronizeEvent(id_t event_id) {
  MPSEvent* event = getInUseEvent(event_id);
  event->synchronize();
}

bool MPSEventPool::queryEvent(id_t event_id) {
  MPSEvent* event = getInUseEvent(event_id);
  return event->query();
}

double MPSEventPool::elapsedTime(id_t start_event_id, id_t end_event_id) {
  // First make sure the command buffers containing both events have completed.
  dispatch_sync(m_default_stream->queue(), ^() {
    m_default_stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEvent* start_event = getInUseEvent(start_event_id, false);
  MPSEvent* end_event = getInUseEvent(end_event_id, false);
  TORCH_CHECK(start_event->isTimingEnabled() && end_event->isTimingEnabled(),
              "Events were not created with argument 'enable_timing=True'");
  TORCH_CHECK(start_event->isRecorded() && end_event->isRecorded(),
              "Both events must be recorded before calculating elapsed time");
  const double start_time = start_event->waitForCpuSync();
  const double end_time = end_event->waitForCpuSync();

  TORCH_CHECK(
      end_time >= start_time, "End event ", end_event_id, " was not recorded after start event ", start_event_id);
  return (end_time - start_time) * 1e3;
}

MPSEvent* MPSEventPool::getInUseEvent(id_t event_id, bool locked) {
  if (locked) {
    m_mutex.lock();
  }
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  MPSEvent* event = m_in_use_events[event_id]->get();
  if (locked) {
    m_mutex.unlock();
  }
  return event;
}

std::shared_ptr<MPSEventPool> getMPSEventPool() {
  static std::shared_ptr<MPSEventPool> event_pool = std::make_shared<MPSEventPool>(getDefaultMPSStream());
  return event_pool;
}

} // namespace at::mps
