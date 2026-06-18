// Copyright (c) 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/message_pump_default.h"

#include "base/logging.h"
#include "base/message_loop.h"
#include "base/scoped_nsautorelease_pool.h"

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
// Single-OS-thread cooperative (JSPI fiber) build: this base::Thread Run loop runs on a
// fiber, and event_.Wait() is its ONLY cooperative yield point. When work is posted
// continuously (e.g. an IPC/necko task that re-posts under load) did_work stays true and
// the loop spins via "continue" without ever reaching Wait(), so no other fiber -- not even
// the one that would end the re-post cycle -- can run, and the single thread livelocks.
// Yield cooperatively on the did_work path so the scheduler can run other fibers.
#  include <sched.h>
#  define STJ_COOP_YIELD() sched_yield()
#else
#  define STJ_COOP_YIELD() ((void)0)
#endif

#include "mozilla/BackgroundHangMonitor.h"
#include "mozilla/ProfilerLabels.h"
#include "mozilla/ProfilerThreadSleep.h"

namespace base {

MessagePumpDefault::MessagePumpDefault()
    : keep_running_(true), event_(false, false) {}

void MessagePumpDefault::Run(Delegate* delegate) {
  AUTO_PROFILER_LABEL("MessagePumpDefault::Run", OTHER);

  DCHECK(keep_running_) << "Quit must have been called outside of Run!";

  const MessageLoop* const loop = MessageLoop::current();
  mozilla::BackgroundHangMonitor hangMonitor(loop->thread_name().c_str(),
                                             loop->transient_hang_timeout(),
                                             loop->permanent_hang_timeout());

  for (;;) {
    ScopedNSAutoreleasePool autorelease_pool;

    hangMonitor.NotifyActivity();
    bool did_work = delegate->DoWork();
    if (!keep_running_) break;

    hangMonitor.NotifyActivity();
    did_work |= delegate->DoDelayedWork(&delayed_work_time_);
    if (!keep_running_) break;

    if (did_work) {
      STJ_COOP_YIELD();
      continue;
    }

    hangMonitor.NotifyActivity();
    did_work = delegate->DoIdleWork();
    if (!keep_running_) break;

    if (did_work) {
      STJ_COOP_YIELD();
      continue;
    }

    if (delayed_work_time_.is_null()) {
      hangMonitor.NotifyWait();
      AUTO_PROFILER_LABEL("MessagePumpDefault::Run:Wait", IDLE);
      event_.Wait();
    } else {
      TimeDelta delay = delayed_work_time_ - TimeTicks::Now();
      if (delay > TimeDelta()) {
        hangMonitor.NotifyWait();
        AUTO_PROFILER_LABEL("MessagePumpDefault::Run:Wait", IDLE);
        event_.TimedWait(delay);
      } else {
        // It looks like delayed_work_time_ indicates a time in the past, so we
        // need to call DoDelayedWork now.
        delayed_work_time_ = TimeTicks();
      }
    }
    // Since event_ is auto-reset, we don't need to do anything special here
    // other than service each delegate method.
  }

  keep_running_ = true;
}

void MessagePumpDefault::Quit() { keep_running_ = false; }

void MessagePumpDefault::ScheduleWork() {
  // Since this can be called on any thread, we need to ensure that our Run
  // loop wakes up.
  event_.Signal();
}

void MessagePumpDefault::ScheduleDelayedWork(
    const TimeTicks& delayed_work_time) {
  // We know that we can't be blocked on Wait right now since this method can
  // only be called on the same thread as Run, so we only need to update our
  // record of how long to sleep when we do sleep.
  delayed_work_time_ = delayed_work_time;
}

}  // namespace base
