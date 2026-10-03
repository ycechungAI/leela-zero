/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BATCHQUEUE_H_INCLUDED
#define BATCHQUEUE_H_INCLUDED

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#ifndef NDEBUG
struct batch_stats_t {
    std::atomic<size_t> single_evals{0};
    std::atomic<size_t> batch_evals{0};
};
inline batch_stats_t batch_stats;
#endif

// Device-independent batching of NN evaluation requests, shared by the GPU
// backends. Search threads call submit() and block; backend worker threads
// call pickup() to get a batch, evaluate it, write the outputs, and call
// complete() to wake the submitters.
class BatchQueue {
public:
    class Entry {
    public:
        Entry(const std::vector<float>& input, std::vector<float>& output_pol,
              std::vector<float>& output_val)
            : in(input), out_p(output_pol), out_v(output_val) {}

        // Valid until complete() is called on this entry.
        const std::vector<float>& in;
        std::vector<float>& out_p;
        std::vector<float>& out_v;

    private:
        friend class BatchQueue;
        std::mutex mutex;
        std::condition_variable cv;
        bool done{false};
    };
    using Batch = std::list<std::shared_ptr<Entry>>;

    // Search thread: queue one evaluation and block until a worker completes
    // it. Returns false if the queue was drained (the caller should abort).
    bool submit(const std::vector<float>& input, std::vector<float>& output_pol,
                std::vector<float>& output_val) {
        auto entry = std::make_shared<Entry>(input, output_pol, output_val);
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_queue.push_back(entry);
            if (m_single_eval_in_progress.load()) {
                m_waittime += 2;
            }
        }
        m_cv.notify_one();

        std::unique_lock<std::mutex> lk(entry->mutex);
        // The predicate guards against spurious wakeups returning before the
        // outputs have been written.
        entry->cv.wait(lk, [&entry]() { return entry->done; });
        return !m_draining;
    }

    // Worker thread: block until a batch is ready or the queue shuts down.
    // An empty batch means shutdown; the worker should exit.
    //
    // Batch scheduling heuristic:
    // 1) Wait up to m_waittime milliseconds for a full batch.
    // 2) If no full batch arrived, do a single eval instead.
    //
    // m_waittime prevents deadlock: a batch may never fill when evals are
    // stuck on a critical path. If a batch couldn't be formed in time, we hit
    // the critical path and do a single eval; wait 1 ms less next time. If
    // more evals arrived while a single eval was running, that was the wrong
    // call; submit() makes the next wait 2 ms longer.
    Batch pickup(const size_t max_batch) {
        Batch batch;
        size_t count = 0;

        std::unique_lock<std::mutex> lk(m_mutex);
        while (true) {
            if (!m_running) {
                return batch;
            }
            count = m_queue.size();
            if (count >= max_batch) {
                count = max_batch;
                break;
            }

            const auto timeout = !m_cv.wait_for(
                lk, std::chrono::milliseconds(m_waittime), [this, max_batch]() {
                    return !m_running || m_queue.size() >= max_batch;
                });

            if (!m_queue.empty() && timeout
                && m_single_eval_in_progress.exchange(true) == false) {
                // Waited long enough but couldn't form a batch, and no other
                // single eval is in progress: do one from this thread.
                if (m_waittime > 1) {
                    m_waittime--;
                }
                count = 1;
                break;
            }
        }

        auto end = begin(m_queue);
        std::advance(end, count);
        std::move(begin(m_queue), end, std::back_inserter(batch));
        m_queue.erase(begin(m_queue), end);

#ifndef NDEBUG
        if (count == 1) {
            batch_stats.single_evals++;
        } else {
            batch_stats.batch_evals++;
        }
#endif
        return batch;
    }

    // Worker thread: outputs for every entry in the batch have been written.
    void complete(const Batch& batch) {
        for (const auto& entry : batch) {
            mark_done(*entry);
        }
        if (batch.size() == 1) {
            m_single_eval_in_progress = false;
        }
    }

    // Wake all workers so pickup() returns an empty batch.
    void shutdown() {
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_running = false;
        }
        m_cv.notify_all();
    }

    // Wake every queued submitter; they (and any submit() that finishes while
    // draining) return false. In-flight batches still complete normally.
    void drain() {
        m_draining = true;

        Batch pending;
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            std::move(m_queue.begin(), m_queue.end(),
                      std::back_inserter(pending));
            m_queue.clear();
        }
        for (const auto& entry : pending) {
            mark_done(*entry);
        }
    }

    void resume() {
        m_draining = false;
    }

    bool empty() {
        std::unique_lock<std::mutex> lk(m_mutex);
        return m_queue.empty();
    }

private:
    static void mark_done(Entry& entry) {
        {
            std::unique_lock<std::mutex> lk(entry.mutex);
            entry.done = true;
        }
        entry.cv.notify_all();
    }

    std::mutex m_mutex;
    std::condition_variable m_cv;
    Batch m_queue;
    bool m_running{true};               // protected by m_mutex
    int m_waittime{10};                 // ms; protected by m_mutex
    std::atomic<bool> m_draining{false};
    // Set while a non-batched (single) eval is in progress.
    std::atomic<bool> m_single_eval_in_progress{false};
};

#endif
