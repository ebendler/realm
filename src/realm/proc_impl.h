/* Copyright 2024 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Processor/ProcessorGroup implementations for Realm

#ifndef REALM_PROC_IMPL_H
#define REALM_PROC_IMPL_H

#include "realm/processor.h"
#include "realm/id.h"

#include "realm/atomics.h"
#include "realm/network.h"
#include "realm/operation.h"
#include "realm/profiling.h"
#include "realm/sampling.h"

#include "realm/runtime_impl.h"
#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"

#include "realm/tasks.h"
#include "realm/threads.h"
#include "realm/codedesc.h"

namespace Realm {

    class ProcessorGroupImpl;

    namespace ThreadLocal {
      // if nonzero, prevents application thread from yielding execution
      //  resources on an Event wait
      extern thread_local int scheduler_lock;
    };

    class ProcessorImpl {
    public:
      ProcessorImpl(RuntimeImpl *runtime_impl, Processor _me, Processor::Kind _kind,
                    int _num_cores = 1);

      virtual ~ProcessorImpl(void);

      virtual void enqueue_task(Task *task) = 0;
      virtual void enqueue_tasks(Task::TaskList& tasks, size_t num_tasks) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event,
			      GenEventImpl *finish_event,
			      EventImpl::gen_t finish_gen,
                              int priority) = 0;

      // starts worker threads and performs any per-processor initialization
      virtual void start_threads(void);

      // blocks until things are cleaned up
      virtual void shutdown(void);

      virtual void add_to_group(ProcessorGroupImpl *group) = 0;

      virtual void remove_from_group(ProcessorGroupImpl *group) = 0;

      virtual bool register_task(Processor::TaskFuncID func_id,
				 CodeDescriptor& codedesc,
				 const ByteArrayRef& user_data);

      // runs an internal Realm operation on this processor
      virtual void add_internal_task(InternalTask *task);

      GenEventImpl *create_genevent();
      void free_genevent(GenEventImpl *);

      // A helper function to get the kind of a processor when we only have the processor
      // ID
      static Processor::Kind get_processor_kind(RuntimeImpl *runtime_impl,
                                                Processor processor);

    protected:
      friend class Task;

      // Event free list cache variables
      LocalEventTableAllocator::FreeList free_local_events;

      virtual void execute_task(Processor::TaskFuncID func_id,
				const ByteArrayRef& task_args);

      struct DeferredSpawnCache {
        static const size_t MAX_ENTRIES = 4;
        Mutex mutex;
        EventImpl *events[MAX_ENTRIES];
        EventImpl::gen_t generations[MAX_ENTRIES];
        Task *tasks[MAX_ENTRIES];
        size_t counts[MAX_ENTRIES];
        size_t ages[MAX_ENTRIES];
        size_t current_age;

        void clear() {
          current_age = 0;
          memset(events, 0, sizeof(events));
          memset(tasks, 0, sizeof(tasks));
          memset(counts, 0, sizeof(counts));
          memset(ages, 0, sizeof(ages));
          memset(generations, 0, sizeof(generations));
        }

        void flush() {
          for (size_t i = 0; i < MAX_ENTRIES; i++) {
            if (tasks[i] != nullptr) {
              tasks[i]->remove_reference();
            }
          }
          clear();
        }
      };

      // helper function for spawn implementations
      void enqueue_or_defer_task(Task *task, Event start_event,
				 DeferredSpawnCache *cache);

    public:
      Processor me;
      Processor::Kind kind;
      int num_cores;
    }; 

    // generic local task processor - subclasses must create and configure a task
    // scheduler and pass in with the set_scheduler() method
    class LocalTaskProcessor : public ProcessorImpl {
    public:
      LocalTaskProcessor(RuntimeImpl *runtime_impl, Processor _me, Processor::Kind _kind,
                         int num_cores = 1);
      virtual ~LocalTaskProcessor(void);

      virtual void enqueue_task(Task *task);
      virtual void enqueue_tasks(Task::TaskList& tasks, size_t num_tasks);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event,
			      GenEventImpl *finish_event,
			      EventImpl::gen_t finish_gen,
                              int priority);

      virtual bool register_task(Processor::TaskFuncID func_id,
				 CodeDescriptor& codedesc,
				 const ByteArrayRef& user_data);

      // starts worker threads and performs any per-processor initialization
      virtual void start_threads(void);

      // blocks until things are cleaned up
      virtual void shutdown(void);

      virtual void add_to_group(ProcessorGroupImpl *group);

      virtual void remove_from_group(ProcessorGroupImpl *group);

      // runs an internal Realm operation on this processor
      virtual void add_internal_task(InternalTask *task);

    protected:
      void set_scheduler(ThreadedTaskScheduler *_sched);

      ThreadedTaskScheduler *sched;
      TaskQueue task_queue; // ready tasks
      ProfilingGauges::AbsoluteRangeGauge<int> ready_task_count;
      DeferredSpawnCache deferred_spawn_cache;

      struct TaskTableEntry {
	Processor::TaskFuncPtr fnptr;
	ByteArray user_data;
      };

      RWLock task_table_mutex;
      std::map<Processor::TaskFuncID, TaskTableEntry> task_table;

      virtual void execute_task(Processor::TaskFuncID func_id,
				const ByteArrayRef& task_args);
    };

    // three simple subclasses for:
    // a) "CPU" processors, which request a dedicated core and use user threads
    //      when possible
    // b) "utility" processors, which also use user threads but share cores with
    //      other runtime threads
    // c) "IO" processors, which use kernel threads so that blocking IO calls
    //      are permitted
    //
    // each of these is implemented just by supplying the right kind of scheduler to
    //  LocalTaskProcessor in the constructor

    class LocalCPUProcessor : public LocalTaskProcessor {
    public:
      LocalCPUProcessor(RuntimeImpl *runtime_impl, Processor _me, CoreReservationSet &crs,
                        size_t _stack_size, bool _force_kthreads,
                        BackgroundWorkManager *bgwork, long long bgwork_timeslice);
      virtual ~LocalCPUProcessor(void);
    protected:
      CoreReservation *core_rsrv;
    };

    class LocalUtilityProcessor : public LocalTaskProcessor {
    public:
      LocalUtilityProcessor(RuntimeImpl *runtime_impl, Processor _me,
                            CoreReservationSet &crs, size_t _stack_size,
                            bool _force_kthreads, bool _pin_util_proc,
                            BackgroundWorkManager *bgwork, long long bgwork_timeslice);
      virtual ~LocalUtilityProcessor(void);
    protected:
      CoreReservation *core_rsrv;
    };

    class LocalIOProcessor : public LocalTaskProcessor {
    public:
      LocalIOProcessor(RuntimeImpl *runtime_impl, Processor _me, CoreReservationSet &crs,
                       size_t _stack_size, int _concurrent_io_threads);
      virtual ~LocalIOProcessor(void);
    protected:
      CoreReservation *core_rsrv;
    };

    class RemoteProcessor : public ProcessorImpl {
    public:
      RemoteProcessor(RuntimeImpl *runtime_impl, Processor _me, Processor::Kind _kind,
                      int _num_cores = 1);
      virtual ~RemoteProcessor(void);

      virtual void enqueue_task(Task *task);
      virtual void enqueue_tasks(Task::TaskList& tasks, size_t num_tasks);

      virtual void add_to_group(ProcessorGroupImpl *group);

      virtual void remove_from_group(ProcessorGroupImpl *group);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event,
			      GenEventImpl *finish_event,
			      EventImpl::gen_t finish_gen,
                              int priority);
    };

    class ProcessorGroupImpl : public ProcessorImpl {
    public:
      // TODO:: pass in runtime_impl, currently we are getting an error when we try to
      // pass in runtime_impl dynamic_table.inl:129:40: error: constructor for
      // 'Realm::DynamicTableNode<Realm::ProcessorGroupImpl, 16, Realm::UnfairMutex,
      // unsigned long long>' must explicitly initialize the member 'elems' which does not
      // have a default constructor
      //   |   DynamicTableNode<ET, _SIZE, LT, IT>::DynamicTableNode(int _level, IT
      //   _first_index, IT _last_index)
      // We need to update the DynamicTableAllocator to take in a runtime_impl
      ProcessorGroupImpl(void);

      virtual ~ProcessorGroupImpl(void);

      static const ID::ID_Types ID_TYPE = ID::ID_PROCGROUP;

      void init(ID _me, int _owner);

      static ID make_id(const ProcessorGroupImpl &dummy, int owner, ID::IDType index)
      {
        return ID::make_procgroup(owner, 0, index);
      }

      void set_group_members(span<const Processor> member_list);

      void destroy(void);

      void get_group_members(std::vector<Processor>& member_list);

      virtual void enqueue_task(Task *task);
      virtual void enqueue_tasks(Task::TaskList& tasks, size_t num_tasks);

      virtual void add_to_group(ProcessorGroupImpl *group);

      virtual void remove_from_group(ProcessorGroupImpl *group);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event,
			      GenEventImpl *finish_event,
			      EventImpl::gen_t finish_gen,
                              int priority);

    public: //protected:
      bool members_valid;
      bool members_requested;
      std::vector<ProcessorImpl *> members;
      ReservationImpl lock;
      ProcessorGroupImpl *next_free;

      void request_group_members(void);

      TaskQueue task_queue; // ready tasks
      ProfilingGauges::AbsoluteRangeGauge<int> *ready_task_count;
      DeferredSpawnCache deferred_spawn_cache;

      class DeferredDestroy : public EventWaiter {
      public:
	void defer(ProcessorGroupImpl *_pg, Event wait_on);
	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	ProcessorGroupImpl *pg;
      };
      DeferredDestroy deferred_destroy;
    };
    
    // a task registration can take a while if remote processors and/or JITs are
    //  involved
    class TaskRegistration : public Operation {
    public:
      TaskRegistration(const CodeDescriptor& _codedesc,
		       const ByteArrayRef& _userdata,
		       GenEventImpl *_finish_event,
		       EventImpl::gen_t _finish_gen,
		       const ProfilingRequestSet &_requests);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~TaskRegistration(void);

    public:
      virtual void print(std::ostream& os) const;

      CodeDescriptor codedesc;
      ByteArray userdata;
    };

    class RemoteTaskRegistration : public Operation::AsyncWorkItem {
    public:
      RemoteTaskRegistration(TaskRegistration *reg_op, int _target_node);

      virtual void request_cancellation(void);

      virtual void print(std::ostream& os) const;

    protected:
      int target_node;
    };

    // active messages
    struct RegisterTaskMessage {
      NodeID sender;
      Processor::TaskFuncID func_id;
      Processor::Kind kind;
      RemoteTaskRegistration *reg_op;

      static void handle_message(NodeID sender,const RegisterTaskMessage &msg,
				 const void *data, size_t datalen);
    };
    
    struct RegisterTaskCompleteMessage {
      RemoteTaskRegistration *reg_op;
      bool successful;

      static void handle_message(NodeID sender,const RegisterTaskCompleteMessage &msg,
				 const void *data, size_t datalen);

    };

    struct SpawnTaskMessage {
      Processor proc;
      Event finish_event;
      Processor::TaskFuncID func_id;
      size_t offset, total_bytes;

      static void handle_message(NodeID sender,const SpawnTaskMessage &msg,
				 const void *data, size_t datalen);
    };

    struct ProcGroupCreateMessage {
      ProcessorGroup pgrp;
      size_t num_members;

      static void handle_message(NodeID sender, const ProcGroupCreateMessage &msg,
				 const void *data, size_t datalen);
    };

    struct ProcGroupDestroyMessage {
      ProcessorGroup pgrp;
      Event wait_on;

      static void handle_message(NodeID sender, const ProcGroupDestroyMessage &msg,
				 const void *data, size_t datalen);
    };

    struct ProcGroupDestroyAckMessage {
      ProcessorGroup pgrp;

      static void handle_message(NodeID sender, const ProcGroupDestroyAckMessage &msg,
				 const void *data, size_t datalen);
    };

    namespace ThreadLocal {
      // Assume zero initialized
      extern thread_local Processor current_processor;
    }

}; // namespace Realm

#endif // ifndef REALM_PROC_IMPL_H
