/*
*	Copyright (C) 2010 Sebastian Held (sebastian.held@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_MULTITHREAD_H
#define ENGINE_MULTITHREAD_H

#include "operator_multithread.h"
#include "engine_sse_compressed.h"

#include <boost/thread.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define OPENEMS_CPU_RELAX() _mm_pause()
#else
#define OPENEMS_CPU_RELAX() ((void)0)
#endif
#include <boost/fusion/include/list.hpp>
#include <boost/fusion/container/list/list_fwd.hpp>
#include <boost/fusion/include/list_fwd.hpp>

#include "tools/useful.h"
#if defined(_WIN32) && !defined(__GNUC__)
#include <Winsock2.h> // for struct timeval
#else
#include <sys/time.h>
#endif


class Engine_Multithread;

namespace NS_Engine_Multithread
{

class DBG   // debug
{
public:
	DBG() {}
	~DBG() { std::cout << os.str();}
	std::ostringstream& cout() {return os;}
protected:
	std::ostringstream os;
};

class Timer   //debug
{
public:
	Timer() {gettimeofday(&t1,NULL);}
	double elapsed() {gettimeofday(&t2,NULL); return (t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)*1e-6;}
protected:
	timeval t1,t2;
};

//! Hybrid spin/blocking barrier for the per-timestep synchronisation of the worker threads.
/*!
  boost::barrier is built on a mutex and a condition variable, so every wait() ends up in the
  kernel. The FDTD loop passes several barriers per time step, which dominates the run time on
  small grids. This barrier first spins on an atomic generation counter (cheap when the other
  threads arrive within microseconds, the common case on small grids) and then blocks on a
  condition variable. Blocking matters on large, memory-bound grids: threads that busy-wait for a
  slower neighbour would otherwise keep their cores at full power (lowering the turbo clock of the
  working cores) and compete for shared core resources.
*/
class SpinBarrier
{
public:
	explicit SpinBarrier(unsigned int count, unsigned int spins = 2000)
		: m_count(count), m_spins(spins), m_waiting(count), m_generation(0) {}

	void wait()
	{
		const unsigned int gen = m_generation.load(std::memory_order_acquire);
		if (m_waiting.fetch_sub(1, std::memory_order_acq_rel) == 1)
		{
			// last thread to arrive: re-arm the barrier, then release the others
			m_waiting.store(m_count, std::memory_order_relaxed);
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_generation.fetch_add(1, std::memory_order_release);
			}
			m_cond.notify_all();
			return;
		}
		for (unsigned int n = 0; n < m_spins; ++n)
		{
			if (m_generation.load(std::memory_order_acquire) != gen)
				return;
			OPENEMS_CPU_RELAX();
		}
		std::unique_lock<std::mutex> lock(m_mutex);
		m_cond.wait(lock, [&]{ return m_generation.load(std::memory_order_acquire) != gen; });
	}

protected:
	const unsigned int m_count;
	const unsigned int m_spins;
	alignas(64) std::atomic<unsigned int> m_waiting;
	alignas(64) std::atomic<unsigned int> m_generation;
	std::mutex m_mutex;
	std::condition_variable m_cond;
};

class thread
{
public:
	thread( Engine_Multithread* ptr, unsigned int start, unsigned int stop, unsigned int stop_h, unsigned int threadID );
	void operator()();

protected:
	unsigned int m_start, m_stop, m_stop_h, m_threadID;
	Engine_Multithread *m_enginePtr;
};
} // namespace


class Engine_Multithread : public Engine_SSE_Compressed
{
	friend class NS_Engine_Multithread::thread;
	friend class Engine_CylinderMultiGrid;
public:
	static Engine_Multithread* New(const Operator_Multithread* op, unsigned int numThreads = 0);
	virtual ~Engine_Multithread();

	virtual void setNumThreads( unsigned int numThreads );
	virtual void Init();
	virtual void Reset();
	virtual void NextInterval(float curr_speed);

	//! Iterate \a iterTS number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	virtual void DoPreVoltageUpdates()  {throw std::runtime_error("Engine_Multithread::DoPreVoltageUpdates without thread ID is invalid");}
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {throw std::runtime_error("Engine_Multithread::DoPostVoltageUpdates without thread ID is invalid");}
	virtual void DoPostVoltageUpdates(int threadID);
	virtual void Apply2Voltages()       {throw std::runtime_error("Engine_Multithread::Apply2Voltages without thread ID is invalid");}
	virtual void Apply2Voltages(int threadID);

	virtual void DoPreCurrentUpdates()  {throw std::runtime_error("Engine_Multithread::DoPreCurrentUpdates without thread ID is invalid");}
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() {throw std::runtime_error("Engine_Multithread::DoPostCurrentUpdates without thread ID is invalid");}
	virtual void DoPostCurrentUpdates(int threadID);
	virtual void Apply2Current()        {throw std::runtime_error("Engine_Multithread::Apply2Current without thread ID is invalid");}
	virtual void Apply2Current(int threadID);

protected:
	Engine_Multithread(const Operator_Multithread* op);
	void changeNumThreads(unsigned int numThreads);
	const Operator_Multithread* m_Op_MT;
	boost::thread_group *m_thread_group;
	boost::barrier *m_startBarrier, *m_stopBarrier;
	NS_Engine_Multithread::SpinBarrier *m_IterateBarrier;

	//! Rebuild m_ActiveExts from the no-op state the extensions recorded during the first time step.
	void UpdateActiveExtensions();
	//! Run one hook phase of all extensions that do work in it, with one barrier behind each of them.
	template <typename Hook> void RunExtensionPhase(int phaseIdx, Hook hook, bool reverse, int threadID);
	//! Per phase (see Engine_Extension::Phase, bit index) the extensions that have to be executed.
	std::vector<Engine_Extension*> m_ActiveExts[6];
	bool m_ActiveExtsValid;
	volatile unsigned int m_iterTS;
	unsigned int m_numThreads; //!< number of worker threads
	unsigned int m_max_numThreads; //!< max. number of worker threads
	volatile bool m_stopThreads;
	bool m_opt_speed;
	float m_last_speed;

#ifdef ENABLE_DEBUG_TIME
	std::map<boost::thread::id, std::vector<double> > m_timer_list;
#endif
};

#endif // ENGINE_MULTITHREAD_H
