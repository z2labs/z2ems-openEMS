/*
*	Copyright (C) 2026 Zoltan Doczi
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include <vector>
#include <unordered_map>
#include "operator.h"
#include "engine.h"
#include "engine_cuda_kernels.h"

class Operator_Ext_Excitation;
class Operator_Ext_UPML;
class Operator_Ext_Mur_ABC;
class Operator_Ext_LumpedRLC;
class Operator_Ext_LorentzMaterial;
class Operator_Ext_DebyeMaterial;

//! Operator of the CUDA engine: the basic operator (full coefficient arrays), engine on the GPU.
class Operator_CUDA : public Operator
{
public:
	static Operator_CUDA* New();
	virtual Engine* CreateEngine();
protected:
	Operator_CUDA();
	//! the coefficient set-up of the basic operator, split over the CPU threads (x ranges, as Operator_Multithread)
	virtual bool Calc_EC();
	virtual bool CalcPEC();
	unsigned int SetupThreads() const;
};

/*!
  FDTD engine on an NVIDIA GPU (--engine=cuda).

  Fields and operator coefficients live in GPU memory, the update formulas are those of the basic
  engine. The extensions run on the GPU too, in the CPU engine's priority order; supported: excitation,
  UPML, Mur ABC, lumped RLC, Drude/Lorentz materials, conducting sheets, Debye materials.
  Any other engine extension (TFSF, steady state, cylinder) stops the run with a message, use a CPU
  engine for those models.

  Probes read single field values through GetVolt/GetCurr: the first read of a cell registers it, after
  every IterateTS() all registered values are gathered in one kernel and one small copy. Full-field
  access (energy end criterion, dumps) calls SyncHost(), which copies the whole fields to the host arrays.
*/
class Engine_CUDA : public Engine
{
public:
	static Engine_CUDA* New(const Operator* op);
	virtual ~Engine_CUDA();

	virtual void Init();
	virtual void Reset();
	virtual bool IterateTS(unsigned int iterTS);
	virtual void SyncHost();
	virtual bool FastEnergy(double& volt_sq, double& curr_sq);

	//! GPU memory guard: checksums of the operator coefficients, verified at every energy check (default on)
	void SetMemoryGuard(bool on) {m_guard = on;}
	bool GetMemoryGuard() const {return m_guard;}
	//! default for new engines: -1 = from the environment (Z2_CUDA_GUARD, on unless "0"), 0 = off, 1 = on
	static int s_guard_default;

	virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const {return GetVolt(n, pos[0], pos[1], pos[2]);}
	virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const {return GetCurr(n, pos[0], pos[1], pos[2]);}
	virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value);
	virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value) {SetVolt(n, pos[0], pos[1], pos[2], value);}
	virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value);
	virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value) {SetCurr(n, pos[0], pos[1], pos[2], value);}

protected:
	Engine_CUDA(const Operator* op);

	size_t Index(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{ return ((size_t(n) * numLines[0] + x) * numLines[1] + y) * numLines[2] + z; }
	FDTD_FLOAT Probe(int which, size_t index) const;
	void Gather();
	void Fail(const char* what) const;

	z2cu_grid m_g;
	size_t m_cells;
	bool m_host_valid;

	bool m_guard;
	unsigned long long m_coef_hash[4];
	void GuardInit();
	void GuardCheck();

	// excitation
	struct Exc { unsigned int count; unsigned int *x, *y, *z; unsigned short* dir; float* amp; unsigned int* delay; };
	Exc m_exc_v, m_exc_i;
	float *m_sig_v, *m_sig_i;
	unsigned int m_sig_len;
	int m_sig_period;
	bool m_has_exc;

	std::vector<z2cu_upml> m_upml_v, m_upml_i;
	struct Mur { z2cu_mur m; unsigned int start_TS; };
	std::vector<Mur> m_mur;
	struct RLC { z2cu_rlc r; };
	std::vector<RLC> m_rlc;
	std::vector<std::vector<z2cu_lorentz> > m_lor;  // one entry per extension, one z2cu_lorentz per order
	std::vector<z2cu_debye> m_deb;

	// GPU extensions in the CPU engine's order (priority descending, equal priority: reverse creation)
	enum ExtKind {EXT_EXC, EXT_UPML, EXT_MUR, EXT_RLC, EXT_LOR, EXT_DEB};
	struct ExtStep { ExtKind kind; size_t idx; int prio; size_t created; };
	std::vector<ExtStep> m_steps;
	std::vector<void*> m_dev_buffers;          // everything allocated on the device

	// probe cache: key = which * 2^62 + linear index -> slot
	mutable std::unordered_map<unsigned long long, unsigned int> m_probe_slot;
	mutable std::vector<unsigned int> m_probe_index;
	mutable std::vector<unsigned char> m_probe_which;
	mutable std::vector<float> m_probe_value;
	mutable bool m_probe_dirty;
	unsigned int *m_d_probe_index;
	unsigned char* m_d_probe_which;
	float* m_d_probe_value;
	unsigned int m_d_probe_cap;

	void* DevCopy(const void* src, size_t bytes);
	void InitExtensionsCUDA();
	float* Upload3(float* const* src, unsigned int count);   // [3][count] from src[n][i]
	void AddLorentz(Operator_Ext_LorentzMaterial* l);
	void AddDebye(Operator_Ext_DebyeMaterial* d);
	int VoltPre(const ExtStep& s);
	int VoltPost(const ExtStep& s);
	int VoltApply(const ExtStep& s);
	int CurrPre(const ExtStep& s);
	int CurrPost(const ExtStep& s);
	int CurrApply(const ExtStep& s);
	int StepKernels();                       // one time step, all kernels
	void* m_graph;                           // the time step as a CUDA graph (NULL: not captured yet / not used)
	bool m_use_graph;                        // Z2_CUDA_NOGRAPH=1 launches the kernels one by one
};

#endif // ENGINE_CUDA_H
