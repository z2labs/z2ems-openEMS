/*
*	Copyright (C) 2026 Zoltan Doczi
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#include <iostream>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include "engine_cuda.h"
#include <cmath>
#include "extensions/operator_extension.h"
#include "extensions/engine_extension.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/operator_ext_mur_abc.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/operator_ext_lorentzmaterial.h"
#include "extensions/operator_ext_debyematerial.h"
#include <algorithm>
#include "excitation.h"

using std::cout;
using std::cerr;
using std::endl;

// ---------------------------------------------------------------- operator
Operator_CUDA* Operator_CUDA::New()
{
	cout << "Create FDTD operator (CUDA)" << endl;
	Operator_CUDA* op = new Operator_CUDA();
	op->Init();
	return op;
}

Operator_CUDA::Operator_CUDA() : Operator()
{
}

unsigned int Operator_CUDA::SetupThreads() const
{
	unsigned int n = std::thread::hardware_concurrency();
	return std::max(1u, std::min(n ? n : 1u, numLines[0] / 2));
}

bool Operator_CUDA::Calc_EC()
{
	if (CSX == NULL)
	{
		cerr << "Operator_CUDA::Calc_EC: CSX not given or invalid!!!" << endl;
		return false;
	}
	MainOp->SetPos(0, 0, 0);
	unsigned int nt = SetupThreads();
	std::vector<std::thread> th;
	for (unsigned int t = 0; t < nt; ++t)
	{
		unsigned int a = numLines[0] * t / nt, b = numLines[0] * (t + 1) / nt;
		if (b > a)
			th.emplace_back([this, a, b]() {Calc_EC_Range(a, b - 1);});
	}
	for (auto& t : th)
		t.join();
	return true;
}

bool Operator_CUDA::CalcPEC()
{
	unsigned int nt = SetupThreads();
	std::vector<unsigned int> cnt(6 * nt, 0);
	std::vector<std::thread> th;
	for (unsigned int t = 0; t < nt; ++t)
	{
		unsigned int a = numLines[0] * t / nt, b = numLines[0] * (t + 1) / nt;
		if (b > a)
			th.emplace_back([this, a, b, &cnt, t]() {CalcPEC_Range(a, b - 1, &cnt[6 * t]);});
	}
	for (auto& t : th)
		t.join();
	for (int n = 0; n < 3; ++n)
	{
		m_Nr_PEC[n] = 0;
		m_Nr_ConformalPEC[n] = 0;
		for (unsigned int t = 0; t < nt; ++t)
		{
			m_Nr_PEC[n] += cnt[6 * t + n];
			m_Nr_ConformalPEC[n] += cnt[6 * t + 3 + n];
		}
	}
	CalcPEC_Curves();
	return true;
}

Engine* Operator_CUDA::CreateEngine()
{
	m_Engine = Engine_CUDA::New(this);
	return m_Engine;
}

// ---------------------------------------------------------------- engine
Engine_CUDA* Engine_CUDA::New(const Operator* op)
{
	char name[256];
	size_t mem = 0;
	int ma = 0, mi = 0;
	if (z2cu_device_info(name, sizeof(name), &mem, &ma, &mi))
		throw std::runtime_error(std::string("CUDA engine: no usable GPU: ") + z2cu_last_error());
	cout << "Create FDTD engine (CUDA, " << name << ", sm_" << ma << mi << ", " << mem / 1048576 << " MiB)" << endl;
	Engine_CUDA* e = new Engine_CUDA(op);
	e->Init();
	return e;
}

Engine_CUDA::Engine_CUDA(const Operator* op) : Engine(op)
{
	m_type = BASIC;           // host-side field layout is the basic engine's
	std::memset(&m_g, 0, sizeof(m_g));
	std::memset(&m_exc_v, 0, sizeof(m_exc_v));
	std::memset(&m_exc_i, 0, sizeof(m_exc_i));
	m_sig_v = m_sig_i = NULL;
	m_sig_len = 0;
	m_sig_period = 0;
	m_has_exc = false;
	m_cells = 0;
	m_host_valid = true;
	m_probe_dirty = false;
	m_d_probe_index = NULL;
	m_d_probe_which = NULL;
	m_d_probe_value = NULL;
	m_d_probe_cap = 0;
	m_graph = NULL;
	const char* ng = getenv("Z2_CUDA_NOGRAPH");
	m_use_graph = !(ng && *ng && *ng != '0');
	const char* gd = getenv("Z2_CUDA_GUARD");
	m_guard = s_guard_default < 0 ? !(gd && *gd && *gd == '0') : (s_guard_default != 0);
	for (int k = 0; k < 4; ++k)
		m_coef_hash[k] = 0;
}

int Engine_CUDA::s_guard_default = -1;

Engine_CUDA::~Engine_CUDA()
{
	this->Reset();
}

void Engine_CUDA::Fail(const char* what) const
{
	throw std::runtime_error(std::string("CUDA engine: ") + what + ": " + z2cu_last_error());
}

void* Engine_CUDA::DevCopy(const void* src, size_t bytes)
{
	void* d = z2cu_alloc(bytes);
	if (!d)
		Fail("out of GPU memory");
	if (src && z2cu_upload(d, src, bytes))
		Fail("upload");
	m_dev_buffers.push_back(d);
	return d;
}

void Engine_CUDA::Init()
{
	numTS = 0;
	if (z2cu_init())
		Fail("stream");
	// host arrays: kept for full-field access (energy, dumps) after SyncHost()
	volt_ptr = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("volt", numLines);
	curr_ptr = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("curr", numLines);
	m_cells = size_t(numLines[0]) * numLines[1] * numLines[2];
	size_t bytes = 3 * m_cells * sizeof(float);
	m_g.nx = numLines[0];
	m_g.ny = numLines[1];
	m_g.nz = numLines[2];
	m_g.volt = (float*)DevCopy(NULL, bytes);
	m_g.curr = (float*)DevCopy(NULL, bytes);
	if (z2cu_memset(m_g.volt, 0, bytes) || z2cu_memset(m_g.curr, 0, bytes))
		Fail("memset");
	m_g.vv = (const float*)DevCopy(Op->vv_ptr->data(), bytes);
	m_g.vi = (const float*)DevCopy(Op->vi_ptr->data(), bytes);
	m_g.ii = (const float*)DevCopy(Op->ii_ptr->data(), bytes);
	m_g.iv = (const float*)DevCopy(Op->iv_ptr->data(), bytes);
	cout << "CUDA engine: " << m_cells / 1e6 << " MCells, " << 6.0 * bytes / 1048576 << " MiB of fields and coefficients on the GPU" << endl;
	InitExtensionsCUDA();
	GuardInit();
	m_host_valid = true;
}

void Engine_CUDA::GuardInit()
{
	if (!m_guard)
		return;
	const float* c[4] = {m_g.vv, m_g.vi, m_g.ii, m_g.iv};
	for (int k = 0; k < 4; ++k)
		if (z2cu_hash(c[k], 3 * m_cells, &m_coef_hash[k]))
			Fail("memory guard");
	cout << "CUDA engine: memory guard on (coefficient checksums verified at every energy check; Z2_CUDA_GUARD=0 or --gpu-guard=0 turns it off)" << endl;
}

void Engine_CUDA::GuardCheck()
{
	// The coefficients never change during a run: a different checksum is a GPU memory error (a bit flip in
	// VRAM or a bad write), which would silently corrupt the result or blow the run up a few hundred
	// steps later. Stop here with a clear message, so a runner can restart the simulation.
	if (!m_guard)
		return;
	static const char* names[4] = {"vv", "vi", "ii", "iv"};
	const float* c[4] = {m_g.vv, m_g.vi, m_g.ii, m_g.iv};
	for (int k = 0; k < 4; ++k)
	{
		unsigned long long h = 0;
		if (z2cu_hash(c[k], 3 * m_cells, &h))
			Fail("memory guard");
		if (h != m_coef_hash[k])
		{
			cerr << "CUDA engine: MEMORY GUARD FAILED: coefficient array " << names[k] << " changed on the GPU at timestep " << numTS
			     << " (GPU memory error; the result is not trustworthy, restart the simulation)" << endl;
			throw std::runtime_error("CUDA engine: GPU memory error detected by the memory guard");
		}
	}
}

void Engine_CUDA::InitExtensionsCUDA()
{
	Operator_Ext_Excitation* exc = NULL;
	for (size_t n = 0; n < Op->GetNumberOfExtentions(); ++n)
	{
		Operator_Extension* oe = Op->GetExtension(n);
		Engine_Extension* probe = oe->CreateEngineExtention();
		if (!probe)
			continue;                 // operator-only extension: already in the coefficients
		ExtStep step;
		step.prio = probe->GetPriority();
		step.created = n;
		delete probe;
		if (Operator_Ext_Excitation* e = dynamic_cast<Operator_Ext_Excitation*>(oe))
		{
			exc = e;
			step.kind = EXT_EXC;
			step.idx = 0;
			m_steps.push_back(step);
			continue;
		}
		if (Operator_Ext_DebyeMaterial* d = dynamic_cast<Operator_Ext_DebyeMaterial*>(oe))
		{
			step.kind = EXT_DEB;
			step.idx = m_deb.size();
			AddDebye(d);
			m_steps.push_back(step);
			continue;
		}
		if (Operator_Ext_LorentzMaterial* l = dynamic_cast<Operator_Ext_LorentzMaterial*>(oe))   // also conducting sheets
		{
			step.kind = EXT_LOR;
			step.idx = m_lor.size();
			AddLorentz(l);
			m_steps.push_back(step);
			continue;
		}
		if (Operator_Ext_UPML* u = dynamic_cast<Operator_Ext_UPML*>(oe))
		{
			z2cu_upml v, i;
			for (int k = 0; k < 3; ++k)
			{
				v.start[k] = i.start[k] = u->m_StartPos[k];
				v.n[k] = i.n[k] = u->m_numLines[k];
			}
			size_t b = 3 * size_t(u->m_numLines[0]) * u->m_numLines[1] * u->m_numLines[2] * sizeof(float);
			v.c = (const float*)DevCopy(u->vv.data(), b);
			v.cfo = (const float*)DevCopy(u->vvfo.data(), b);
			v.cfn = (const float*)DevCopy(u->vvfn.data(), b);
			v.flux = (float*)DevCopy(NULL, b);
			i.c = (const float*)DevCopy(u->ii.data(), b);
			i.cfo = (const float*)DevCopy(u->iifo.data(), b);
			i.cfn = (const float*)DevCopy(u->iifn.data(), b);
			i.flux = (float*)DevCopy(NULL, b);
			if (z2cu_memset(v.flux, 0, b) || z2cu_memset(i.flux, 0, b))
				Fail("memset");
			step.kind = EXT_UPML;
			step.idx = m_upml_v.size();
			m_steps.push_back(step);
			m_upml_v.push_back(v);
			m_upml_i.push_back(i);
			continue;
		}
		if (Operator_Ext_Mur_ABC* m = dynamic_cast<Operator_Ext_Mur_ABC*>(oe))
		{
			Mur mm;
			mm.m.ny = m->m_ny;
			mm.m.nyP = m->m_nyP;
			mm.m.nyPP = m->m_nyPP;
			mm.m.line = m->m_LineNr;
			mm.m.line_shift = m->m_LineNr_Shift;
			mm.m.n0 = m->m_numLines[0];
			mm.m.n1 = m->m_numLines[1];
			size_t b = size_t(mm.m.n0) * mm.m.n1 * sizeof(float);
			mm.m.coeff_P = (const float*)DevCopy(m->m_Mur_Coeff_nyP.data(), b);
			mm.m.coeff_PP = (const float*)DevCopy(m->m_Mur_Coeff_nyPP.data(), b);
			mm.m.v_P = (float*)DevCopy(NULL, b);
			mm.m.v_PP = (float*)DevCopy(NULL, b);
			mm.start_TS = mm.m.start_TS = 0;
			step.kind = EXT_MUR;
			step.idx = m_mur.size();
			m_steps.push_back(step);
			m_mur.push_back(mm);
			continue;
		}
		if (Operator_Ext_LumpedRLC* l = dynamic_cast<Operator_Ext_LumpedRLC*>(oe))
		{
			RLC e;
			std::memset(&e, 0, sizeof(e));
			unsigned int n = l->RLC_count;
			e.r.count = n;
			if (n)
			{
				size_t fb = n * sizeof(float), ub = n * sizeof(unsigned int);
				e.r.x = (const unsigned int*)DevCopy(l->v_RLC_pos[0], ub);
				e.r.y = (const unsigned int*)DevCopy(l->v_RLC_pos[1], ub);
				e.r.z = (const unsigned int*)DevCopy(l->v_RLC_pos[2], ub);
				e.r.dir = (const int*)DevCopy(l->v_RLC_dir, n * sizeof(int));
				e.r.ilv = (const float*)DevCopy(l->v_RLC_ilv, fb);
				e.r.i2v = (const float*)DevCopy(l->v_RLC_i2v, fb);
				e.r.vv2 = (const float*)DevCopy(l->v_RLC_vv2, fb);
				e.r.vj1 = (const float*)DevCopy(l->v_RLC_vj1, fb);
				e.r.vj2 = (const float*)DevCopy(l->v_RLC_vj2, fb);
				e.r.vvd = (const float*)DevCopy(l->v_RLC_vvd, fb);
				e.r.ib0 = (const float*)DevCopy(l->v_RLC_ib0, fb);
				e.r.b1 = (const float*)DevCopy(l->v_RLC_b1, fb);
				e.r.b2 = (const float*)DevCopy(l->v_RLC_b2, fb);
				e.r.Il = (float*)DevCopy(NULL, fb);
				e.r.Vd = (float*)DevCopy(NULL, 3 * fb);
				e.r.J = (float*)DevCopy(NULL, 3 * fb);
				if (z2cu_memset(e.r.Il, 0, fb) || z2cu_memset(e.r.Vd, 0, 3 * fb) || z2cu_memset(e.r.J, 0, 3 * fb))
					Fail("memset");
				step.kind = EXT_RLC;
				step.idx = m_rlc.size();
				m_steps.push_back(step);
				m_rlc.push_back(e);
			}
			continue;
		}
		throw std::runtime_error("CUDA engine: extension '" + oe->GetExtensionName() +
		                         "' is not supported on the GPU yet; run this model with a CPU engine (e.g. --engine=multithreaded)");
	}

	if (exc && (exc->Volt_Count || exc->Curr_Count))
	{
		m_has_exc = true;
		Excitation* sig = Op->GetExcitationSignal();
		m_sig_len = sig->GetLength();
		m_sig_v = (float*)DevCopy(sig->GetVoltageSignal(), (m_sig_len + 1) * sizeof(float));
		m_sig_i = (float*)DevCopy(sig->GetCurrentSignal(), (m_sig_len + 1) * sizeof(float));
		m_sig_period = sig->GetSignalPeriod() > 0 ? int(sig->GetSignalPeriod() / sig->GetTimestep()) : 0;
		Exc* dst[2] = {&m_exc_v, &m_exc_i};
		for (int w = 0; w < 2; ++w)
		{
			Exc& d = *dst[w];
			unsigned int cnt = w ? exc->Curr_Count : exc->Volt_Count;
			unsigned int** idx = w ? exc->Curr_index : exc->Volt_index;
			d.count = cnt;
			if (!cnt)
				continue;
			d.x = (unsigned int*)DevCopy(idx[0], cnt * sizeof(unsigned int));
			d.y = (unsigned int*)DevCopy(idx[1], cnt * sizeof(unsigned int));
			d.z = (unsigned int*)DevCopy(idx[2], cnt * sizeof(unsigned int));
			d.dir = (unsigned short*)DevCopy(w ? exc->Curr_dir : exc->Volt_dir, cnt * sizeof(unsigned short));
			d.amp = (float*)DevCopy(w ? exc->Curr_amp : exc->Volt_amp, cnt * sizeof(float));
			d.delay = (unsigned int*)DevCopy(w ? exc->Curr_delay : exc->Volt_delay, cnt * sizeof(unsigned int));
		}
		// a Mur ABC on the plane of a voltage excitation starts after the excitation (as Engine_Ext_Mur_ABC)
		for (Mur& mm : m_mur)
		{
			int maxDelay = -1;
			for (unsigned int n = 0; n < exc->Volt_Count; ++n)
				if (((int)exc->Volt_dir[n] == mm.m.nyP || (int)exc->Volt_dir[n] == mm.m.nyPP) && exc->Volt_index[mm.m.ny][n] == mm.m.line)
					maxDelay = std::max(maxDelay, (int)exc->Volt_delay[n]);
			if (maxDelay >= 0)
			{
				mm.start_TS = mm.m.start_TS = maxDelay + m_sig_len + 10;
				cerr << "CUDA engine: excitation inside Mur ABC " << mm.m.ny << ", switched on at timestep " << mm.start_TS << endl;
			}
		}
	}
	if (!m_has_exc)
		m_steps.erase(std::remove_if(m_steps.begin(), m_steps.end(), [](const ExtStep& s) {return s.kind == EXT_EXC;}), m_steps.end());
	// Engine::SortExtensionByPriority(): stable sort ascending, then reversed
	std::sort(m_steps.begin(), m_steps.end(), [](const ExtStep& a, const ExtStep& b)
	          {return a.prio != b.prio ? a.prio > b.prio : a.created > b.created;});
	size_t cells_disp = 0;
	for (auto& l : m_lor)
		for (auto& o : l)
			cells_disp += o.count;
	for (auto& d : m_deb)
		cells_disp += d.count;
	cout << "CUDA engine: extensions on the GPU: excitation " << (m_has_exc ? "yes" : "no") << ", UPML regions " << m_upml_v.size()
	     << ", Mur faces " << m_mur.size() << ", lumped RLC groups " << m_rlc.size() << ", Lorentz/sheet " << m_lor.size()
	     << ", Debye " << m_deb.size() << " (" << cells_disp << " dispersive cells)" << endl;
}

float* Engine_CUDA::Upload3(float* const* src, unsigned int count)
{
	std::vector<float> h(3 * size_t(count));
	for (int n = 0; n < 3; ++n)
		std::copy(src[n], src[n] + count, h.begin() + size_t(n) * count);
	return (float*)DevCopy(h.data(), h.size() * sizeof(float));
}

void Engine_CUDA::AddLorentz(Operator_Ext_LorentzMaterial* l)
{
	std::vector<z2cu_lorentz> orders;
	for (int o = 0; o < l->m_Order; ++o)
	{
		z2cu_lorentz d;
		std::memset(&d, 0, sizeof(d));
		unsigned int cnt = l->m_LM_Count.at(o);
		bool on[2] = {l->m_volt_ADE_On[o], l->m_curr_ADE_On[o]};
		if (!cnt || !(on[0] || on[1]))
			continue;
		d.count = cnt;
		size_t ub = cnt * sizeof(unsigned int), fb = 3 * size_t(cnt) * sizeof(float);
		d.x = (const unsigned int*)DevCopy(l->m_LM_pos[o][0], ub);
		d.y = (const unsigned int*)DevCopy(l->m_LM_pos[o][1], ub);
		d.z = (const unsigned int*)DevCopy(l->m_LM_pos[o][2], ub);
		FDTD_FLOAT*** c_int[2] = {l->v_int_ADE, l->i_int_ADE};
		FDTD_FLOAT*** c_ext[2] = {l->v_ext_ADE, l->i_ext_ADE};
		FDTD_FLOAT*** c_lor[2] = {l->v_Lor_ADE, l->i_Lor_ADE};
		bool lor[2] = {l->m_volt_Lor_ADE_On[o], l->m_curr_Lor_ADE_On[o]};
		for (int s = 0; s < 2; ++s)
		{
			d.on[s] = on[s];
			if (!on[s])
				continue;
			d.c_int[s] = Upload3(c_int[s][o], cnt);
			d.c_ext[s] = Upload3(c_ext[s][o], cnt);
			d.A[s] = (float*)DevCopy(NULL, fb);
			if (z2cu_memset(d.A[s], 0, fb))
				Fail("memset");
			d.lor_on[s] = lor[s];
			if (!lor[s])
				continue;
			d.c_lor[s] = Upload3(c_lor[s][o], cnt);
			d.L[s] = (float*)DevCopy(NULL, fb);
			if (z2cu_memset(d.L[s], 0, fb))
				Fail("memset");
		}
		orders.push_back(d);
	}
	m_lor.push_back(orders);
}

void Engine_CUDA::AddDebye(Operator_Ext_DebyeMaterial* e)
{
	z2cu_debye d;
	std::memset(&d, 0, sizeof(d));
	unsigned int cnt = e->m_LM_Count.empty() ? 0 : e->m_LM_Count.at(0);
	d.poles = e->m_PoleCount;
	if (cnt && e->m_volt_ADE_On[0] && d.poles > 0)
	{
		d.count = cnt;
		size_t ub = cnt * sizeof(unsigned int), fb = 3 * size_t(cnt) * sizeof(float);
		d.x = (const unsigned int*)DevCopy(e->m_LM_pos[0][0], ub);
		d.y = (const unsigned int*)DevCopy(e->m_LM_pos[0][1], ub);
		d.z = (const unsigned int*)DevCopy(e->m_LM_pos[0][2], ub);
		std::vector<float> relax(d.poles * 3 * size_t(cnt)), drive(relax.size());
		for (int o = 0; o < d.poles; ++o)
			for (int n = 0; n < 3; ++n)
			{
				size_t off = (size_t(o) * 3 + n) * cnt;
				std::copy(e->v_relax_ADE[o][n], e->v_relax_ADE[o][n] + cnt, relax.begin() + off);
				std::copy(e->v_drive_ADE[o][n], e->v_drive_ADE[o][n] + cnt, drive.begin() + off);
			}
		d.relax = (const float*)DevCopy(relax.data(), relax.size() * sizeof(float));
		d.drive = (const float*)DevCopy(drive.data(), drive.size() * sizeof(float));
		d.solve = Upload3(e->v_solve_ADE, cnt);
		d.pole = (float*)DevCopy(NULL, relax.size() * sizeof(float));
		d.pre = (float*)DevCopy(NULL, fb);
		d.A = (float*)DevCopy(NULL, fb);
		if (z2cu_memset(d.pole, 0, relax.size() * sizeof(float)) || z2cu_memset(d.pre, 0, fb) || z2cu_memset(d.A, 0, fb))
			Fail("memset");
	}
	m_deb.push_back(d);
}

int Engine_CUDA::VoltPre(const ExtStep& s)
{
	int err = 0;
	switch (s.kind)
	{
	case EXT_UPML: return z2cu_upml_pre(m_g.volt, &m_g, &m_upml_v[s.idx]);
	case EXT_MUR:
		return z2cu_mur_step(m_g.volt, &m_g, &m_mur[s.idx].m, 0);
	case EXT_RLC:
		return z2cu_rlc_pre(&m_rlc[s.idx].r);
	case EXT_LOR:
		for (z2cu_lorentz& d : m_lor[s.idx])
			err |= z2cu_lorentz_pre(m_g.volt, &m_g, &d, 0);
		return err;
	case EXT_DEB: return z2cu_debye_pre(m_g.volt, &m_g, &m_deb[s.idx]);
	default: return 0;
	}
}

int Engine_CUDA::VoltPost(const ExtStep& s)
{
	switch (s.kind)
	{
	case EXT_UPML: return z2cu_upml_post(m_g.volt, &m_g, &m_upml_v[s.idx]);
	case EXT_MUR:
		return z2cu_mur_step(m_g.volt, &m_g, &m_mur[s.idx].m, 1);
	case EXT_DEB: return z2cu_debye_post(m_g.volt, &m_g, &m_deb[s.idx]);
	default: return 0;
	}
}

int Engine_CUDA::VoltApply(const ExtStep& s)
{
	int err = 0;
	switch (s.kind)
	{
	case EXT_EXC:
		return z2cu_excite(m_g.volt, &m_g, m_exc_v.count, m_exc_v.x, m_exc_v.y, m_exc_v.z, m_exc_v.dir, m_exc_v.amp,
		                   m_exc_v.delay, m_sig_v, m_sig_len, m_sig_period);
	case EXT_MUR:
		return z2cu_mur_step(m_g.volt, &m_g, &m_mur[s.idx].m, 2);
	case EXT_RLC:
		return z2cu_rlc_apply(m_g.volt, &m_g, &m_rlc[s.idx].r);
	case EXT_LOR:
		for (z2cu_lorentz& d : m_lor[s.idx])
			if (d.on[0])
				err |= z2cu_disp_apply(m_g.volt, &m_g, d.count, d.x, d.y, d.z, d.A[0]);
		return err;
	case EXT_DEB:
	{
		z2cu_debye& d = m_deb[s.idx];
		return z2cu_disp_apply(m_g.volt, &m_g, d.count, d.x, d.y, d.z, d.A);
	}
	default: return 0;
	}
}

int Engine_CUDA::CurrPre(const ExtStep& s)
{
	int err = 0;
	switch (s.kind)
	{
	case EXT_UPML: return z2cu_upml_pre(m_g.curr, &m_g, &m_upml_i[s.idx]);
	case EXT_LOR:
		for (z2cu_lorentz& d : m_lor[s.idx])
			err |= z2cu_lorentz_pre(m_g.curr, &m_g, &d, 1);
		return err;
	default: return 0;
	}
}

int Engine_CUDA::CurrPost(const ExtStep& s)
{
	return s.kind == EXT_UPML ? z2cu_upml_post(m_g.curr, &m_g, &m_upml_i[s.idx]) : 0;
}

int Engine_CUDA::CurrApply(const ExtStep& s)
{
	int err = 0;
	switch (s.kind)
	{
	case EXT_EXC:
		return z2cu_excite(m_g.curr, &m_g, m_exc_i.count, m_exc_i.x, m_exc_i.y, m_exc_i.z, m_exc_i.dir, m_exc_i.amp,
		                   m_exc_i.delay, m_sig_i, m_sig_len, m_sig_period);
	case EXT_LOR:
		for (z2cu_lorentz& d : m_lor[s.idx])
			if (d.on[1])
				err |= z2cu_disp_apply(m_g.curr, &m_g, d.count, d.x, d.y, d.z, d.A[1]);
		return err;
	default: return 0;
	}
}

void Engine_CUDA::Reset()
{
	z2cu_graph_free(m_graph);
	m_graph = NULL;
	for (void* p : m_dev_buffers)
		z2cu_free(p);
	m_dev_buffers.clear();
	z2cu_free(m_d_probe_index);
	z2cu_free(m_d_probe_which);
	z2cu_free(m_d_probe_value);
	m_d_probe_index = NULL;
	m_d_probe_which = NULL;
	m_d_probe_value = NULL;
	m_d_probe_cap = 0;
	m_upml_v.clear();
	m_upml_i.clear();
	m_mur.clear();
	m_rlc.clear();
	m_lor.clear();
	m_deb.clear();
	m_steps.clear();
	m_has_exc = false;
	m_probe_slot.clear();
	m_probe_index.clear();
	m_probe_which.clear();
	m_probe_value.clear();
	std::memset(&m_g, 0, sizeof(m_g));
	Engine::Reset();
}

int Engine_CUDA::StepKernels()
{
	// same phase order as Engine::IterateTS(); pre phases run the extensions in reverse order
	int err = 0;
	size_t ns = m_steps.size();
	for (size_t k = ns; k-- > 0;)
		err |= VoltPre(m_steps[k]);
	err |= z2cu_update_volt(&m_g);
	for (size_t k = 0; k < ns; ++k)
		err |= VoltPost(m_steps[k]);
	for (size_t k = 0; k < ns; ++k)
		err |= VoltApply(m_steps[k]);
	for (size_t k = ns; k-- > 0;)
		err |= CurrPre(m_steps[k]);
	err |= z2cu_update_curr(&m_g);
	for (size_t k = 0; k < ns; ++k)
		err |= CurrPost(m_steps[k]);
	for (size_t k = 0; k < ns; ++k)
		err |= CurrApply(m_steps[k]);
	err |= z2cu_next_ts();
	return err;
}

bool Engine_CUDA::IterateTS(unsigned int iterTS)
{
	if (m_use_graph && !m_graph && iterTS)
	{
		// one time step as a CUDA graph: a single launch instead of one per kernel (small PCB models
		// spend a third of their GPU time in launch gaps otherwise)
		if (z2cu_graph_begin() == 0)
		{
			int err = StepKernels();
			m_graph = z2cu_graph_end();
			if (err || !m_graph)
			{
				cerr << "CUDA engine: time step graph not available (" << z2cu_last_error() << "), launching kernels one by one" << endl;
				z2cu_graph_free(m_graph);
				m_graph = NULL;
				m_use_graph = false;
			}
		}
		else
			m_use_graph = false;
	}
	for (unsigned int it = 0; it < iterTS; ++it)
	{
		int err = m_graph ? z2cu_graph_launch(m_graph) : StepKernels();
		if (err)
			Fail("time step");
		++numTS;
	}
	m_host_valid = false;
	Gather();
	return true;
}

bool Engine_CUDA::FastEnergy(double& volt_sq, double& curr_sq)
{
	if (z2cu_energy(&m_g, &volt_sq, &curr_sq))
		Fail("energy");
	if (m_guard && !(std::isfinite(volt_sq) && std::isfinite(curr_sq)))
	{
		cerr << "CUDA engine: MEMORY GUARD FAILED: the field energy is not finite at timestep " << numTS
		     << " (the run blew up: a GPU memory error, or an unstable setup; restart the simulation)" << endl;
		throw std::runtime_error("CUDA engine: non-finite field energy");
	}
	GuardCheck();
	return true;
}

void Engine_CUDA::Gather()
{
	unsigned int n = (unsigned int)m_probe_index.size();
	if (!n)
		return;
	if (m_probe_dirty || n > m_d_probe_cap)
	{
		if (n > m_d_probe_cap)
		{
			z2cu_free(m_d_probe_index);
			z2cu_free(m_d_probe_which);
			z2cu_free(m_d_probe_value);
			m_d_probe_cap = std::max(n, 2 * m_d_probe_cap);
			m_d_probe_index = (unsigned int*)z2cu_alloc(m_d_probe_cap * sizeof(unsigned int));
			m_d_probe_which = (unsigned char*)z2cu_alloc(m_d_probe_cap);
			m_d_probe_value = (float*)z2cu_alloc(m_d_probe_cap * sizeof(float));
			if (!m_d_probe_index || !m_d_probe_which || !m_d_probe_value)
				Fail("probe buffers");
		}
		if (z2cu_upload(m_d_probe_index, m_probe_index.data(), n * sizeof(unsigned int)) ||
		    z2cu_upload(m_d_probe_which, m_probe_which.data(), n))
			Fail("probe upload");
		m_probe_dirty = false;
	}
	if (z2cu_gather(&m_g, n, m_d_probe_index, m_d_probe_which, m_d_probe_value) ||
	    z2cu_download(m_probe_value.data(), m_d_probe_value, n * sizeof(float)))
		Fail("probe gather");
}

FDTD_FLOAT Engine_CUDA::Probe(int which, size_t index) const
{
	if (m_host_valid)
		return (which ? curr_ptr : volt_ptr)->data()[index];
	unsigned long long key = ((unsigned long long)which << 62) | index;
	auto it = m_probe_slot.find(key);
	if (it != m_probe_slot.end())
		return m_probe_value[it->second];
	// first read of this cell: fetch it now, gather it after every following IterateTS()
	float v = 0;
	if (z2cu_download(&v, (which ? m_g.curr : m_g.volt) + index, sizeof(float)))
		Fail("probe read");
	m_probe_slot[key] = (unsigned int)m_probe_index.size();
	m_probe_index.push_back((unsigned int)index);
	m_probe_which.push_back((unsigned char)which);
	m_probe_value.push_back(v);
	m_probe_dirty = true;
	return v;
}

FDTD_FLOAT Engine_CUDA::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	return Probe(0, Index(n, x, y, z));
}

FDTD_FLOAT Engine_CUDA::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	return Probe(1, Index(n, x, y, z));
}

void Engine_CUDA::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
	size_t i = Index(n, x, y, z);
	if (z2cu_upload(m_g.volt + i, &value, sizeof(float)))
		Fail("set");
	volt_ptr->data()[i] = value;
	auto it = m_probe_slot.find(i);
	if (it != m_probe_slot.end())
		m_probe_value[it->second] = value;
}

void Engine_CUDA::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
	size_t i = Index(n, x, y, z);
	if (z2cu_upload(m_g.curr + i, &value, sizeof(float)))
		Fail("set");
	curr_ptr->data()[i] = value;
	auto it = m_probe_slot.find((1ULL << 62) | i);
	if (it != m_probe_slot.end())
		m_probe_value[it->second] = value;
}

void Engine_CUDA::SyncHost()
{
	if (m_host_valid)
		return;
	size_t bytes = 3 * m_cells * sizeof(float);
	if (z2cu_download(volt_ptr->data(), m_g.volt, bytes) || z2cu_download(curr_ptr->data(), m_g.curr, bytes))
		Fail("sync to host");
	m_host_valid = true;
}
