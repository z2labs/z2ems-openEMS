/*
*	Copyright (C) 2026 Zoltan Doczi
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

// CUDA kernels of the openEMS CUDA engine. The update formulas are those of the basic engine
// (FDTD/engine.cpp) and of the extensions they replace, one thread per cell (or per entry).

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "engine_cuda_kernels.h"

static char g_err[256] = "";
static cudaStream_t g_stream = 0;      // every kernel runs here, so one time step can be captured as a CUDA graph
static int* g_ts = NULL;               // device copy of the time step counter (read by the time-dependent kernels)

static int check(const char* what)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
		return 1;
	}
	return 0;
}

#define IDX(g, n, x, y, z) ((((size_t)(n) * (g).nx + (x)) * (g).ny + (y)) * (g).nz + (z))

// ---------------------------------------------------------------- core
// The fields and coefficients never alias: __restrict__ plus __ldg() let Maxwell/Pascal cards serve the
// read-only data (coefficients, the other field) from the read-only/texture cache.
__global__ void k_volt(z2cu_grid g)
{
	unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = blockIdx.z;
	if (z >= g.nz || y >= g.ny) return;
	unsigned int sx = x > 0, sy = y > 0, sz = z > 0;
	const float* __restrict__ I = g.curr;
	float* __restrict__ V = g.volt;
	const float* __restrict__ vv = g.vv;
	const float* __restrict__ vi = g.vi;
	size_t i0 = IDX(g, 0, x, y, z), i1 = IDX(g, 1, x, y, z), i2 = IDX(g, 2, x, y, z);
	float c0 = __ldg(I + i0), c1 = __ldg(I + i1), c2 = __ldg(I + i2);
	// x
	V[i0] = V[i0] * __ldg(vv + i0) + __ldg(vi + i0) * (c2 - __ldg(I + IDX(g, 2, x, y - sy, z)) - c1 + __ldg(I + IDX(g, 1, x, y, z - sz)));
	// y
	V[i1] = V[i1] * __ldg(vv + i1) + __ldg(vi + i1) * (c0 - __ldg(I + IDX(g, 0, x, y, z - sz)) - c2 + __ldg(I + IDX(g, 2, x - sx, y, z)));
	// z
	V[i2] = V[i2] * __ldg(vv + i2) + __ldg(vi + i2) * (c1 - __ldg(I + IDX(g, 1, x - sx, y, z)) - c0 + __ldg(I + IDX(g, 0, x, y - sy, z)));
}

__global__ void k_curr(z2cu_grid g)
{
	unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = blockIdx.z;
	if (z >= g.nz - 1 || y >= g.ny - 1) return;
	const float* __restrict__ V = g.volt;
	float* __restrict__ I = g.curr;
	const float* __restrict__ ii = g.ii;
	const float* __restrict__ iv = g.iv;
	size_t i0 = IDX(g, 0, x, y, z), i1 = IDX(g, 1, x, y, z), i2 = IDX(g, 2, x, y, z);
	float v0 = __ldg(V + i0), v1 = __ldg(V + i1), v2 = __ldg(V + i2);
	I[i0] = I[i0] * __ldg(ii + i0) + __ldg(iv + i0) * (v2 - __ldg(V + IDX(g, 2, x, y + 1, z)) - v1 + __ldg(V + IDX(g, 1, x, y, z + 1)));
	I[i1] = I[i1] * __ldg(ii + i1) + __ldg(iv + i1) * (v0 - __ldg(V + IDX(g, 0, x, y, z + 1)) - v2 + __ldg(V + IDX(g, 2, x + 1, y, z)));
	I[i2] = I[i2] * __ldg(ii + i2) + __ldg(iv + i2) * (v1 - __ldg(V + IDX(g, 1, x + 1, y, z)) - v0 + __ldg(V + IDX(g, 0, x, y + 1, z)));
}

extern "C" int z2cu_update_volt(const z2cu_grid* g)
{
	dim3 b(32, 8, 1), n((g->nz + 31) / 32, (g->ny + 7) / 8, g->nx);
	k_volt<<<n, b, 0, g_stream>>>(*g);
	return check("update_volt");
}

extern "C" int z2cu_update_curr(const z2cu_grid* g)
{
	if (g->nx < 2) return 0;
	dim3 b(32, 8, 1), n((g->nz + 31) / 32, (g->ny + 7) / 8, g->nx - 1);
	k_curr<<<n, b, 0, g_stream>>>(*g);
	return check("update_curr");
}

// ---------------------------------------------------------------- excitation
__global__ void k_excite(float* F, z2cu_grid g, unsigned int count, const unsigned int* x, const unsigned int* y,
                         const unsigned int* z, const unsigned short* dir, const float* amp, const unsigned int* delay,
                         const float* signal, unsigned int length, int period, const int* ts)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= count) return;
	int e = *ts - (int)delay[k];
	e *= (e > 0);
	if (period > 0)
		e %= period;
	e *= (e < (int)length);
	atomicAdd(&F[IDX(g, dir[k], x[k], y[k], z[k])], amp[k] * signal[e]);   // entries may share a cell
}

extern "C" int z2cu_excite(float* field, const z2cu_grid* g, unsigned int count, const unsigned int* x, const unsigned int* y,
                           const unsigned int* z, const unsigned short* dir, const float* amp, const unsigned int* delay,
                           const float* signal, unsigned int length, int period)
{
	if (!count) return 0;
	k_excite<<<(count + 255) / 256, 256, 0, g_stream>>>(field, *g, count, x, y, z, dir, amp, delay, signal, length, period, g_ts);
	return check("excite");
}

// ---------------------------------------------------------------- UPML
#define BIDX(u, n, x, y, z) ((((size_t)(n) * (u).n[0] + (x)) * (u).n[1] + (y)) * (u).n[2] + (z))

__global__ void k_upml(float* F, z2cu_grid g, z2cu_upml u, int post)
{
	unsigned int lz = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ly = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int lx = blockIdx.z;
	if (lz >= u.n[2] || ly >= u.n[1]) return;
	for (int n = 0; n < 3; ++n)
	{
		size_t b = BIDX(u, n, lx, ly, lz);
		size_t i = IDX(g, n, lx + u.start[0], ly + u.start[1], lz + u.start[2]);
		if (!post)
		{
			float f = u.c[b] * F[i] - u.cfo[b] * u.flux[b];
			F[i] = u.flux[b];
			u.flux[b] = f;
		}
		else
		{
			float f = u.flux[b];
			u.flux[b] = F[i];
			F[i] = f + u.cfn[b] * u.flux[b];
		}
	}
}

static int upml(float* field, const z2cu_grid* g, const z2cu_upml* u, int post)
{
	if (!u->n[0] || !u->n[1] || !u->n[2]) return 0;
	dim3 b(32, 8, 1), n((u->n[2] + 31) / 32, (u->n[1] + 7) / 8, u->n[0]);
	k_upml<<<n, b, 0, g_stream>>>(field, *g, *u, post);
	return check(post ? "upml_post" : "upml_pre");
}
extern "C" int z2cu_upml_pre(float* field, const z2cu_grid* g, const z2cu_upml* u) { return upml(field, g, u, 0); }
extern "C" int z2cu_upml_post(float* field, const z2cu_grid* g, const z2cu_upml* u) { return upml(field, g, u, 1); }

// ---------------------------------------------------------------- Mur ABC
__global__ void k_mur(float* V, z2cu_grid g, z2cu_mur m, int stage, const int* ts)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.y;
	if (j >= m.n1 || i >= m.n0 || *ts < (int)m.start_TS) return;
	unsigned int p[3], s[3];
	p[m.ny] = m.line; s[m.ny] = m.line_shift;
	p[m.nyP] = s[m.nyP] = i;
	p[m.nyPP] = s[m.nyPP] = j;
	size_t w = (size_t)i * m.n1 + j;
	size_t aP = IDX(g, m.nyP, p[0], p[1], p[2]), aPP = IDX(g, m.nyPP, p[0], p[1], p[2]);
	size_t sP = IDX(g, m.nyP, s[0], s[1], s[2]), sPP = IDX(g, m.nyPP, s[0], s[1], s[2]);
	if (stage == 0)
	{
		m.v_P[w] = V[sP] - m.coeff_P[w] * V[aP];
		m.v_PP[w] = V[sPP] - m.coeff_PP[w] * V[aPP];
	}
	else if (stage == 1)
	{
		m.v_P[w] += m.coeff_P[w] * V[sP];
		m.v_PP[w] += m.coeff_PP[w] * V[sPP];
	}
	else
	{
		V[aP] = m.v_P[w];
		V[aPP] = m.v_PP[w];
	}
}

extern "C" int z2cu_mur_step(float* volt, const z2cu_grid* g, const z2cu_mur* m, int stage)
{
	if (!m->n0 || !m->n1) return 0;
	dim3 b(64, 1, 1), n((m->n1 + 63) / 64, m->n0, 1);
	k_mur<<<n, b, 0, g_stream>>>(volt, *g, *m, stage, g_ts);
	return check("mur");
}

// ---------------------------------------------------------------- lumped RLC
__global__ void k_rlc_pre(z2cu_rlc r, const int* ts)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= r.count) return;
	int rv = (2 * (*ts + 1)) % 3;
	const float* Vd1 = r.Vd + (size_t)((rv + 1) % 3) * r.count;
	r.Il[k] += r.i2v[k] * r.ilv[k] * Vd1[k];
}

__global__ void k_rlc_apply(float* V, z2cu_grid g, z2cu_rlc r, const int* ts)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= r.count) return;
	int rv = (2 * (*ts + 1)) % 3, rj = rv;
	float* Vd0 = r.Vd + (size_t)(rv % 3) * r.count;
	const float* Vd2 = r.Vd + (size_t)((rv + 2) % 3) * r.count;
	float* J0 = r.J + (size_t)(rj % 3) * r.count;
	const float* J1 = r.J + (size_t)((rj + 1) % 3) * r.count;
	const float* J2 = r.J + (size_t)((rj + 2) % 3) * r.count;
	size_t c = IDX(g, r.dir[k], r.x[k], r.y[k], r.z[k]);
	float vd = V[c];
	vd = r.vvd[k] * (vd - r.Il[k] + r.vv2[k] * Vd2[k] + r.vj1[k] * J1[k] + r.vj2[k] * J2[k]);
	J0[k] = r.ib0[k] * (vd - Vd2[k]) - (r.b1[k] * r.ib0[k]) * J1[k] - (r.b2[k] * r.ib0[k]) * J2[k];
	Vd0[k] = vd;
	V[c] = vd;
}

extern "C" int z2cu_rlc_pre(const z2cu_rlc* r)
{
	if (!r->count) return 0;
	k_rlc_pre<<<(r->count + 255) / 256, 256, 0, g_stream>>>(*r, g_ts);
	return check("rlc_pre");
}

extern "C" int z2cu_rlc_apply(float* volt, const z2cu_grid* g, const z2cu_rlc* r)
{
	if (!r->count) return 0;
	k_rlc_apply<<<(r->count + 255) / 256, 256, 0, g_stream>>>(volt, *g, *r, g_ts);
	return check("rlc_apply");
}

// ---------------------------------------------------------------- dispersive materials
__global__ void k_lorentz_pre(const float* F, z2cu_grid g, z2cu_lorentz d, int s)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d.count) return;
	for (int n = 0; n < 3; ++n)
	{
		size_t k = (size_t)n * d.count + i;
		float f = F[IDX(g, n, d.x[i], d.y[i], d.z[i])];
		if (d.lor_on[s])
		{
			d.L[s][k] += d.c_lor[s][k] * d.A[s][k];
			d.A[s][k] = d.A[s][k] * d.c_int[s][k] + d.c_ext[s][k] * (f - d.L[s][k]);
		}
		else
			d.A[s][k] = d.A[s][k] * d.c_int[s][k] + d.c_ext[s][k] * f;
	}
}

__global__ void k_disp_apply(float* F, z2cu_grid g, unsigned int count, const unsigned int* x, const unsigned int* y,
                             const unsigned int* z, const float* A)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;
	for (int n = 0; n < 3; ++n)
		F[IDX(g, n, x[i], y[i], z[i])] -= A[(size_t)n * count + i];
}

__global__ void k_debye_pre(const float* V, z2cu_grid g, z2cu_debye d)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d.count) return;
	for (int n = 0; n < 3; ++n)
	{
		float v = V[IDX(g, n, d.x[i], d.y[i], d.z[i])];
		float pre = 0;
		for (int o = 0; o < d.poles; ++o)
		{
			size_t k = ((size_t)o * 3 + n) * d.count + i;
			float W = d.relax[k] * d.pole[k] + d.drive[k] * v;
			pre += W - d.pole[k];
			d.pole[k] = W;
		}
		d.pre[(size_t)n * d.count + i] = pre;
	}
}

__global__ void k_debye_post(const float* V, z2cu_grid g, z2cu_debye d)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d.count) return;
	for (int n = 0; n < 3; ++n)
	{
		size_t c = (size_t)n * d.count + i;
		float raw = V[IDX(g, n, d.x[i], d.y[i], d.z[i])];
		float v = (raw - d.pre[c]) * d.solve[c];
		d.A[c] = raw - v;
		for (int o = 0; o < d.poles; ++o)
		{
			size_t k = ((size_t)o * 3 + n) * d.count + i;
			d.pole[k] += d.drive[k] * v;
		}
	}
}

extern "C" int z2cu_lorentz_pre(float* field, const z2cu_grid* g, const z2cu_lorentz* d, int side)
{
	if (!d->count || !d->on[side]) return 0;
	k_lorentz_pre<<<(d->count + 255) / 256, 256, 0, g_stream>>>(field, *g, *d, side);
	return check("lorentz_pre");
}

extern "C" int z2cu_disp_apply(float* field, const z2cu_grid* g, unsigned int count, const unsigned int* x, const unsigned int* y,
                               const unsigned int* z, const float* A)
{
	if (!count) return 0;
	k_disp_apply<<<(count + 255) / 256, 256, 0, g_stream>>>(field, *g, count, x, y, z, A);
	return check("disp_apply");
}

extern "C" int z2cu_debye_pre(const float* volt, const z2cu_grid* g, const z2cu_debye* d)
{
	if (!d->count) return 0;
	k_debye_pre<<<(d->count + 255) / 256, 256, 0, g_stream>>>(volt, *g, *d);
	return check("debye_pre");
}

extern "C" int z2cu_debye_post(const float* volt, const z2cu_grid* g, const z2cu_debye* d)
{
	if (!d->count) return 0;
	k_debye_post<<<(d->count + 255) / 256, 256, 0, g_stream>>>(volt, *g, *d);
	return check("debye_post");
}

// ---------------------------------------------------------------- probes
__global__ void k_gather(const float* V, const float* I, unsigned int count, const unsigned int* index, const unsigned char* which, float* out)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= count) return;
	out[k] = which[k] ? I[index[k]] : V[index[k]];
}

extern "C" int z2cu_gather(const z2cu_grid* g, unsigned int count, const unsigned int* index, const unsigned char* which, float* out)
{
	if (!count) return 0;
	k_gather<<<(count + 255) / 256, 256, 0, g_stream>>>(g->volt, g->curr, count, index, which, out);
	return check("gather");
}

// ---------------------------------------------------------------- memory
extern "C" void* z2cu_alloc(size_t bytes)
{
	void* p = NULL;
	if (cudaMalloc(&p, bytes ? bytes : 4) != cudaSuccess) { check("alloc"); return NULL; }
	return p;
}
extern "C" int z2cu_free(void* p) { return p ? (cudaFree(p) != cudaSuccess) : 0; }
extern "C" int z2cu_upload(void* d, const void* s, size_t n) { if (!n) return 0; if (cudaMemcpy(d, s, n, cudaMemcpyHostToDevice) != cudaSuccess) return check("upload"); return 0; }
extern "C" int z2cu_download(void* d, const void* s, size_t n) { if (!n) return 0; if (cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost) != cudaSuccess) return check("download"); return 0; }
extern "C" int z2cu_memset(void* p, int v, size_t n) { if (!n) return 0; if (cudaMemset(p, v, n) != cudaSuccess) return check("memset"); return 0; }
extern "C" int z2cu_sync(void) { if (cudaDeviceSynchronize() != cudaSuccess) return check("sync"); return check("sync"); }

// ---------------------------------------------------------------- stream, time step, graphs
extern "C" int z2cu_init(void)
{
	if (!g_stream && cudaStreamCreate(&g_stream) != cudaSuccess) return check("stream");
	if (!g_ts && cudaMalloc(&g_ts, sizeof(int)) != cudaSuccess) return check("ts alloc");
	return z2cu_set_ts(0);
}

extern "C" int z2cu_set_ts(int ts)
{
	if (cudaMemcpy(g_ts, &ts, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return check("set ts");
	return 0;
}

__global__ void k_next_ts(int* ts) { ++*ts; }
extern "C" int z2cu_next_ts(void)
{
	k_next_ts<<<1, 1, 0, g_stream>>>(g_ts);
	return check("next ts");
}

extern "C" int z2cu_graph_begin(void)
{
	if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return check("capture begin");
	return 0;
}

extern "C" void* z2cu_graph_end(void)
{
	cudaGraph_t graph;
	cudaGraphExec_t exec = NULL;
	if (cudaStreamEndCapture(g_stream, &graph) != cudaSuccess) { check("capture end"); return NULL; }
	cudaError_t e = cudaGraphInstantiate(&exec, graph, 0);
	cudaGraphDestroy(graph);
	if (e != cudaSuccess) { check("graph instantiate"); return NULL; }
	return (void*)exec;
}

extern "C" int z2cu_graph_launch(void* exec)
{
	if (cudaGraphLaunch((cudaGraphExec_t)exec, g_stream) != cudaSuccess) return check("graph launch");
	return 0;
}

extern "C" int z2cu_graph_free(void* exec)
{
	return exec ? (cudaGraphExecDestroy((cudaGraphExec_t)exec) != cudaSuccess) : 0;
}

// ---------------------------------------------------------------- energy estimate (sum of squares, as CalcFastEnergy)
__global__ void k_energy(const float* __restrict__ F, z2cu_grid g, double* partial)
{
	extern __shared__ double sh[];
	double acc = 0.0;
	size_t n = (size_t)(g.nx - 1) * (g.ny - 1) * (g.nz - 1);
	for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < n; c += (size_t)gridDim.x * blockDim.x)
	{
		unsigned int z = c % (g.nz - 1), y = (c / (g.nz - 1)) % (g.ny - 1), x = c / ((size_t)(g.nz - 1) * (g.ny - 1));
		for (int k = 0; k < 3; ++k)
		{
			float f = __ldg(F + IDX(g, k, x, y, z));
			acc += (double)(f * f);
		}
	}
	sh[threadIdx.x] = acc;
	__syncthreads();
	for (unsigned int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1)
	{
		if (threadIdx.x < s2) sh[threadIdx.x] += sh[threadIdx.x + s2];
		__syncthreads();
	}
	if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

extern "C" int z2cu_energy(const z2cu_grid* g, double* volt_sq, double* curr_sq)
{
	enum { B = 256, N = 256 };
	static double* part = NULL;
	double h[2 * N];
	if (g->nx < 2 || g->ny < 2 || g->nz < 2) { *volt_sq = *curr_sq = 0; return 0; }
	if (!part && cudaMalloc(&part, 2 * N * sizeof(double)) != cudaSuccess) return check("energy alloc");
	k_energy<<<N, B, B * sizeof(double), g_stream>>>(g->volt, *g, part);
	k_energy<<<N, B, B * sizeof(double), g_stream>>>(g->curr, *g, part + N);
	if (cudaMemcpyAsync(h, part, sizeof(h), cudaMemcpyDeviceToHost, g_stream) != cudaSuccess || cudaStreamSynchronize(g_stream) != cudaSuccess)
		return check("energy");
	double a = 0, b = 0;
	for (int k = 0; k < N; ++k) { a += h[k]; b += h[N + k]; }
	*volt_sq = a; *curr_sq = b;
	return check("energy");
}
// ---------------------------------------------------------------- memory guard (checksum of a static array)
// Order-independent 64-bit sum of the element bits mixed with the index: a single changed element changes
// the sum. Used on the coefficient arrays, which never change during a run, to catch GPU memory errors.
__global__ void k_hash(const unsigned int* __restrict__ a, size_t n, unsigned long long* partial)
{
	extern __shared__ unsigned long long shh[];
	unsigned long long acc = 0;
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
		acc += ((unsigned long long)__ldg(a + i) + 1ull) * ((unsigned long long)i * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull);
	shh[threadIdx.x] = acc;
	__syncthreads();
	for (unsigned int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1)
	{
		if (threadIdx.x < s2) shh[threadIdx.x] += shh[threadIdx.x + s2];
		__syncthreads();
	}
	if (threadIdx.x == 0) partial[blockIdx.x] = shh[0];
}

extern "C" int z2cu_hash(const float* a, size_t n, unsigned long long* out)
{
	enum { B = 256, N = 256 };
	static unsigned long long* part = NULL;
	unsigned long long h[N];
	if (!part && cudaMalloc(&part, N * sizeof(unsigned long long)) != cudaSuccess) return check("hash alloc");
	k_hash<<<N, B, B * sizeof(unsigned long long), g_stream>>>((const unsigned int*)a, n, part);
	if (cudaMemcpyAsync(h, part, sizeof(h), cudaMemcpyDeviceToHost, g_stream) != cudaSuccess || cudaStreamSynchronize(g_stream) != cudaSuccess)
		return check("hash");
	unsigned long long s = 0;
	for (int k = 0; k < N; ++k) s += h[k];
	*out = s;
	return check("hash");
}

extern "C" const char* z2cu_last_error(void) { return g_err; }
extern "C" int z2cu_device_info(char* name, size_t len, size_t* total_mem, int* cc_major, int* cc_minor)
{
	cudaDeviceProp p;
	if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return check("device");
	strncpy(name, p.name, len - 1); name[len - 1] = 0;
	*total_mem = p.totalGlobalMem; *cc_major = p.major; *cc_minor = p.minor;
	return 0;
}
