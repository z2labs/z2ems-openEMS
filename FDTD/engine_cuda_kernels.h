/*
*	Copyright (C) 2026 Zoltan Doczi
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

// Plain C interface between the CUDA engine (host C++, built by the normal compiler) and its kernels
// (engine_cuda_kernels.cu, built by nvcc). Only POD types cross this boundary, so the two compilers
// never have to agree on a C++ ABI.

#ifndef ENGINE_CUDA_KERNELS_H
#define ENGINE_CUDA_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	unsigned int nx, ny, nz;        // mesh lines
	float *volt, *curr;             // fields, layout [n][x][y][z]
	const float *vv, *vi, *ii, *iv; // operator coefficients, same layout
} z2cu_grid;

// core Yee updates (the basic engine's formulas)
int z2cu_update_volt(const z2cu_grid* g);
int z2cu_update_curr(const z2cu_grid* g);

// stream and device time step: call z2cu_init() once; the time-dependent kernels read the device copy of
// the time step, so one time step can be captured as a CUDA graph and replayed
int z2cu_init(void);
int z2cu_set_ts(int ts);
int z2cu_next_ts(void);                 // ++ts on the device (last kernel of a time step)
int z2cu_graph_begin(void);             // capture the kernels launched until z2cu_graph_end()
void* z2cu_graph_end(void);             // executable graph, NULL on failure
int z2cu_graph_launch(void* graph);
int z2cu_graph_free(void* graph);

// energy estimate as Engine_Interface_FDTD::CalcFastEnergy: sums of the squared volt and curr values
int z2cu_energy(const z2cu_grid* g, double* volt_sq, double* curr_sq);
// memory guard: order-independent 64-bit checksum of n floats (a changed element changes it)
int z2cu_hash(const float* a, size_t n, unsigned long long* out);

// soft excitation: field[dir[k]][x[k]][y[k]][z[k]] += amp[k] * signal[idx(ts - delay[k])], ts = device time step
int z2cu_excite(float* field, const z2cu_grid* g, unsigned int count, const unsigned int* x, const unsigned int* y,
                const unsigned int* z, const unsigned short* dir, const float* amp, const unsigned int* delay,
                const float* signal, unsigned int length, int period);

// UPML box (one per boundary region); stage 0 = pre-update, 1 = post-update; field = volt or curr
typedef struct
{
	unsigned int start[3], n[3];    // box in the global mesh, box size
	const float *c, *cfo, *cfn;     // vv/vvfo/vvfn (or ii/iifo/iifn), layout [n][x][y][z] of the box
	float* flux;
} z2cu_upml;
int z2cu_upml_pre(float* field, const z2cu_grid* g, const z2cu_upml* u);
int z2cu_upml_post(float* field, const z2cu_grid* g, const z2cu_upml* u);

// Mur ABC on one face; stage 0 pre, 1 post, 2 apply
typedef struct
{
	int ny, nyP, nyPP;
	unsigned int line, line_shift;
	unsigned int start_TS;          // the face starts working at this time step (after an excitation on it)
	unsigned int n0, n1;            // lines along nyP, nyPP
	const float *coeff_P, *coeff_PP;
	float *v_P, *v_PP;              // work arrays [n0][n1]
} z2cu_mur;
int z2cu_mur_step(float* volt, const z2cu_grid* g, const z2cu_mur* m, int stage);

// lumped RLC elements (Engine_Ext_LumpedRLC): state Il[count], Vd[3][count], J[3][count] in rotating buffers
typedef struct
{
	unsigned int count;
	const unsigned int *x, *y, *z;
	const int* dir;
	const float *ilv, *i2v, *vv2, *vj1, *vj2, *vvd, *ib0, *b1, *b2;
	float *Il, *Vd, *J;             // Vd, J: 3 * count
} z2cu_rlc;
// pre: Il += i2v*ilv*Vd[1]; apply: node voltage update. The Vd/J buffers rotate every time step
// (Vd[k] = buf[(r+k)%3], r = 2 (ts+1) mod 3, from the device time step)
int z2cu_rlc_pre(const z2cu_rlc* r);
int z2cu_rlc_apply(float* volt, const z2cu_grid* g, const z2cu_rlc* r);

// Drude/Lorentz ADE (Engine_Ext_LorentzMaterial, also conducting sheets): one order o of one extension.
// field 0 = volt side, 1 = curr side; all arrays [3][count] (component-major), NULL when that part is off
typedef struct
{
	unsigned int count;
	const unsigned int *x, *y, *z;
	int on[2], lor_on[2];
	const float *c_int[2], *c_ext[2], *c_lor[2];
	float *A[2], *L[2];
} z2cu_lorentz;
int z2cu_lorentz_pre(float* field, const z2cu_grid* g, const z2cu_lorentz* d, int side);
int z2cu_disp_apply(float* field, const z2cu_grid* g, unsigned int count, const unsigned int* x, const unsigned int* y,
                    const unsigned int* z, const float* A);

// Debye poles (Engine_Ext_DebyeMaterial): implicit joint solve; pole arrays [poles][3][count], others [3][count]
typedef struct
{
	unsigned int count;
	int poles;
	const unsigned int *x, *y, *z;
	const float *relax, *drive, *solve;
	float *pole, *pre, *A;
} z2cu_debye;
int z2cu_debye_pre(const float* volt, const z2cu_grid* g, const z2cu_debye* d);
int z2cu_debye_post(const float* volt, const z2cu_grid* g, const z2cu_debye* d);

// sparse gather of probe values: out[k] = field_k[index[k]] (field 0 = volt, 1 = curr)
int z2cu_gather(const z2cu_grid* g, unsigned int count, const unsigned int* index, const unsigned char* which, float* out);

// device memory helpers
void* z2cu_alloc(size_t bytes);
int z2cu_free(void* p);
int z2cu_upload(void* dst, const void* src, size_t bytes);
int z2cu_download(void* dst, const void* src, size_t bytes);
int z2cu_memset(void* p, int value, size_t bytes);
int z2cu_sync(void);
const char* z2cu_last_error(void);
int z2cu_device_info(char* name, size_t len, size_t* total_mem, int* cc_major, int* cc_minor);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_CUDA_KERNELS_H
