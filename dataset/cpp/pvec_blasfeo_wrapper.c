#include "pvec_blasfeo_wrapper.h"
#include "pmt_heap.h"
#include "pmt_aux.h"
#include <assert.h>
#include <blasfeo_common.h>

extern void make_int_multiple_of(int multiple_of, int * n);

struct pvec * c_pmt_create_pvec(int m) {	
    // assign current address of global heap to pvec pointer
    struct pvec *pvec = (struct pvec *) ___c_pmt_8_heap;
    void *pvec_address = ___c_pmt_8_heap;
    
    // advance global heap address
    ___c_pmt_8_heap += sizeof(struct pvec);
    
    
    // create (zeroed) blasfeo_dvec and advance global heap
    c_pmt_assign_and_advance_blasfeo_dvec(m, &(pvec->bvec));

	return (struct pvec *)(pvec_address);
}


void c_pmt_assign_and_advance_blasfeo_dvec(int m, struct blasfeo_dvec **bvec) {
    // assign current address of global heap to blasfeo dvec pointer
    assert((size_t) ___c_pmt_8_heap % 8 == 0 && "pointer not 8-byte aligned!");
    *bvec = (struct blasfeo_dvec *) ___c_pmt_8_heap;
    //
    // advance global heap address
    ___c_pmt_8_heap += sizeof(struct blasfeo_dvec);

    // assign current address of global heap to memory in blasfeo dvec
    char *pmem_ptr = (char *)___c_pmt_64_heap; 
    // align_char_to(64, &pmem_ptr);
    ___c_pmt_64_heap = pmem_ptr;
    assert((size_t) ___c_pmt_64_heap % 64 == 0 && "dvec not 64-byte aligned!");
    blasfeo_create_dvec(m, *bvec, ___c_pmt_64_heap);

    // advance global heap address
    int memsize = (*bvec)->memsize;
    make_int_multiple_of(64, &memsize);
    ___c_pmt_64_heap += memsize;	

    // zero allocated memory
	int i;
	double *pa = (*bvec)->pa;
    int size = (*bvec)->memsize;
	for(i=0; i<size/8; i++) pa[i] = 0.0;
	char *ca = (char *) pa;
	i *= 8;
	for(; i<size; i++) ca[i] = 0;
	return;
}

// auxiliary
void c_pmt_pvec_fill(struct pvec *a, double fill_value) {
    int m = a->bvec->m;

    for(int i = 0; i < m; i++)
        blasfeo_dvecin1(fill_value, a->bvec, i);
}

void c_pmt_pvec_set_el(struct pvec *a, int i, double fill_value) {

    blasfeo_dvecin1(fill_value, a->bvec, i);
}

double c_pmt_pvec_get_el(struct pvec *a, int i) {

    blasfeo_dvecex1(a->bvec, i);
}

void c_pmt_pvec_copy(struct pvec *a, struct pvec *b) {
    int m = a->bvec->m;
    double value;

    for(int i = 0; i < m; i++) {
        value = blasfeo_dvecex1(a->bvec, i);
        blasfeo_dvecin1(value, b->bvec, i);
    }
}

void c_pmt_pvec_print(struct pvec *a) {
    int m = a->bvec->m;

    blasfeo_print_dvec(m, a->bvec, 0);
}
