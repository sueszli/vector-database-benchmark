#include "common.h"

namespace soa {

struct Point {
    float x[N];
    float y[N];
    float z[N];
};

Point ps;

void compute() {
    for (int i = 0; i < N; i++) {
        ps.x[i] = ps.x[i] + ps.y[i] + ps.z[i];
    }
}

}
