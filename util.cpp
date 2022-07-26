#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <float.h>

#include "util.h"


double read_timer_ms() {
    struct timespec tm;
    clock_gettime(CLOCK_REALTIME, &tm);
    return (double) tm.tv_sec * 1000.0 + (double) tm.tv_nsec/1.0e6;
}

