#include "kernel.h"


float* dt_host;
DtKernelArgs* dtArgs;
RKKernelArgs* RKArgs[3];
FluxKernelArgs* fluxArgs[3];
collBCKernelArgs* BCArgs[3];
