/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 *
 * Contributions by:
 *   Martin Hilgeman <martin.hilgeman@amd.com>
 *   Samuel Antao <samuel.antao@amd.com>
 * 
 * Description:
 *   RCCL tuner plugin to force the Tree algorithm on small allreduce
 ************************************************************************/

#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "tuner.h"

#define PLUGIN_NAME "My RCCL Plugin"

/*
 * Helper macro for logging, using the provided logFunction
*/
#define PLUGIN_LOG(level, subsys, ...) \
    do { \
        logFunction(level, subsys, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

struct myContext_t {
    size_t nRanks;
    size_t nNodes;
    ncclDebugLogger_t logFunction;
};

#define ContextMaxEntries 1024
static int myContextEntriesIsInitialized = 0;
static struct myContext_t myContextEntries[ContextMaxEntries];

ncclResult_t myTunerInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) {

    if (nRanks == 0) {
        PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]: No ranks specified for tunner.", PLUGIN_NAME);
        return ncclInternalError;
    }

    // Context entries lazy initialization
    if (!myContextEntriesIsInitialized) {
        memset(myContextEntries, 0, sizeof(myContextEntries));
        myContextEntriesIsInitialized = 1;
    }
    
    // Find a context.
    struct myContext_t *c = NULL;
    for(size_t i=0; i<ContextMaxEntries; ++i) {
        struct myContext_t *cc = &myContextEntries[i];
        if(cc->nRanks == 0) {
            c = cc;
            break;
        }
    }

    // Fail if we didn't manage to obtain a context.
    if (c == NULL) {
        PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]: Ran out of contexts objects for tunner.", PLUGIN_NAME);
        return ncclInternalError;
    }

    c->nRanks = nRanks;
    c->nNodes = nNodes;
    c->logFunction = logFunction;
    *context = (void *)c;

    PLUGIN_LOG(NCCL_LOG_INFO, NCCL_INIT, "[%s] Initialized.", PLUGIN_NAME);
    PLUGIN_LOG(NCCL_LOG_INFO, NCCL_INIT, "[%s]:  nRanks: %zu, nNodes: %zu, context %p", PLUGIN_NAME, nRanks, nNodes, *context);
    return ncclSuccess;
}

ncclResult_t myTunerGetCollInfo(void *context, ncclFunc_t collType, size_t nBytes, 
                                int collNetSupport, int nvlsSupport, int numPipeOps,
                                int* algorithm, int* protocol, int* nChannels) {
    if (!context) {
        return ncclInternalError;
    }

    struct myContext_t *c = (struct myContext_t *)context;
    ncclDebugLogger_t logFunction = c->logFunction;

    if (collType == ncclFuncAllReduce && nBytes <= 64 && c->nNodes >= 16) {
        PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]:    AllReduce detected (nBytes: %zu). Forcing Ring algorithm.", PLUGIN_NAME, nBytes);

        *algorithm = NCCL_ALGO_TREE;
        *protocol = NCCL_PROTO_LL;
        PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]:    Selected NCCL_PROTO_LL protocol.", PLUGIN_NAME);

        /*
         * Protocol selection:
         * NCCL_PROTO_LL for very small messages (low latency)
         * NCCL_PROTO_LL128 for small to medium messages (optimized for 128-byte segments)
         * NCCL_PROTO_SIMPLE for larger messages (high bandwidth)
         */

        PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]:    Algorithm: Ring, Protocol: %s, nChannels: %d",
                   PLUGIN_NAME,
                   (*protocol == NCCL_PROTO_SIMPLE ? "SIMPLE" : (*protocol == NCCL_PROTO_LL ? "LL" : "LL128")),
                   *nChannels);
    } else {
        PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]:    Collective type %d. Standard algorithm.", PLUGIN_NAME, collType);
    }

    return ncclSuccess;
}

ncclResult_t myTunerDestroy(void *context) {
    if (!context) {
        return ncclInternalError;
    }

    struct myContext_t *c = (struct myContext_t *)context;
    ncclDebugLogger_t logFunction = c->logFunction;

    c->nNodes = 0;
    c->nRanks = 0;
    c->logFunction = NULL;

    PLUGIN_LOG(NCCL_LOG_INFO, NCCL_COLL, "[%s]: Destroyed.", PLUGIN_NAME);
    return ncclSuccess;
}

const ncclTuner_v2_t ncclTunerPlugin_v2 = {
    .name = PLUGIN_NAME,
    .init = myTunerInit,
    .getCollInfo = myTunerGetCollInfo,
    .destroy = myTunerDestroy
};

ncclResult_t ncclTunerGetInterface(int version, ncclTuner_v2_t **tunerInterface) {
    if (version != 2) {
        return ncclInvalidArgument;
    }
    *tunerInterface = (ncclTuner_v2_t *)&ncclTunerPlugin_v2;

    return ncclSuccess;
}
