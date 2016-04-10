/* Copyright (c) 2016 Philip Taylor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and/or associated documentation files (the "Materials"), to
 * deal in the Materials without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Materials, and to permit persons to whom the Materials
 * are furnished to do so, subject to the following conditions:
 *
 * The above copyright notice(s) and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE
 * USE OR OTHER DEALINGS IN THE MATERIALS
 */

typedef enum _SYNC_ERROR {
    DEVLIMITS_NONE,                          // Used for INFO & other non-error messages
    DEVLIMITS_INVALID_INSTANCE,              // Invalid instance used
    DEVLIMITS_INVALID_PHYSICAL_DEVICE,       // Invalid physical device used
    DEVLIMITS_INVALID_INHERITED_QUERY,       // Invalid use of inherited query
    DEVLIMITS_INVALID_ATTACHMENT_COUNT,      // Invalid value for the number of attachments
    DEVLIMITS_MUST_QUERY_COUNT,              // Failed to make initial call to an API to query the count
    DEVLIMITS_INVALID_CALL_SEQUENCE,         // Flag generic case of an invalid call sequence by the app
    DEVLIMITS_INVALID_FEATURE_REQUESTED,     // App requested a feature not supported by physical device
    DEVLIMITS_COUNT_MISMATCH,                // App requesting a count value different than actual value
    DEVLIMITS_INVALID_QUEUE_CREATE_REQUEST,  // Invalid queue requested based on queue family properties
    DEVLIMITS_INVALID_UNIFORM_BUFFER_OFFSET, // Uniform buffer offset violates device limit granularity
    DEVLIMITS_INVALID_STORAGE_BUFFER_OFFSET, // Storage buffer offset violates device limit granularity
    DEVLIMITS_INVALID_BUFFER_UPDATE_ALIGNMENT,  // Alignment requirement for buffer update is violated
} SYNC_ERROR;
