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

#include "vk_loader_platform.h"
#include "vk_layer.h"

#include <map>
#include <set>

enum sync_msg
{
    SYNC_MSG_NONE,                           // Used for INFO & other non-error messages
    SYNC_MSG_INVALID_PARAM,
};

struct sync_command_buffer
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
};

struct sync_command_pool
{
    VkCommandPool command_pool;
    std::set<VkCommandBuffer> command_buffers;
};

struct sync_device
{
    std::map<VkCommandPool, sync_command_pool> command_pools;
    std::map<VkCommandBuffer, sync_command_buffer> command_buffers;
};
