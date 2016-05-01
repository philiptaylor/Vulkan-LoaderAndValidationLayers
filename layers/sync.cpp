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


/**
 * Synchronization validation:
 *
 * Conceptually, we want to construct a graph (DAG) of all commands that are
 * executed on a device, and validate that the synchronization is correct
 * across the entire DAG.
 *
 * To construct the DAG, we need to intercept every vkQueueSubmit, vkQueueBindSparse,
 * plus the construction of the command buffers that get submitted.
 *
 * To be more helpful to applications, we want to report errors as soon as possible,
 * i.e. during the command buffer construction (though in some cases it's impossible
 * to detect the error until submission).
 */


#include <unordered_map>
#include <mutex>
#include <sstream>

#include "sync.h"

#include "vk_loader_platform.h"
#include "vk_dispatch_table_helper.h"
#include "vk_layer_logging.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_utils.h"

#define LAYER_FN(ret) VK_LAYER_EXPORT VKAPI_ATTR ret VKAPI_CALL

#define _LOG_GENERIC(level, layer_data, objType, object, messageCode, fmt, ...) \
    log_msg((layer_data)->report_data, VK_DEBUG_REPORT_DEBUG_BIT_EXT, \
        VK_DEBUG_REPORT_OBJECT_TYPE_##objType##_EXT, (uint64_t)(object), \
        __LINE__, (messageCode), "SYNC", (fmt), __VA_ARGS__)

#define LOG_INFO(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_INFORMATION_BIT_EXT, layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_WARN(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_WARNING_BIT_EXT, layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_PERF(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_ERROR(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_ERROR_BIT_EXT, layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_DEBUG(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_DEBUG_BIT_EXT, layer_data, objType, object, messageCode, fmt, __VA_ARGS__)

struct layer_instance_data
{
    debug_report_data *report_data;
    std::vector<VkDebugReportCallbackEXT> logging_callback;
    VkLayerInstanceDispatchTable dispatch;
};

struct layer_device_data
{
    debug_report_data *report_data;
    std::vector<VkDebugReportCallbackEXT> logging_callback;
    VkLayerDispatchTable dispatch;

    std::mutex sync_mutex;
    sync_device sync;
};

static std::mutex g_layer_map_mutex;
static std::unordered_map<void *, std::unique_ptr<layer_instance_data>> g_layer_instance_data_map;
static std::unordered_map<void *, std::unique_ptr<layer_device_data>> g_layer_device_data_map;

static layer_instance_data *_get_layer_instance_data(void *obj)
{
    std::lock_guard<std::mutex> lock(g_layer_map_mutex);

    auto it = g_layer_instance_data_map.find(get_dispatch_key(obj));
    assert(it != g_layer_instance_data_map.end());
    if (it == g_layer_instance_data_map.end())
        return nullptr;
    return it->second.get();
}

static layer_device_data *_get_layer_device_data(void *obj)
{
    std::lock_guard<std::mutex> lock(g_layer_map_mutex);

    auto it = g_layer_device_data_map.find(get_dispatch_key(obj));
    assert(it != g_layer_device_data_map.end());
    if (it == g_layer_device_data_map.end())
        return nullptr;
    return it->second.get();
}

static layer_instance_data *get_layer_instance_data(VkInstance instance)
{
    return _get_layer_instance_data(instance);
}

static layer_instance_data *get_layer_instance_data(VkPhysicalDevice physicalDevice)
{
    return _get_layer_instance_data(physicalDevice);
}

static layer_device_data *get_layer_device_data(VkDevice device)
{
    return _get_layer_device_data(device);
}

static layer_device_data *get_layer_device_data(VkQueue commandBuffer)
{
    return _get_layer_device_data(commandBuffer);
}

static layer_device_data *get_layer_device_data(VkCommandBuffer commandBuffer)
{
    return _get_layer_device_data(commandBuffer);
}


static sync_command_buffer *get_sync_command_buffer(VkCommandBuffer commandBuffer, const char *func)
{
    auto device_data = get_layer_device_data(commandBuffer);

    if (LOG_DEBUG(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_NONE, func))
        return nullptr;

    std::lock_guard<std::mutex> lock(device_data->sync_mutex);

    auto it = device_data->sync.command_buffers.find(commandBuffer);
    if (it == device_data->sync.command_buffers.end())
    {
        if (LOG_ERROR(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_INVALID_PARAM,
                "%s called with unknown commandBuffer", func))
            return nullptr;
    }

    return &it->second;
}

static const VkExtensionProperties instance_extensions[] = {{VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_SPEC_VERSION}};

LAYER_FN(VkResult) vkEnumerateInstanceExtensionProperties(
    const char *pLayerName,
    uint32_t *pCount,
    VkExtensionProperties *pProperties)
{
    return util_GetExtensionProperties(1, instance_extensions, pCount, pProperties);
}

LAYER_FN(VkResult) vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice,
    const char *pLayerName,
    uint32_t *pCount,
    VkExtensionProperties *pProperties)
{
    // TODO: why's it written this way?

    if (pLayerName == NULL) {
        auto instance_data = get_layer_instance_data(physicalDevice);
        return instance_data->dispatch.EnumerateDeviceExtensionProperties(physicalDevice, NULL, pCount, pProperties);
    } else {
        return util_GetExtensionProperties(0, nullptr, pCount, pProperties);
    }
}

static const VkLayerProperties global_layers[] = {{
    "VK_LAYER_XXX_sync", VK_LAYER_API_VERSION, 1, "Experimental Validation Layer",
}};

LAYER_FN(VkResult) vkEnumerateInstanceLayerProperties(
    uint32_t *pCount, VkLayerProperties *pProperties)
{
    return util_GetLayerProperties(ARRAY_SIZE(global_layers), global_layers, pCount, pProperties);
}

LAYER_FN(VkResult) vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t *pCount,
    VkLayerProperties *pProperties)
{
    return util_GetLayerProperties(ARRAY_SIZE(global_layers), global_layers, pCount, pProperties);
}

LAYER_FN(VkResult) vkCreateInstance(
    const VkInstanceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkInstance *pInstance)
{
    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(NULL, "vkCreateInstance");
    if (!fpCreateInstance)
        return VK_ERROR_INITIALIZATION_FAILED;

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS)
        return result;

    std::unique_ptr<layer_instance_data> instance_data(new layer_instance_data);
    layer_init_instance_dispatch_table(*pInstance, &instance_data->dispatch, fpGetInstanceProcAddr);

    instance_data->report_data = debug_report_create_instance(
        &instance_data->dispatch, *pInstance,
        pCreateInfo->enabledExtensionCount, pCreateInfo->ppEnabledExtensionNames);

    layer_debug_actions(instance_data->report_data, instance_data->logging_callback, pAllocator, "xxx_sync");

    {
        std::lock_guard<std::mutex> lock(g_layer_map_mutex);
        bool inserted = g_layer_instance_data_map.insert(std::make_pair(
            get_dispatch_key(*pInstance),
            std::move(instance_data))
        ).second;
        assert(inserted == true);
    }

    return VK_SUCCESS;
}

LAYER_FN(void) vkDestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks *pAllocator)
{
    layer_instance_data *instance_data = get_layer_instance_data(instance);
    instance_data->dispatch.DestroyInstance(instance, pAllocator);

    // Clean up logging callback, if any
    while (!instance_data->logging_callback.empty())
    {
        VkDebugReportCallbackEXT callback = instance_data->logging_callback.back();
        layer_destroy_msg_callback(instance_data->report_data, callback, pAllocator);
        instance_data->logging_callback.pop_back();
    }

    layer_debug_report_destroy_instance(instance_data->report_data);

    {
        std::lock_guard<std::mutex> lock(g_layer_map_mutex);
        g_layer_instance_data_map.erase(get_dispatch_key(instance));
    }
}

LAYER_FN(VkResult) vkCreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDevice *pDevice)
{
    auto instance_data = get_layer_instance_data(physicalDevice);

    VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(NULL, "vkCreateDevice");
    if (!fpCreateDevice)
        return VK_ERROR_INITIALIZATION_FAILED;

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS)
        return result;

    std::unique_ptr<layer_device_data> device_data(new layer_device_data);

    layer_init_device_dispatch_table(*pDevice, &device_data->dispatch, fpGetDeviceProcAddr);

    device_data->report_data = layer_debug_report_create_device(instance_data->report_data, *pDevice);

    {
        std::lock_guard<std::mutex> lock(g_layer_map_mutex);
        bool inserted = g_layer_device_data_map.insert(std::make_pair(
            get_dispatch_key(*pDevice),
            std::move(device_data)
        )).second;
        assert(inserted == true);
    }

    return result;
}

LAYER_FN(void) vkDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);
    device_data->dispatch.DestroyDevice(device, pAllocator);

    layer_debug_report_destroy_device(device);

    {
        std::lock_guard<std::mutex> lock(g_layer_map_mutex);
        g_layer_device_data_map.erase(get_dispatch_key(device));
    }
}

LAYER_FN(VkResult) vkCreateDebugReportCallbackEXT(
    VkInstance instance,
    const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugReportCallbackEXT *pMsgCallback)
{
    auto instance_data = get_layer_instance_data(instance);

    VkResult result = instance_data->dispatch.CreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pMsgCallback);
    if (result != VK_SUCCESS)
        return result;

    return layer_create_msg_callback(instance_data->report_data, pCreateInfo, pAllocator, pMsgCallback);
}

LAYER_FN(void) vkDestroyDebugReportCallbackEXT(
    VkInstance instance,
    VkDebugReportCallbackEXT msgCallback,
    const VkAllocationCallbacks *pAllocator)
{
    auto instance_data = get_layer_instance_data(instance);

    instance_data->dispatch.DestroyDebugReportCallbackEXT(instance, msgCallback, pAllocator);

    layer_destroy_msg_callback(instance_data->report_data, msgCallback, pAllocator);
}

LAYER_FN(void) vkDebugReportMessageEXT(
    VkInstance instance,
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT objType,
    uint64_t object,
    size_t location,
    int32_t msgCode,
    const char *pLayerPrefix,
    const char *pMsg)
{
    auto instance_data = get_layer_instance_data(instance);
    instance_data->dispatch.DebugReportMessageEXT(instance, flags, objType, object, location, msgCode, pLayerPrefix, pMsg);
}





LAYER_FN(VkResult) vkQueueSubmit(
    VkQueue queue,
    uint32_t submitCount,
    const VkSubmitInfo *pSubmits,
    VkFence fence)
{
    auto device_data = get_layer_device_data(queue);
    bool skipCall = false;

    if (LOG_DEBUG(device_data, QUEUE, queue, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        for (uint32_t i = 0; i < submitCount; ++i)
        {
            for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; ++j)
            {
                auto command_buffer = pSubmits[i].pCommandBuffers[j];
                skipCall |= LOG_DEBUG(device_data, QUEUE, queue, SYNC_MSG_NONE, " -- submitted: %p", command_buffer);

                auto it = device_data->sync.command_buffers.find(command_buffer);
                if (it == device_data->sync.command_buffers.end())
                {
                    skipCall |= LOG_ERROR(device_data, COMMAND_BUFFER, command_buffer, SYNC_MSG_INVALID_PARAM,
                        "vkQueueSubmit called with unknown pSubmits[%u].pCommandsBuffers[%u]", i, j);
                    continue;
                }

                sync_command_buffer &buf = it->second;

                std::stringstream str;
                for (auto &cmd : buf.commands)
                {
                    str << "    ";
                    cmd->to_string(str);
                    str << "\n";
                }
                skipCall |= LOG_INFO(device_data, COMMAND_BUFFER, command_buffer, SYNC_MSG_NONE,
                    "Command buffer contents:\n%s", str.str().c_str());
            }
        }
    }

    /*
     * CreateCommandPool
     * ResetCommandPool - puts all bufs back in initial state
     * DestroyCommandPool - frees all bufs
     * AllocateCommandBuffers
     * ResetCommandBuffer - puts in initial state
     * FreeCommandBuffers
     * BeginCommandBuffer
     * EndCommandBuffer
     **/

    if (skipCall)
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.QueueSubmit(queue, submitCount, pSubmits, fence);
}

LAYER_FN(VkResult) vkCreateCommandPool(
    VkDevice device,
    const VkCommandPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkCommandPool *pCommandPool)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = device_data->dispatch.CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
    if (result != VK_SUCCESS)
        return result;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        sync_command_pool command_pool;
        command_pool.command_pool = *pCommandPool;

        bool inserted = device_data->sync.command_pools.insert(std::make_pair(
            *pCommandPool,
            command_pool
        )).second;

        if (!inserted)
        {
            if (LOG_ERROR(device_data, COMMAND_POOL, *pCommandPool, SYNC_MSG_INTERNAL_ERROR,
                    "Internal error in vkCreateCommandPool: new pool already exists"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
    }

    return result;
}

LAYER_FN(void) vkDestroyCommandPool(
    VkDevice device,
    VkCommandPool commandPool,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, COMMAND_POOL, commandPool, SYNC_MSG_NONE, __FUNCTION__))
        return;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_pools.find(commandPool);
        if (it == device_data->sync.command_pools.end())
        {
            if (LOG_ERROR(device_data, COMMAND_POOL, commandPool, SYNC_MSG_INVALID_PARAM,
                "vkDestroyCommandPool called with unknown commandPool"))
                return;

        }
        else
        {
            // Remove the device's state for all the buffers in this pool
            for (auto command_buffer : it->second.command_buffers)
            {
                size_t removed = device_data->sync.command_buffers.erase(command_buffer);
                assert(removed == 1);
            }

            // Remove the device's state for this pool
            device_data->sync.command_pools.erase(it);
        }
    }

    device_data->dispatch.DestroyCommandPool(device, commandPool, pAllocator);
}

LAYER_FN(VkResult) vkResetCommandPool(
    VkDevice device,
    VkCommandPool commandPool,
    VkCommandPoolResetFlags flags)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, COMMAND_POOL, commandPool, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_pools.find(commandPool);
        if (it == device_data->sync.command_pools.end())
        {
            if (LOG_ERROR(device_data, COMMAND_POOL, commandPool, SYNC_MSG_INVALID_PARAM,
                    "vkResetCommandPool called with unknown commandPool"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
        else
        {
            for (auto command_buffer : it->second.command_buffers)
            {
                auto it2 = device_data->sync.command_buffers.find(command_buffer);
                assert(it2 != device_data->sync.command_buffers.end());

                it2->second.reset();
            }
        }
    }

    return device_data->dispatch.ResetCommandPool(device, commandPool, flags);
}

LAYER_FN(VkResult) vkAllocateCommandBuffers(
    VkDevice device,
    const VkCommandBufferAllocateInfo *pAllocateInfo,
    VkCommandBuffer *pCommandBuffers)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, COMMAND_POOL, pAllocateInfo->commandPool, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = device_data->dispatch.AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if (result != VK_SUCCESS)
        return result;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_pools.find(pAllocateInfo->commandPool);
        if (it == device_data->sync.command_pools.end())
        {
            if (LOG_ERROR(device_data, COMMAND_POOL, pAllocateInfo->commandPool, SYNC_MSG_INVALID_PARAM,
                    "vkAllocateCommandBuffers called with unknown commandPool"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
        else
        {
            for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; ++i)
            {
                sync_command_buffer command_buffer;

                command_buffer.reset();

                command_buffer.command_buffer = pCommandBuffers[i];
                command_buffer.command_pool = pAllocateInfo->commandPool;
                command_buffer.level = pAllocateInfo->level;

                bool inserted;

                inserted = device_data->sync.command_buffers.insert(std::make_pair(
                    pCommandBuffers[i],
                    std::move(command_buffer)
                )).second;
                if (!inserted)
                {
                    if (LOG_ERROR(device_data, COMMAND_BUFFER, pCommandBuffers[i], SYNC_MSG_INTERNAL_ERROR,
                            "Internal error in vkAllocateCommandBuffers: new buffer already exists"))
                        result = VK_ERROR_VALIDATION_FAILED_EXT;
                }

                inserted = it->second.command_buffers.insert(pCommandBuffers[i]).second;
                assert(inserted);
            }
        }
    }

    return result;
}

LAYER_FN(void) vkFreeCommandBuffers(
    VkDevice device,
    VkCommandPool commandPool,
    uint32_t commandBufferCount,
    const VkCommandBuffer *pCommandBuffers)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, COMMAND_POOL, commandPool, SYNC_MSG_NONE, __FUNCTION__))
        return;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_pools.find(commandPool);
        if (it == device_data->sync.command_pools.end())
        {
            if (LOG_ERROR(device_data, COMMAND_POOL, commandPool, SYNC_MSG_INVALID_PARAM,
                    "vkFreeCommandBuffers called with unknown commandPool"))
                return;
        }
        else
        {
            for (uint32_t i = 0; i < commandBufferCount; ++i)
            {
                size_t removed;

                removed = it->second.command_buffers.erase(pCommandBuffers[i]);
                if (removed != 1)
                {
                    if (LOG_ERROR(device_data, COMMAND_BUFFER, pCommandBuffers[i], SYNC_MSG_INVALID_PARAM,
                            "vkFreeCommandBuffers called with unknown pCommandBuffers[%u]", i))
                        return;
                }
                else
                {
                    removed = device_data->sync.command_buffers.erase(pCommandBuffers[i]);
                    assert(removed == 1);
                }
            }
        }
    }

    return device_data->dispatch.FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}

LAYER_FN(VkResult) vkBeginCommandBuffer(
    VkCommandBuffer commandBuffer,
    const VkCommandBufferBeginInfo *pBeginInfo)
{
    auto device_data = get_layer_device_data(commandBuffer);

    if (LOG_DEBUG(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_buffers.find(commandBuffer);
        if (it == device_data->sync.command_buffers.end())
        {
            if (LOG_ERROR(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_INVALID_PARAM,
                    "vkBeginCommandBuffer called with unknown commandBuffer"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        sync_command_buffer &buf = it->second;

        buf.reset();
        buf.state = sync_command_buffer_state::RECORDING;
        buf.flags = pBeginInfo->flags;

        if (buf.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY)
        {
            buf.renderPass = pBeginInfo->pInheritanceInfo->renderPass;
            buf.subpass = pBeginInfo->pInheritanceInfo->subpass;
            buf.framebuffer = pBeginInfo->pInheritanceInfo->framebuffer;
            buf.occlusionQueryEnable = pBeginInfo->pInheritanceInfo->occlusionQueryEnable;
            buf.queryFlags = pBeginInfo->pInheritanceInfo->queryFlags;
            buf.pipelineStatistics = pBeginInfo->pInheritanceInfo->pipelineStatistics;
        }
    }

    return device_data->dispatch.BeginCommandBuffer(commandBuffer, pBeginInfo);
}

LAYER_FN(VkResult) vkEndCommandBuffer(
    VkCommandBuffer commandBuffer)
{
    auto device_data = get_layer_device_data(commandBuffer);

    if (LOG_DEBUG(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_buffers.find(commandBuffer);
        if (it == device_data->sync.command_buffers.end())
        {
            if (LOG_ERROR(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_INVALID_PARAM,
                    "vkEndCommandBuffer called with unknown commandBuffer"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        it->second.state = sync_command_buffer_state::EXECUTABLE;
    }

    return device_data->dispatch.EndCommandBuffer(commandBuffer);
}

LAYER_FN(VkResult) vkResetCommandBuffer(
    VkCommandBuffer commandBuffer,
    VkCommandBufferResetFlags flags)
{
    auto device_data = get_layer_device_data(commandBuffer);

    if (LOG_DEBUG(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.command_buffers.find(commandBuffer);
        if (it == device_data->sync.command_buffers.end())
        {
            if (LOG_ERROR(device_data, COMMAND_BUFFER, commandBuffer, SYNC_MSG_INVALID_PARAM,
                    "vkResetCommandBuffer called with unknown commandBuffer"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        it->second.reset();
    }

    return device_data->dispatch.ResetCommandBuffer(commandBuffer, flags);
}

LAYER_FN(void) vkCmdDraw(
    VkCommandBuffer commandBuffer,
    uint32_t vertexCount,
    uint32_t instanceCount,
    uint32_t firstVertex,
    uint32_t firstInstance)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_draw(vertexCount, instanceCount, firstVertex, firstInstance));

    device_data->dispatch.CmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}

LAYER_FN(void) vkCmdDrawIndexed(
    VkCommandBuffer commandBuffer,
    uint32_t indexCount,
    uint32_t instanceCount,
    uint32_t firstIndex,
    int32_t vertexOffset,
    uint32_t firstInstance)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance));

    device_data->dispatch.CmdDrawIndexed(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

LAYER_FN(void) vkCmdPipelineBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask,
    VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount,
    const VkMemoryBarrier *pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier *pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier *pImageMemoryBarriers)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_pipeline_barrier(
        srcStageMask, dstStageMask, dependencyFlags,
        memoryBarrierCount, pMemoryBarriers,
        bufferMemoryBarrierCount, pBufferMemoryBarriers,
        imageMemoryBarrierCount, pImageMemoryBarriers));

    device_data->dispatch.CmdPipelineBarrier(commandBuffer,
        srcStageMask, dstStageMask, dependencyFlags,
        memoryBarrierCount, pMemoryBarriers,
        bufferMemoryBarrierCount, pBufferMemoryBarriers,
        imageMemoryBarrierCount, pImageMemoryBarriers);
}

LAYER_FN(void) vkCmdBeginRenderPass(
    VkCommandBuffer commandBuffer,
    const VkRenderPassBeginInfo *pRenderPassBegin,
    VkSubpassContents contents)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_begin_render_pass(pRenderPassBegin, contents));

    device_data->dispatch.CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);
}

LAYER_FN(void) vkCmdNextSubpass(
    VkCommandBuffer commandBuffer,
    VkSubpassContents contents)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_next_subpass(contents));

    device_data->dispatch.CmdNextSubpass(commandBuffer, contents);
}

LAYER_FN(void) vkCmdEndRenderPass(
    VkCommandBuffer commandBuffer)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_end_render_pass());

    device_data->dispatch.CmdEndRenderPass(commandBuffer);
}

LAYER_FN(PFN_vkVoidFunction) vkGetDeviceProcAddr(VkDevice device, const char *funcName)
{
#define X(name) \
    if (!strcmp(funcName, #name)) \
        return (PFN_vkVoidFunction)name;

    X(vkGetDeviceProcAddr);
    // XXX: docs say CreateDevice is required here, which sounds wrong
    X(vkDestroyDevice);

    X(vkQueueSubmit);
// typedef VkResult (VKAPI_PTR *PFN_vkQueueWaitIdle)(VkQueue queue);
// typedef VkResult (VKAPI_PTR *PFN_vkDeviceWaitIdle)(VkDevice device);

// typedef VkResult (VKAPI_PTR *PFN_vkQueueBindSparse)(VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo* pBindInfo, VkFence fence);
// typedef VkResult (VKAPI_PTR *PFN_vkCreateFence)(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence);
// typedef void (VKAPI_PTR *PFN_vkDestroyFence)(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator);
// typedef VkResult (VKAPI_PTR *PFN_vkResetFences)(VkDevice device, uint32_t fenceCount, const VkFence* pFences);
// typedef VkResult (VKAPI_PTR *PFN_vkGetFenceStatus)(VkDevice device, VkFence fence);
// typedef VkResult (VKAPI_PTR *PFN_vkWaitForFences)(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout);
// typedef VkResult (VKAPI_PTR *PFN_vkCreateSemaphore)(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore);
// typedef void (VKAPI_PTR *PFN_vkDestroySemaphore)(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator);
// typedef VkResult (VKAPI_PTR *PFN_vkCreateEvent)(VkDevice device, const VkEventCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkEvent* pEvent);
// typedef void (VKAPI_PTR *PFN_vkDestroyEvent)(VkDevice device, VkEvent event, const VkAllocationCallbacks* pAllocator);
// typedef VkResult (VKAPI_PTR *PFN_vkGetEventStatus)(VkDevice device, VkEvent event);
// typedef VkResult (VKAPI_PTR *PFN_vkSetEvent)(VkDevice device, VkEvent event);
// typedef VkResult (VKAPI_PTR *PFN_vkResetEvent)(VkDevice device, VkEvent event);

// typedef VkResult (VKAPI_PTR *PFN_vkCreateRenderPass)(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass);
// typedef void (VKAPI_PTR *PFN_vkDestroyRenderPass)(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator);

    X(vkCreateCommandPool);
    X(vkDestroyCommandPool);
    X(vkResetCommandPool);
    X(vkAllocateCommandBuffers);
    X(vkFreeCommandBuffers);
    X(vkBeginCommandBuffer);
    X(vkEndCommandBuffer);
    X(vkResetCommandBuffer);

    X(vkCmdDraw);
    X(vkCmdDrawIndexed);
// typedef void (VKAPI_PTR *PFN_vkCmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);
// typedef void (VKAPI_PTR *PFN_vkCmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);
// typedef void (VKAPI_PTR *PFN_vkCmdDispatch)(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
// typedef void (VKAPI_PTR *PFN_vkCmdDispatchIndirect)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);
// typedef void (VKAPI_PTR *PFN_vkCmdCopyBuffer)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions);
// typedef void (VKAPI_PTR *PFN_vkCmdCopyImage)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions);
// typedef void (VKAPI_PTR *PFN_vkCmdBlitImage)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, VkFilter filter);
// typedef void (VKAPI_PTR *PFN_vkCmdCopyBufferToImage)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions);
// typedef void (VKAPI_PTR *PFN_vkCmdCopyImageToBuffer)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions);
// typedef void (VKAPI_PTR *PFN_vkCmdUpdateBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const uint32_t* pData);
// typedef void (VKAPI_PTR *PFN_vkCmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data);
// typedef void (VKAPI_PTR *PFN_vkCmdClearColorImage)(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearColorValue* pColor, uint32_t rangeCount, const VkImageSubresourceRange* pRanges);
// typedef void (VKAPI_PTR *PFN_vkCmdClearDepthStencilImage)(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const VkImageSubresourceRange* pRanges);
// typedef void (VKAPI_PTR *PFN_vkCmdClearAttachments)(VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t rectCount, const VkClearRect* pRects);
// typedef void (VKAPI_PTR *PFN_vkCmdResolveImage)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageResolve* pRegions);
// typedef void (VKAPI_PTR *PFN_vkCmdSetEvent)(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask);
// typedef void (VKAPI_PTR *PFN_vkCmdResetEvent)(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask);
// typedef void (VKAPI_PTR *PFN_vkCmdWaitEvents)(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers);
    X(vkCmdPipelineBarrier);
// typedef void (VKAPI_PTR *PFN_vkCmdBeginQuery)(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags);
// typedef void (VKAPI_PTR *PFN_vkCmdEndQuery)(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query);
// typedef void (VKAPI_PTR *PFN_vkCmdResetQueryPool)(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount);
// typedef void (VKAPI_PTR *PFN_vkCmdWriteTimestamp)(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage, VkQueryPool queryPool, uint32_t query);
// typedef void (VKAPI_PTR *PFN_vkCmdCopyQueryPoolResults)(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags);
// typedef void (VKAPI_PTR *PFN_vkCmdPushConstants)(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues);
    X(vkCmdBeginRenderPass);
    X(vkCmdNextSubpass);
    X(vkCmdEndRenderPass);
// typedef void (VKAPI_PTR *PFN_vkCmdExecuteCommands)(VkCommandBuffer commandBuffer, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers);


#undef X

    if (!device)
        return nullptr;

    auto device_data = get_layer_device_data(device);
    if (!device_data->dispatch.GetDeviceProcAddr)
        return nullptr;
    return device_data->dispatch.GetDeviceProcAddr(device, funcName);
}

LAYER_FN(PFN_vkVoidFunction) vkGetInstanceProcAddr(VkInstance instance, const char *funcName)
{
#define X(name) \
    if (!strcmp(funcName, #name)) \
        return (PFN_vkVoidFunction)name;

    X(vkGetInstanceProcAddr);
    X(vkCreateInstance)
    X(vkDestroyInstance);
    X(vkCreateDevice);

#undef X

    if (!instance)
        return nullptr;

    auto instance_data = get_layer_instance_data(instance);

    PFN_vkVoidFunction fptr = debug_report_get_instance_proc_addr(instance_data->report_data, funcName);
    if (fptr)
        return fptr;

    if (!instance_data->dispatch.GetInstanceProcAddr)
        return nullptr;
    return instance_data->dispatch.GetInstanceProcAddr(instance, funcName);
}


void sync_command_buffer::reset()
{
    state = sync_command_buffer_state::INITIAL;

    flags = 0;

    renderPass = 0;
    subpass = 0;
    framebuffer = 0;
    occlusionQueryEnable = VK_FALSE;
    queryFlags = 0;
    pipelineStatistics = 0;

    commands.clear();
}
