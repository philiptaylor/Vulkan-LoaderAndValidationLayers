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

/**
 * TODO:
 *
 * v1:
 *  Detect incorrect usage of image transitions:
 *    Track memory bound to image
 *    Layout transition barrier counts as read+write events on image's memory
 *    Track image view constructed from image
 *    Track descriptor set layouts
 *    Track descriptor sets
 *    Look at descriptor sets bound during draw (which refer to image views) (assume draw will access all bound memory)
 *    Construct a DAG of read+write events on stages
 *    Complain that the DAG has race conditions
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

#define LOG_INFO(layer_data, objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_INFORMATION_BIT_EXT,         layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_WARN(layer_data, objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_WARNING_BIT_EXT,             layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_PERF(layer_data, objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_ERROR(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_ERROR_BIT_EXT,               layer_data, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_DEBUG(layer_data, objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_DEBUG_BIT_EXT,               layer_data, objType, object, messageCode, fmt, __VA_ARGS__)

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
    uint32_t *pCount,
    VkLayerProperties *pProperties)
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

    // layer_init_device_dispatch_table doesn't do extensions, so do those manually
    device_data->dispatch.CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)fpGetDeviceProcAddr(*pDevice, "vkCreateSwapchainKHR");
    device_data->dispatch.DestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)fpGetDeviceProcAddr(*pDevice, "vkDestroySwapchainKHR");
    device_data->dispatch.GetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)fpGetDeviceProcAddr(*pDevice, "vkGetSwapchainImagesKHR");
    device_data->dispatch.AcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)fpGetDeviceProcAddr(*pDevice, "vkAcquireNextImageKHR");
    device_data->dispatch.QueuePresentKHR = (PFN_vkQueuePresentKHR)fpGetDeviceProcAddr(*pDevice, "vkQueuePresentKHR");

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


static bool dump_command_buffer(
    layer_device_data *device_data,
    sync_command_buffer &buf)
{
    VkPipeline graphics_pipeline = VK_NULL_HANDLE;
    VkPipeline compute_pipeline = VK_NULL_HANDLE;

    for (auto &cmd : buf.commands)
    {
        cmd->update_pipeline_binding(&graphics_pipeline, &compute_pipeline);
        if (cmd->is_draw())
        {
            if (graphics_pipeline == VK_NULL_HANDLE)
            {
                return LOG_ERROR(device_data, COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with no pipeline bound");
            }

            auto pipeline = device_data->sync.graphics_pipelines.find(graphics_pipeline);
            if (pipeline == device_data->sync.graphics_pipelines.end())
            {
                return LOG_ERROR(device_data, COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with unknown pipeline bound");
            }

            auto layout = device_data->sync.pipeline_layouts.find(pipeline->second.layout);
            if (layout == device_data->sync.pipeline_layouts.end())
            {
                return LOG_ERROR(device_data, COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with pipeline with unknown pipeline layout");
            }

            std::stringstream str;
            str << "Draw command: ";
            cmd->to_string(str);
            str << "\n    Current pipeline:\n      ";
            pipeline->second.to_string(str);
            str << "\n    Current pipeline layout:\n      ";
            layout->second.to_string(str);
            for (auto &setLayout : layout->second.setLayouts)
            {
                auto set_layout = device_data->sync.descriptor_set_layouts.find(setLayout);
                if (set_layout == device_data->sync.descriptor_set_layouts.end())
                {
                    return LOG_ERROR(device_data, COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with pipeline layout with unknown descriptor set layout");
                }

                str << "\n        ";
                set_layout->second.to_string(str);
            }

            if (LOG_INFO(device_data, COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "%s", str.str().c_str()))
                return true;
        }
    }

    return false;
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

                {
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

                skipCall |= dump_command_buffer(device_data, buf);

            }
        }
    }

    if (skipCall)
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.QueueSubmit(queue, submitCount, pSubmits, fence);
}

LAYER_FN(VkResult) vkQueueWaitIdle(
    VkQueue queue)
{
    auto device_data = get_layer_device_data(queue);

    if (LOG_DEBUG(device_data, QUEUE, queue, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.QueueWaitIdle(queue);
}

LAYER_FN(VkResult) vkDeviceWaitIdle(
    VkDevice device)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.DeviceWaitIdle(device);
}

// LAYER_FN(VkResult) vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);
// LAYER_FN(void) vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator);
// LAYER_FN(VkResult) vkMapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData);
// LAYER_FN(void) vkUnmapMemory(VkDevice device, VkDeviceMemory memory);
// LAYER_FN(VkResult) vkFlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges);
// LAYER_FN(VkResult) vkInvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges);
//
// LAYER_FN(VkResult) vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset);
// LAYER_FN(VkResult) vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset);

// LAYER_FN(VkResult) vkQueueBindSparse(VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo* pBindInfo, VkFence fence);
// LAYER_FN(VkResult) vkCreateFence(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence);
// LAYER_FN(void) vkDestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator);
// LAYER_FN(VkResult) vkResetFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences);
// LAYER_FN(VkResult) vkGetFenceStatus(VkDevice device, VkFence fence);
// LAYER_FN(VkResult) vkWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout);

LAYER_FN(VkResult) vkCreateSemaphore(
    VkDevice device,
    const VkSemaphoreCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkSemaphore *pSemaphore)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
}

LAYER_FN(void) vkDestroySemaphore(
    VkDevice device,
    VkSemaphore semaphore,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, SEMAPHORE, semaphore, SYNC_MSG_NONE, __FUNCTION__))
        return;

    return device_data->dispatch.DestroySemaphore(device, semaphore, pAllocator);
}

// LAYER_FN(VkResult) vkCreateEvent(VkDevice device, const VkEventCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkEvent* pEvent);
// LAYER_FN(void) vkDestroyEvent)(VkDevice device, VkEvent event, const VkAllocationCallbacks* pAllocator);
// LAYER_FN(VkResult) vkGetEventStatus(VkDevice device, VkEvent event);
// LAYER_FN(VkResult) vkSetEvent(VkDevice device, VkEvent event);
// LAYER_FN(VkResult) vkResetEvent(VkDevice device, VkEvent event);

// LAYER_FN(VkResult) vkCreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool);
// LAYER_FN(void) vkDestroyQueryPool(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator);
// LAYER_FN(VkResult) vkGetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags);
//
// LAYER_FN(VkResult) vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer);
// LAYER_FN(void) vkDestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator);
//
// LAYER_FN(VkResult) vkCreateBufferView(VkDevice device, const VkBufferViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBufferView* pView);
// LAYER_FN(void) vkDestroyBufferView(VkDevice device, VkBufferView bufferView, const VkAllocationCallbacks* pAllocator);
//
// LAYER_FN(VkResult) vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage);
// LAYER_FN(void) vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator);
//
// LAYER_FN(VkResult) vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView);
// LAYER_FN(void) vkDestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator);
//
// LAYER_FN(VkResult) vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule);
// LAYER_FN(void) vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator);

LAYER_FN(VkResult) vkCreateGraphicsPipelines(
    VkDevice device,
    VkPipelineCache pipelineCache,
    uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo *pCreateInfos,
    const VkAllocationCallbacks *pAllocator,
    VkPipeline *pPipelines)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = device_data->dispatch.CreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    if (result != VK_SUCCESS)
        return result;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        for (uint32_t p = 0; p < createInfoCount; ++p)
        {
            sync_graphics_pipeline pipeline;
            pipeline.pipeline = pPipelines[p];

            auto &createInfo = pCreateInfos[p];

            pipeline.flags = createInfo.flags;

            for (uint32_t i = 0; i < createInfo.stageCount; ++i)
            {
                sync_graphics_pipeline::shader_stage stage;
                stage.flags = createInfo.pStages[i].flags;
                stage.stage = createInfo.pStages[i].stage;
                stage.module = createInfo.pStages[i].module;
                stage.name = createInfo.pStages[i].pName;

                pipeline.stages.push_back(std::move(stage));
            }

            pipeline.vertexInputState.flags = createInfo.pVertexInputState->flags;
            pipeline.vertexInputState.vertexBindingDescriptions = std::vector<VkVertexInputBindingDescription>(
                createInfo.pVertexInputState->pVertexBindingDescriptions,
                createInfo.pVertexInputState->pVertexBindingDescriptions + createInfo.pVertexInputState->vertexBindingDescriptionCount
            );
            pipeline.vertexInputState.vertexAttributeDescriptions = std::vector<VkVertexInputAttributeDescription>(
                createInfo.pVertexInputState->pVertexAttributeDescriptions,
                createInfo.pVertexInputState->pVertexAttributeDescriptions + createInfo.pVertexInputState->vertexAttributeDescriptionCount
            );

            pipeline.inputAssemblyState.flags = createInfo.pInputAssemblyState->flags;
            pipeline.inputAssemblyState.topology = createInfo.pInputAssemblyState->topology;
            pipeline.inputAssemblyState.primitiveRestartEnable = createInfo.pInputAssemblyState->primitiveRestartEnable;

            pipeline.layout = createInfo.layout;
            pipeline.renderPass = createInfo.renderPass;
            pipeline.subpass = createInfo.subpass;

            bool inserted = device_data->sync.graphics_pipelines.insert(std::make_pair(
                pPipelines[p],
                pipeline
            )).second;

            if (!inserted)
            {
                if (LOG_ERROR(device_data, PIPELINE, pPipelines[p], SYNC_MSG_INTERNAL_ERROR,
                        "Internal error in vkCreateGraphicsPipelines: new pipeline already exists"))
                    return VK_ERROR_VALIDATION_FAILED_EXT;
            }
        }
    }

    return result;
}

// LAYER_FN(VkResult) vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);

LAYER_FN(void) vkDestroyPipeline(
    VkDevice device,
    VkPipeline pipeline,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, PIPELINE, pipeline, SYNC_MSG_NONE, __FUNCTION__))
        return;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.graphics_pipelines.find(pipeline);
        if (it == device_data->sync.graphics_pipelines.end())
        {
            // TODO: this might be a compute pipeline, so check that too

            if (LOG_ERROR(device_data, PIPELINE, pipeline, SYNC_MSG_INVALID_PARAM,
                    "vkDestroyPipeline called with unknown pipeline"))
                return;
        }
        else
        {
            device_data->sync.graphics_pipelines.erase(it);
        }
    }

    device_data->dispatch.DestroyPipeline(device, pipeline, pAllocator);
}

LAYER_FN(VkResult) vkCreatePipelineLayout(
    VkDevice device,
    const VkPipelineLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkPipelineLayout *pPipelineLayout)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = device_data->dispatch.CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);
    if (result != VK_SUCCESS)
        return result;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        sync_pipeline_layout pipeline_layout;
        pipeline_layout.pipeline_layout = *pPipelineLayout;

        pipeline_layout.flags = pCreateInfo->flags;
        pipeline_layout.setLayouts = std::vector<VkDescriptorSetLayout>(pCreateInfo->pSetLayouts, pCreateInfo->pSetLayouts + pCreateInfo->setLayoutCount);
        pipeline_layout.pushConstantRanges = std::vector<VkPushConstantRange>(pCreateInfo->pPushConstantRanges, pCreateInfo->pPushConstantRanges + pCreateInfo->pushConstantRangeCount);

        bool inserted = device_data->sync.pipeline_layouts.insert(std::make_pair(
            *pPipelineLayout,
            pipeline_layout
        )).second;

        if (!inserted)
        {
            if (LOG_ERROR(device_data, PIPELINE_LAYOUT, *pPipelineLayout, SYNC_MSG_INTERNAL_ERROR,
                    "Internal error in vkCreatePipelineLayout: new pipeline layout already exists"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
    }

    return result;
}

LAYER_FN(void) vkDestroyPipelineLayout(
    VkDevice device,
    VkPipelineLayout pipelineLayout,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, PIPELINE_LAYOUT, pipelineLayout, SYNC_MSG_NONE, __FUNCTION__))
        return;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.pipeline_layouts.find(pipelineLayout);
        if (it == device_data->sync.pipeline_layouts.end())
        {
            if (LOG_ERROR(device_data, PIPELINE_LAYOUT, pipelineLayout, SYNC_MSG_INVALID_PARAM,
                    "vkDestroyPipelineLayout called with unknown pipelineLayout"))
                return;
        }
        else
        {
            device_data->sync.pipeline_layouts.erase(it);
        }
    }

    device_data->dispatch.DestroyPipelineLayout(device, pipelineLayout, pAllocator);
}

// LAYER_FN(VkResult) vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler);
// LAYER_FN(void) vkDestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator);

LAYER_FN(VkResult) vkCreateDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDescriptorSetLayout *pSetLayout)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = device_data->dispatch.CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);
    if (result != VK_SUCCESS)
        return result;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        sync_descriptor_set_layout set_layout;
        set_layout.descriptor_set_layout = *pSetLayout;

        set_layout.flags = pCreateInfo->flags;
        for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i)
        {
            sync_descriptor_set_layout::descriptor_set_layout_binding new_binding;
            auto &binding = pCreateInfo->pBindings[i];

            new_binding.binding = binding.binding;
            new_binding.descriptorType = binding.descriptorType;
            new_binding.descriptorCount = binding.descriptorCount;
            new_binding.stageFlags = binding.stageFlags;
            if (binding.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER || binding.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                if (binding.pImmutableSamplers != nullptr)
                    new_binding.immutableSamplers = std::vector<VkSampler>(binding.pImmutableSamplers, binding.pImmutableSamplers + binding.descriptorCount);

            set_layout.bindings.push_back(std::move(new_binding));
        }

        bool inserted = device_data->sync.descriptor_set_layouts.insert(std::make_pair(
            *pSetLayout,
            set_layout
        )).second;

        if (!inserted)
        {
            if (LOG_ERROR(device_data, DESCRIPTOR_SET_LAYOUT, *pSetLayout, SYNC_MSG_INTERNAL_ERROR,
                    "Internal error in vkCreateDescriptorSetLayout: new descriptor set layout already exists"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
    }

    return result;
}

LAYER_FN(void) vkDestroyDescriptorSetLayout(
    VkDevice device,
    VkDescriptorSetLayout descriptorSetLayout,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DESCRIPTOR_SET_LAYOUT, descriptorSetLayout, SYNC_MSG_NONE, __FUNCTION__))
        return;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.descriptor_set_layouts.find(descriptorSetLayout);
        if (it == device_data->sync.descriptor_set_layouts.end())
        {
            if (LOG_ERROR(device_data, DESCRIPTOR_SET_LAYOUT, descriptorSetLayout, SYNC_MSG_INVALID_PARAM,
                    "vkDestroyDescriptorSetLayout called with unknown descriptorSetLayout"))
                return;
        }
        else
        {
            device_data->sync.descriptor_set_layouts.erase(it);
        }
    }

    device_data->dispatch.DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);
}

// LAYER_FN(VkResult) vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool);
// LAYER_FN(void) vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator);
// LAYER_FN(VkResult) vkResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags);
//
// LAYER_FN(VkResult) vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets);
// LAYER_FN(VkResult) vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets);
// LAYER_FN(void) vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies);
//
// LAYER_FN(VkResult) vkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer);
// LAYER_FN(void) vkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* pAllocator);

LAYER_FN(VkResult) vkCreateRenderPass(
    VkDevice device,
    const VkRenderPassCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkRenderPass *pRenderPass)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, DEVICE, device, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = device_data->dispatch.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (result != VK_SUCCESS)
        return result;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        sync_render_pass render_pass;
        render_pass.render_pass = *pRenderPass;

        render_pass.flags = pCreateInfo->flags;
        render_pass.attachments = std::vector<VkAttachmentDescription>(pCreateInfo->pAttachments, pCreateInfo->pAttachments + pCreateInfo->attachmentCount);
        for (uint32_t i = 0; i < pCreateInfo->subpassCount; ++i)
        {
            sync_render_pass::subpass_description desc;
            auto &subpass = pCreateInfo->pSubpasses[i];

            desc.flags = subpass.flags;
            desc.pipelineBindPoint = subpass.pipelineBindPoint;
            desc.inputAttachments = std::vector<VkAttachmentReference>(subpass.pInputAttachments, subpass.pInputAttachments + subpass.inputAttachmentCount);
            desc.colorAttachments = std::vector<VkAttachmentReference>(subpass.pColorAttachments, subpass.pColorAttachments + subpass.colorAttachmentCount);
            if (subpass.pResolveAttachments)
                desc.resolveAttachments = std::vector<VkAttachmentReference>(subpass.pResolveAttachments, subpass.pResolveAttachments + subpass.colorAttachmentCount);
            if (subpass.pDepthStencilAttachment)
                desc.depthStencilAttachment.push_back(*subpass.pDepthStencilAttachment);
            desc.preserveAttachments = std::vector<uint32_t>(subpass.pPreserveAttachments, subpass.pPreserveAttachments + subpass.preserveAttachmentCount);

            render_pass.subpasses.push_back(std::move(desc));
        }
        render_pass.dependencies = std::vector<VkSubpassDependency>(pCreateInfo->pDependencies, pCreateInfo->pDependencies + pCreateInfo->dependencyCount);

        bool inserted = device_data->sync.render_passes.insert(std::make_pair(
            *pRenderPass,
            render_pass
        )).second;

        if (!inserted)
        {
            if (LOG_ERROR(device_data, RENDER_PASS, *pRenderPass, SYNC_MSG_INTERNAL_ERROR,
                    "Internal error in vkCreateRenderPass: new render pass already exists"))
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
    }

    return result;
}

LAYER_FN(void) vkDestroyRenderPass(
    VkDevice device,
    VkRenderPass renderPass,
    const VkAllocationCallbacks *pAllocator)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, RENDER_PASS, renderPass, SYNC_MSG_NONE, __FUNCTION__))
        return;

    {
        std::lock_guard<std::mutex> lock(device_data->sync_mutex);

        auto it = device_data->sync.render_passes.find(renderPass);
        if (it == device_data->sync.render_passes.end())
        {
            if (LOG_ERROR(device_data, RENDER_PASS, renderPass, SYNC_MSG_INVALID_PARAM,
                    "vkDestroyRenderPass called with unknown renderPass"))
                return;
        }
        else
        {
            device_data->sync.render_passes.erase(it);
        }
    }

    device_data->dispatch.DestroyRenderPass(device, renderPass, pAllocator);
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



LAYER_FN(void) vkCmdBindPipeline(
    VkCommandBuffer commandBuffer,
    VkPipelineBindPoint pipelineBindPoint,
    VkPipeline pipeline)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_bind_pipeline(pipelineBindPoint, pipeline));

    device_data->dispatch.CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

LAYER_FN(void) vkCmdSetViewport(
    VkCommandBuffer commandBuffer,
    uint32_t firstViewport,
    uint32_t viewportCount,
    const VkViewport *pViewports)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_set_viewport(firstViewport, viewportCount, pViewports));

    device_data->dispatch.CmdSetViewport(commandBuffer, firstViewport, viewportCount, pViewports);
}

LAYER_FN(void) vkCmdSetScissor(
    VkCommandBuffer commandBuffer,
    uint32_t firstScissor,
    uint32_t scissorCount,
    const VkRect2D *pScissors)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_set_scissor(firstScissor, scissorCount, pScissors));

    device_data->dispatch.CmdSetScissor(commandBuffer, firstScissor, scissorCount, pScissors);
}

// LAYER_FN(void) vkCmdSetLineWidth(VkCommandBuffer commandBuffer, float lineWidth);
// LAYER_FN(void) vkCmdSetDepthBias(VkCommandBuffer commandBuffer, float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor);
// LAYER_FN(void) vkCmdSetBlendConstants(VkCommandBuffer commandBuffer, const float blendConstants[4]);
// LAYER_FN(void) vkCmdSetDepthBounds(VkCommandBuffer commandBuffer, float minDepthBounds, float maxDepthBounds);
// LAYER_FN(void) vkCmdSetStencilCompareMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t compareMask);
// LAYER_FN(void) vkCmdSetStencilWriteMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t writeMask);
// LAYER_FN(void) vkCmdSetStencilReference(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t reference);

LAYER_FN(void) vkCmdBindDescriptorSets(
    VkCommandBuffer commandBuffer,
    VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout,
    uint32_t firstSet,
    uint32_t descriptorSetCount,
    const VkDescriptorSet *pDescriptorSets,
    uint32_t dynamicOffsetCount,
    const uint32_t *pDynamicOffsets)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_bind_descriptor_sets(pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets));

    device_data->dispatch.CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

// LAYER_FN(void) vkCmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType);

LAYER_FN(void) vkCmdBindVertexBuffers(
    VkCommandBuffer commandBuffer,
    uint32_t firstBinding,
    uint32_t bindingCount,
    const VkBuffer *pBuffers,
    const VkDeviceSize *pOffsets)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_bind_vertex_buffers(firstBinding, bindingCount, pBuffers, pOffsets));

    device_data->dispatch.CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
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

// LAYER_FN(void) vkCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);
// LAYER_FN(void) vkCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);
// LAYER_FN(void) vkCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z);
// LAYER_FN(void) vkCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset);
// LAYER_FN(void) vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions);

LAYER_FN(void) vkCmdCopyImage(
    VkCommandBuffer commandBuffer,
    VkImage srcImage,
    VkImageLayout srcImageLayout,
    VkImage dstImage,
    VkImageLayout dstImageLayout,
    uint32_t regionCount,
    const VkImageCopy *pRegions)
{
    auto device_data = get_layer_device_data(commandBuffer);

    sync_command_buffer *buf = get_sync_command_buffer(commandBuffer, __func__);
    if (!buf)
        return;

    buf->commands.emplace_back(new sync_cmd_copy_image(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions));

    device_data->dispatch.CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

// LAYER_FN(void) vkCmdBlitImage)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, VkFilter filter);
// LAYER_FN(void) vkCmdCopyBufferToImage)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions);
// LAYER_FN(void) vkCmdCopyImageToBuffer)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions);
// LAYER_FN(void) vkCmdUpdateBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const uint32_t* pData);
// LAYER_FN(void) vkCmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data);
// LAYER_FN(void) vkCmdClearColorImage)(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearColorValue* pColor, uint32_t rangeCount, const VkImageSubresourceRange* pRanges);
// LAYER_FN(void) vkCmdClearDepthStencilImage)(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const VkImageSubresourceRange* pRanges);
// LAYER_FN(void) vkCmdClearAttachments)(VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t rectCount, const VkClearRect* pRects);
// LAYER_FN(void) vkCmdResolveImage)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageResolve* pRegions);
// LAYER_FN(void) vkCmdSetEvent)(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask);
// LAYER_FN(void) vkCmdResetEvent)(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask);
// LAYER_FN(void) vkCmdWaitEvents)(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers);

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

// LAYER_FN(void) vkCmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags);
// LAYER_FN(void) vkCmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query);
// LAYER_FN(void) vkCmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount);
// LAYER_FN(void) vkCmdWriteTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage, VkQueryPool queryPool, uint32_t query);
// LAYER_FN(void) vkCmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags);
// LAYER_FN(void) vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues);

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

// LAYER_FN(void) vkCmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers);

// LAYER_FN(VkResult) vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain);
// LAYER_FN(void) vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator);
// LAYER_FN(VkResult) vkGetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages);

LAYER_FN(VkResult) vkAcquireNextImageKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    uint64_t timeout,
    VkSemaphore semaphore,
    VkFence fence,
    uint32_t *pImageIndex)
{
    auto device_data = get_layer_device_data(device);

    if (LOG_DEBUG(device_data, SWAPCHAIN_KHR, swapchain, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);
}

LAYER_FN(VkResult) vkQueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR *pPresentInfo)
{
    auto device_data = get_layer_device_data(queue);

    if (LOG_DEBUG(device_data, QUEUE, queue, SYNC_MSG_NONE, __FUNCTION__))
        return VK_ERROR_VALIDATION_FAILED_EXT;

    return device_data->dispatch.QueuePresentKHR(queue, pPresentInfo);
}

// LAYER_FN(VkResult) vkCreateSharedSwapchainsKHR(VkDevice device, uint32_t swapchainCount, const VkSwapchainCreateInfoKHR* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchains);




LAYER_FN(PFN_vkVoidFunction) vkGetDeviceProcAddr(VkDevice device, const char *funcName)
{
#define X(name) \
    if (!strcmp(funcName, #name)) \
        return (PFN_vkVoidFunction)name;

    X(vkGetDeviceProcAddr);
    // XXX: docs say CreateDevice is required here, which sounds wrong
    X(vkDestroyDevice);

    X(vkQueueSubmit);
    X(vkQueueWaitIdle);
    X(vkDeviceWaitIdle);

// X(vkAllocateMemory);
// X(vkFreeMemory);
// X(vkMapMemory);
// X(vkUnmapMemory);
// X(vkFlushMappedMemoryRanges);
// X(vkInvalidateMappedMemoryRanges);
//
// X(vkBindBufferMemory);
// X(vkBindImageMemory);

// X(vkQueueBindSparse);
// X(vkCreateFence);
// X(vkDestroyFence);
// X(vkResetFences);
// X(vkGetFenceStatus);
// X(vkWaitForFences);
    X(vkCreateSemaphore);
    X(vkDestroySemaphore);
// X(vkCreateEvent);
// X(vkDestroyEvent);
// X(vkGetEventStatus);
// X(vkSetEvent);
// X(vkResetEvent);

// X(vkCreateQueryPool);
// X(vkDestroyQueryPool);
// X(vkGetQueryPoolResults);
//
// X(vkCreateBuffer);
// X(vkDestroyBuffer);
//
// X(vkCreateBufferView);
// X(vkDestroyBufferView);
//
// X(vkCreateImage);
// X(vkDestroyImage);
//
// X(vkCreateImageView);
// X(vkDestroyImageView);
//
// X(vkCreateShaderModule);
// X(vkDestroyShaderModule);

    X(vkCreateGraphicsPipelines);
// X(vkCreateComputePipelines);
    X(vkDestroyPipeline);

    X(vkCreatePipelineLayout);
    X(vkDestroyPipelineLayout);

// X(vkCreateSampler);
// X(vkDestroySampler);

    X(vkCreateDescriptorSetLayout);
    X(vkDestroyDescriptorSetLayout);

// X(vkCreateDescriptorPool);
// X(vkDestroyDescriptorPool);
// X(vkResetDescriptorPool);
//
// X(vkAllocateDescriptorSets);
// X(vkFreeDescriptorSets);
// X(vkUpdateDescriptorSets);
//
// X(vkCreateFramebuffer);
// X(vkDestroyFramebuffer);

    X(vkCreateRenderPass);
    X(vkDestroyRenderPass);

    X(vkCreateCommandPool);
    X(vkDestroyCommandPool);
    X(vkResetCommandPool);
    X(vkAllocateCommandBuffers);
    X(vkFreeCommandBuffers);
    X(vkBeginCommandBuffer);
    X(vkEndCommandBuffer);
    X(vkResetCommandBuffer);

    X(vkCmdBindPipeline);
    X(vkCmdSetViewport);
    X(vkCmdSetScissor);
// X(vkCmdSetDepthBias);
// X(vkCmdSetBlendConstants);
// X(vkCmdSetDepthBounds);
// X(vkCmdSetStencilCompareMask);
// X(vkCmdSetStencilWriteMask);
// X(vkCmdSetStencilReference);
    X(vkCmdBindDescriptorSets);
// X(vkCmdBindIndexBuffer);
    X(vkCmdBindVertexBuffers);
    X(vkCmdDraw);
    X(vkCmdDrawIndexed);
// X(vkCmdDrawIndirect);
// X(vkCmdDrawIndexedIndirect);
// X(vkCmdDispatch);
// X(vkCmdDispatchIndirect);
// X(vkCmdCopyBuffer);
    X(vkCmdCopyImage);
// X(vkCmdBlitImage);
// X(vkCmdCopyBufferToImage);
// X(vkCmdCopyImageToBuffer);
// X(vkCmdUpdateBuffer);
// X(vkCmdFillBuffer);
// X(vkCmdClearColorImage);
// X(vkCmdClearDepthStencilImage);
// X(vkCmdClearAttachments);
// X(vkCmdResolveImage);
// X(vkCmdSetEvent);
// X(vkCmdResetEvent);
// X(vkCmdWaitEvents);
    X(vkCmdPipelineBarrier);
// X(vkCmdBeginQuery);
// X(vkCmdEndQuery);
// X(vkCmdResetQueryPool);
// X(vkCmdWriteTimestamp);
// X(vkCmdCopyQueryPoolResults);
// X(vkCmdPushConstants);
    X(vkCmdBeginRenderPass);
    X(vkCmdNextSubpass);
    X(vkCmdEndRenderPass);
// X(vkCmdExecuteCommands);

// X(vkCreateSwapchainKHR);
// X(vkDestroySwapchainKHR);
// X(vkGetSwapchainImagesKHR);
    X(vkAcquireNextImageKHR);
    X(vkQueuePresentKHR);

// X(vkCreateSharedSwapchainsKHR);


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
