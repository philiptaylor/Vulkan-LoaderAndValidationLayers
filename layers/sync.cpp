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

#include <unordered_map>
#include <mutex>

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

static layer_device_data *get_layer_device_data(VkCommandBuffer commandBuffer)
{
    return _get_layer_device_data(commandBuffer);
}


static const VkExtensionProperties instance_extensions[] = {{VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_SPEC_VERSION}};

LAYER_FN(VkResult) vkEnumerateInstanceExtensionProperties(
    const char *pLayerName, uint32_t *pCount, VkExtensionProperties *pProperties)
{
    return util_GetExtensionProperties(1, instance_extensions, pCount, pProperties);
}

LAYER_FN(VkResult) vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice,
    const char *pLayerName, uint32_t *pCount, VkExtensionProperties *pProperties)
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
    VkPhysicalDevice physicalDevice, uint32_t *pCount, VkLayerProperties *pProperties)
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
    VkInstance instance, const VkAllocationCallbacks *pAllocator)
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

LAYER_FN(void) vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator)
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
    const VkAllocationCallbacks *pAllocator, VkDebugReportCallbackEXT *pMsgCallback)
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
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t object,
    size_t location, int32_t msgCode, const char *pLayerPrefix, const char *pMsg)
{
    auto instance_data = get_layer_instance_data(instance);
    instance_data->dispatch.DebugReportMessageEXT(instance, flags, objType, object, location, msgCode, pLayerPrefix, pMsg);
}







LAYER_FN(VkResult) vkBeginCommandBuffer(
    VkCommandBuffer commandBuffer,
    const VkCommandBufferBeginInfo *pBeginInfo)
{
    bool skipCall = false;
    auto device_data = get_layer_device_data(commandBuffer);

    skipCall |= LOG_DEBUG(device_data, COMMAND_BUFFER, commandBuffer, DEVLIMITS_INVALID_INHERITED_QUERY, "TEST2 22222.");
    skipCall |= LOG_DEBUG(device_data, UNKNOWN, nullptr, DEVLIMITS_INVALID_INHERITED_QUERY, "TEST2 33333.");

    skipCall |= log_msg(
            device_data->report_data, VK_DEBUG_REPORT_DEBUG_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
            reinterpret_cast<uint64_t>(commandBuffer), __LINE__, DEVLIMITS_INVALID_INHERITED_QUERY, "SYNC",
            "TEST TEST TEST.");

    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    if (!skipCall)
        result = device_data->dispatch.BeginCommandBuffer(commandBuffer, pBeginInfo);
    return result;
}





LAYER_FN(PFN_vkVoidFunction) vkGetDeviceProcAddr(VkDevice device, const char *funcName)
{
#define X(name) \
    if (!strcmp(funcName, #name)) \
        return (PFN_vkVoidFunction)name;

    X(vkGetDeviceProcAddr);
    // XXX: docs say CreateDevice is required here, which sounds wrong
    X(vkDestroyDevice);

    X(vkBeginCommandBuffer);

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
