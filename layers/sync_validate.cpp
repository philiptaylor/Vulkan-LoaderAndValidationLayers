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

#include "sync.h"

#include <sstream>

#define _LOG_GENERIC(level, objType, object, messageCode, fmt, ...) \
    log_msg(mReportData, VK_DEBUG_REPORT_DEBUG_BIT_EXT, \
        VK_DEBUG_REPORT_OBJECT_TYPE_##objType##_EXT, (uint64_t)(object), \
        __LINE__, (messageCode), "SYNC", (fmt), __VA_ARGS__)

#define LOG_INFO(objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_INFORMATION_BIT_EXT,         objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_WARN(objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_WARNING_BIT_EXT,             objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_PERF(objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_ERROR(objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_ERROR_BIT_EXT,               objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_DEBUG(objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_DEBUG_BIT_EXT,               objType, object, messageCode, fmt, __VA_ARGS__)

SyncValidator::SyncValidator(sync_device &syncDevice, debug_report_data *reportData)
    : mSyncDevice(syncDevice), mReportData(reportData)
{
}

bool SyncValidator::submitCmdBuffer(VkQueue queue, const sync_command_buffer& buf)
{
    // XXX: need mutex on mSyncDevice

    VkPipeline graphics_pipeline = VK_NULL_HANDLE;
    VkPipeline compute_pipeline = VK_NULL_HANDLE;

    struct Binding
    {
        sync_descriptor_set *descriptor_set;
        uint32_t dynamic_offset; // TODO
    };

    std::map<uint32_t, Binding> graphics_bindings;
    std::map<uint32_t, Binding> compute_bindings;

    for (auto &cmd : buf.commands)
    {
        // TODO: reset state on vkCmdExecuteCommands

        auto bind_pipeline = cmd->as_bind_pipeline();
        if (bind_pipeline)
        {
            if (bind_pipeline->pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
                graphics_pipeline = bind_pipeline->pipeline;
            else if (bind_pipeline->pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
                compute_pipeline = bind_pipeline->pipeline;
        }

        auto bind_descriptor_sets = cmd->as_bind_descriptor_sets();
        if (bind_descriptor_sets)
        {
            // TODO: should look at pipeline layout compatibility here
            // TODO: dynamic offsets

            for (uint32_t i = 0; i < bind_descriptor_sets->descriptorSets.size(); ++i)
            {
                uint32_t set_number = bind_descriptor_sets->firstSet + i;

                auto descriptor_set = mSyncDevice.descriptor_sets.find(bind_descriptor_sets->descriptorSets[i]);
                if (descriptor_set == mSyncDevice.descriptor_sets.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with unknown descriptor set bound");
                }

                Binding binding;
                binding.descriptor_set = &descriptor_set->second;
                binding.dynamic_offset = 0;

                if (bind_descriptor_sets->pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
                    graphics_bindings[set_number] = binding;
                else if (bind_descriptor_sets->pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
                    compute_bindings[set_number] = binding;
            }
        }

        if (cmd->is_draw())
        {
            if (graphics_pipeline == VK_NULL_HANDLE)
            {
                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with no pipeline bound");
            }

            auto pipeline = mSyncDevice.graphics_pipelines.find(graphics_pipeline);
            if (pipeline == mSyncDevice.graphics_pipelines.end())
            {
                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with unknown pipeline bound");
            }

            auto pipeline_layout = mSyncDevice.pipeline_layouts.find(pipeline->second.layout);
            if (pipeline_layout == mSyncDevice.pipeline_layouts.end())
            {
                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with pipeline with unknown pipeline layout");
            }

            std::stringstream str;
            str << "Draw command: ";
            cmd->to_string(str);
            str << "\n    Current pipeline:\n      ";
            pipeline->second.to_string(str);
            str << "\n    Current pipeline layout:\n      ";
            pipeline_layout->second.to_string(str);
            for (auto &setLayout : pipeline_layout->second.setLayouts)
            {
                auto set_layout = mSyncDevice.descriptor_set_layouts.find(setLayout);
                if (set_layout == mSyncDevice.descriptor_set_layouts.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with pipeline layout with unknown descriptor set layout");
                }

                str << "\n        ";
                set_layout->second.to_string(str);
            }
            str << "\n    Current bindings:\n";
            for (auto &binding : graphics_bindings)
            {
                str << "      " << binding.first << ": ";
                binding.second.descriptor_set->to_string(str);
                str << "\n";
            }

            str << "\n    Accessible memory:\n";
            uint32_t set_idx = 0;
            for (auto &set_layout : pipeline_layout->second.setLayouts)
            {
                auto layout = mSyncDevice.descriptor_set_layouts.find(set_layout);
                if (layout == mSyncDevice.descriptor_set_layouts.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with pipeline layout with unknown descriptor set layout");
                }

                auto current_binding = graphics_bindings.find(set_idx);
                if (current_binding == graphics_bindings.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with no descriptor set bound on set %u", set_idx);
                }

                uint32_t binding_idx = 0;
                for (auto &binding : layout->second.bindings)
                {
                    auto current_descriptor = current_binding->second.descriptor_set->bindings.find(binding_idx);
                    if (current_descriptor == current_binding->second.descriptor_set->bindings.end())
                    {
                        return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                            "Draw command called with no descriptor bound on set %u, binding %u", set_idx, binding_idx);
                    }

                    // TODO: should check this is compatible, valid, etc

                    str << "      Set " << set_idx << ", binding " << binding_idx << ":\n";
                    for (uint32_t array_idx = 0; array_idx < binding.descriptorCount; ++array_idx)
                    {
                        str << "        [" << array_idx << "]";
                        switch (binding.descriptorType)
                        {
                        case VK_DESCRIPTOR_TYPE_SAMPLER:
                        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                        {
                            auto &image_info = current_descriptor->second.descriptors.at(array_idx).imageInfo;
                            auto image_view = mSyncDevice.image_views.find(image_info.imageView);
                            if (image_view == mSyncDevice.image_views.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with unknown image view on set %u, binding %u", set_idx, binding_idx);
                            }
                            auto image = mSyncDevice.images.find(image_view->second.image);
                            if (image == mSyncDevice.images.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with image view with unknown image on set %u, binding %u", set_idx, binding_idx);
                            }
                            auto memory = mSyncDevice.device_memories.find(image->second.memory);
                            if (memory == mSyncDevice.device_memories.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with image with unknown memory on set %u, binding %u", set_idx, binding_idx);
                            }
                            str << " memoryRequirements={";
                            str << " size=" << image->second.memoryRequirements.size;
                            str << " alignment=" << image->second.memoryRequirements.alignment;
                            str << " memoryTypeBits=0x" << std::hex << image->second.memoryRequirements.memoryTypeBits << std::dec;
                            str << " }";

                            str << " memory=" << (void *)image->second.memory;
                            str << " {";
                            str << " uid=" << memory->second.uid;
                            str << " allocationSize=" << memory->second.allocationSize;
                            str << " memoryTypeIndex=" << memory->second.memoryTypeIndex;
                            if (memory->second.isMapped)
                            {
                                str << " mapOffset=" << memory->second.mapOffset;
                                str << " mapSize=" << memory->second.mapSize;
                                str << " mapFlags=" << memory->second.mapFlags;
                                str << " pMapData=" << memory->second.pMapData;
                            }
                            else
                            {
                                str << " unmapped";
                            }
                            str << " }";

                            str << " memoryOffset=" << image->second.memoryOffset;
                            str << " subresource={";
                            str << " aspectMask=" << std::hex << image_view->second.subresourceRange.aspectMask << std::dec;
                            str << " baseMipLevel=" << image_view->second.subresourceRange.baseMipLevel;
                            str << " levelCount=" << image_view->second.subresourceRange.levelCount;
                            str << " baseArrayLayer=" << image_view->second.subresourceRange.baseArrayLayer;
                            str << " layerCount=" << image_view->second.subresourceRange.layerCount;
                            str << " }";
                            break;
                        }
                        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                        {
                            auto buffer_view = mSyncDevice.buffer_views.find(current_descriptor->second.descriptors.at(array_idx).bufferView);
                            if (buffer_view == mSyncDevice.buffer_views.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with unknown buffer view on set %u, binding %u", set_idx, binding_idx);
                            }
                            auto buffer = mSyncDevice.buffers.find(buffer_view->second.buffer);
                            if (buffer == mSyncDevice.buffers.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with buffer view with unknown buffer on set %u, binding %u", set_idx, binding_idx);
                            }
                            auto memory = mSyncDevice.device_memories.find(buffer->second.memory);
                            if (memory == mSyncDevice.device_memories.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with buffer with unknown memory on set %u, binding %u", set_idx, binding_idx);
                            }
                            str << " memoryRequirements={";
                            str << " size=" << buffer->second.memoryRequirements.size;
                            str << " alignment=" << buffer->second.memoryRequirements.alignment;
                            str << " memoryTypeBits=0x" << std::hex << buffer->second.memoryRequirements.memoryTypeBits << std::dec;
                            str << " }";

                            str << " memory=" << (void *)buffer->second.memory;
                            str << " {";
                            str << " uid=" << memory->second.uid;
                            str << " allocationSize=" << memory->second.allocationSize;
                            str << " memoryTypeIndex=" << memory->second.memoryTypeIndex;
                            if (memory->second.isMapped)
                            {
                                str << " mapOffset=" << memory->second.mapOffset;
                                str << " mapSize=" << memory->second.mapSize;
                                str << " mapFlags=" << memory->second.mapFlags;
                                str << " pMapData=" << memory->second.pMapData;
                            }
                            else
                            {
                                str << " unmapped";
                            }
                            str << " }";

                            str << " memoryOffset=" << buffer->second.memoryOffset;
                            str << " size=" << buffer->second.size;
                            str << " offset=" << buffer_view->second.offset;
                            str << " range=" << buffer_view->second.range;
                            break;
                        }
                        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        {
                            auto &buffer_info = current_descriptor->second.descriptors.at(array_idx).bufferInfo;
                            auto buffer = mSyncDevice.buffers.find(buffer_info.buffer);
                            if (buffer == mSyncDevice.buffers.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with unknown buffer on set %u, binding %u", set_idx, binding_idx);
                            }
                            auto memory = mSyncDevice.device_memories.find(buffer->second.memory);
                            if (memory == mSyncDevice.device_memories.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with buffer with unknown memory on set %u, binding %u", set_idx, binding_idx);
                            }
                            str << " memoryRequirements={";
                            str << " size=" << buffer->second.memoryRequirements.size;
                            str << " alignment=" << buffer->second.memoryRequirements.alignment;
                            str << " memoryTypeBits=0x" << std::hex << buffer->second.memoryRequirements.memoryTypeBits << std::dec;
                            str << " }";

                            str << " memory=" << (void *)buffer->second.memory;
                            str << " {";
                            str << " uid=" << memory->second.uid;
                            str << " allocationSize=" << memory->second.allocationSize;
                            str << " memoryTypeIndex=" << memory->second.memoryTypeIndex;
                            if (memory->second.isMapped)
                            {
                                str << " mapOffset=" << memory->second.mapOffset;
                                str << " mapSize=" << memory->second.mapSize;
                                str << " mapFlags=" << memory->second.mapFlags;
                                str << " pMapData=" << memory->second.pMapData;
                            }
                            else
                            {
                                str << " unmapped";
                            }
                            str << " }";

                            str << " memoryOffset=" << buffer->second.memoryOffset;
                            str << " size=" << buffer->second.size;
                            break;
                        }
                        default:
                            str << " (INVALID TYPE)";
                            break;
                        }
                        str << "\n";
                    }
                    ++binding_idx;
                }

                ++set_idx;
            }

            if (LOG_INFO(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                "%s", str.str().c_str()))
                return true;
        }
    }

    return false;
}
