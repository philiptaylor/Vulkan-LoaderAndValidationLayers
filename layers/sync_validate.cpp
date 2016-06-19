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

    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;

    struct Binding
    {
        sync_descriptor_set *descriptorSet;
        uint32_t dynamicOffset; // TODO
    };

    std::map<uint32_t, Binding> graphicsBindings;
    std::map<uint32_t, Binding> computeBindings;

    for (auto &cmd : buf.commands)
    {
        // TODO: reset state on vkCmdExecuteCommands

        auto bindPipeline = cmd->as_bind_pipeline();
        if (bindPipeline)
        {
            if (bindPipeline->pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
                graphicsPipeline = bindPipeline->pipeline;
            else if (bindPipeline->pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
                computePipeline = bindPipeline->pipeline;
        }

        auto bindDescriptorSets = cmd->as_bind_descriptor_sets();
        if (bindDescriptorSets)
        {
            // TODO: should look at pipeline layout compatibility here
            // TODO: dynamic offsets

            for (uint32_t i = 0; i < bindDescriptorSets->descriptorSets.size(); ++i)
            {
                uint32_t setNumber = bindDescriptorSets->firstSet + i;

                auto descriptorSet = mSyncDevice.descriptor_sets.find(bindDescriptorSets->descriptorSets[i]);
                if (descriptorSet == mSyncDevice.descriptor_sets.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with unknown descriptor set bound");
                }

                Binding binding;
                binding.descriptorSet = &descriptorSet->second;
                binding.dynamicOffset = 0;

                if (bindDescriptorSets->pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
                    graphicsBindings[setNumber] = binding;
                else if (bindDescriptorSets->pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
                    computeBindings[setNumber] = binding;
            }
        }

        if (cmd->is_draw())
        {
            if (graphicsPipeline == VK_NULL_HANDLE)
            {
                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with no pipeline bound");
            }

            auto pipeline = mSyncDevice.graphics_pipelines.find(graphicsPipeline);
            if (pipeline == mSyncDevice.graphics_pipelines.end())
            {
                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                    "Draw command called with unknown pipeline bound");
            }

            auto pipelineLayout = mSyncDevice.pipeline_layouts.find(pipeline->second.layout);
            if (pipelineLayout == mSyncDevice.pipeline_layouts.end())
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
            pipelineLayout->second.to_string(str);
            for (auto &setLayout : pipelineLayout->second.setLayouts)
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
            for (auto &binding : graphicsBindings)
            {
                str << "      " << binding.first << ": ";
                binding.second.descriptorSet->to_string(str);
                str << "\n";
            }

            str << "\n    Accessible memory:\n";
            uint32_t setIdx = 0;
            for (auto &setLayout : pipelineLayout->second.setLayouts)
            {
                auto layout = mSyncDevice.descriptor_set_layouts.find(setLayout);
                if (layout == mSyncDevice.descriptor_set_layouts.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with pipeline layout with unknown descriptor set layout");
                }

                auto currentBinding = graphicsBindings.find(setIdx);
                if (currentBinding == graphicsBindings.end())
                {
                    return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                        "Draw command called with no descriptor set bound on set %u", setIdx);
                }

                uint32_t bindingIdx = 0;
                for (auto &binding : layout->second.bindings)
                {
                    auto currentDescriptor = currentBinding->second.descriptorSet->bindings.find(bindingIdx);
                    if (currentDescriptor == currentBinding->second.descriptorSet->bindings.end())
                    {
                        return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                            "Draw command called with no descriptor bound on set %u, binding %u", setIdx, bindingIdx);
                    }

                    // TODO: should check this is compatible, valid, etc

                    str << "      Set " << setIdx << ", binding " << bindingIdx << ":\n";
                    for (uint32_t arrayIdx = 0; arrayIdx < binding.descriptorCount; ++arrayIdx)
                    {
                        str << "        [" << arrayIdx << "]";
                        switch (binding.descriptorType)
                        {
                        case VK_DESCRIPTOR_TYPE_SAMPLER:
                        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                        {
                            auto &imageInfo = currentDescriptor->second.descriptors.at(arrayIdx).imageInfo;
                            auto imageView = mSyncDevice.image_views.find(imageInfo.imageView);
                            if (imageView == mSyncDevice.image_views.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with unknown image view on set %u, binding %u", setIdx, bindingIdx);
                            }
                            auto image = mSyncDevice.images.find(imageView->second.image);
                            if (image == mSyncDevice.images.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with image view with unknown image on set %u, binding %u", setIdx, bindingIdx);
                            }
                            auto memory = mSyncDevice.device_memories.find(image->second.memory);
                            if (memory == mSyncDevice.device_memories.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with image with unknown memory on set %u, binding %u", setIdx, bindingIdx);
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
                            str << " aspectMask=" << std::hex << imageView->second.subresourceRange.aspectMask << std::dec;
                            str << " baseMipLevel=" << imageView->second.subresourceRange.baseMipLevel;
                            str << " levelCount=" << imageView->second.subresourceRange.levelCount;
                            str << " baseArrayLayer=" << imageView->second.subresourceRange.baseArrayLayer;
                            str << " layerCount=" << imageView->second.subresourceRange.layerCount;
                            str << " }";
                            break;
                        }
                        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                        {
                            auto bufferView = mSyncDevice.buffer_views.find(currentDescriptor->second.descriptors.at(arrayIdx).bufferView);
                            if (bufferView == mSyncDevice.buffer_views.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with unknown buffer view on set %u, binding %u", setIdx, bindingIdx);
                            }
                            auto buffer = mSyncDevice.buffers.find(bufferView->second.buffer);
                            if (buffer == mSyncDevice.buffers.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with buffer view with unknown buffer on set %u, binding %u", setIdx, bindingIdx);
                            }
                            auto memory = mSyncDevice.device_memories.find(buffer->second.memory);
                            if (memory == mSyncDevice.device_memories.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with buffer with unknown memory on set %u, binding %u", setIdx, bindingIdx);
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
                            str << " offset=" << bufferView->second.offset;
                            str << " range=" << bufferView->second.range;
                            break;
                        }
                        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        {
                            auto &bufferInfo = currentDescriptor->second.descriptors.at(arrayIdx).bufferInfo;
                            auto buffer = mSyncDevice.buffers.find(bufferInfo.buffer);
                            if (buffer == mSyncDevice.buffers.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with unknown buffer on set %u, binding %u", setIdx, bindingIdx);
                            }
                            auto memory = mSyncDevice.device_memories.find(buffer->second.memory);
                            if (memory == mSyncDevice.device_memories.end())
                            {
                                return LOG_ERROR(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                                    "Draw command called with buffer with unknown memory on set %u, binding %u", setIdx, bindingIdx);
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
                    ++bindingIdx;
                }

                ++setIdx;
            }

            if (LOG_INFO(COMMAND_BUFFER, buf.command_buffer, SYNC_MSG_NONE,
                "%s", str.str().c_str()))
                return true;
        }
    }

    return false;
}
