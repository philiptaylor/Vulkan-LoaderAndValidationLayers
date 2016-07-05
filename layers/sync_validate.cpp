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
    log_msg(mReportData, level, \
        VK_DEBUG_REPORT_OBJECT_TYPE_##objType##_EXT, (uint64_t)(object), \
        __LINE__, (messageCode), "SYNC", (fmt), __VA_ARGS__)

#define LOG_INFO(objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_INFORMATION_BIT_EXT,         objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_WARN(objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_WARNING_BIT_EXT,             objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_PERF(objType, object, messageCode, fmt, ...)  _LOG_GENERIC(VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_ERROR(objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_ERROR_BIT_EXT,               objType, object, messageCode, fmt, __VA_ARGS__)
#define LOG_DEBUG(objType, object, messageCode, fmt, ...) _LOG_GENERIC(VK_DEBUG_REPORT_DEBUG_BIT_EXT,               objType, object, messageCode, fmt, __VA_ARGS__)

/*
 * For each command submitted:
 *   Assign command ID = (queue ID, subpass ID, unique ID++)
 *   Construct nodes (cmdId, stage), (cmdId, SRC, stage), etc - store them in a hash set
 *     Actually we don't explicitly need them? they're implicit in the edges
 *   Construct memory nodes (READ|WRITE|FLUSH|INVALIDATE, cmdId, stage, access, mem)
 *
 * For pipeline barriers:
 *   Store a collection of edges:
 *     (upper-bound cmd ID, srcStage) < (barrier cmd ID, SRC, srcStage)
 *     (barrier cmd ID, DST, dstStage) < (lower-bound cmd ID, srcStage)
 *
 *  The bounded cmd IDs include queue ID (exact match), subpass ID (exact match, sorta), "<" or ">" some unique ID
 *  Also need "~" edges - maybe each edge has a cost, '<' is 1, '~' is 0
 *   - or maybe store '<=', and for '~' store both ways round
 *
 *  We need to answer queries like:
 *
 *    For every pair of overlapping memory accesses:
 *      Is W < R?
 *      Is R < W?
 *      Is W < F < I < R?
 *   If these fail we've got a race condition
 *   Print details about the write, and the read
 *
 *   A<B is just a graph search: maintain open, closed sets,
 *   be careful that some edges are implicitly generated by bounded IDs
 *
 */

#define CMP(a, b) do { if (a < b) return true; if (b < a) return false; } while (0)

struct CommandId
{
    static const uint64_t SUBPASS_NONE = ~(uint64_t)0;

    uint64_t queueId;
    uint64_t subpassId;
    uint64_t sequenceId;

    CommandId() : queueId(0), subpassId(0), sequenceId(0) { }

    bool operator<(const CommandId &c) const
    {
        CMP(queueId, c.queueId);
        CMP(subpassId, c.subpassId);
        CMP(sequenceId, c.sequenceId);
        return false;
    }
};

struct MemRegion
{
    enum EType
    {
        INVALID,
        GLOBAL,
        BUFFER,
        IMAGE,
    };

    EType type;

    // If BUFFER:
    VkBuffer buffer;
    VkDeviceSize bufferOffset;
    VkDeviceSize bufferRange;

    // If IMAGE:
    VkImage image;
    VkImageSubresourceRange imageSubresourceRange;

    MemRegion() : type(INVALID),
        buffer(VK_NULL_HANDLE), bufferOffset(0), bufferRange(0),
        image(VK_NULL_HANDLE), imageSubresourceRange()
    {
    }

    bool operator<(const MemRegion &m) const
    {
        CMP(type, m.type);
        CMP(buffer, m.buffer);
        CMP(bufferOffset, m.bufferOffset);
        CMP(bufferRange, m.bufferRange);
        CMP(image, m.image);
        CMP(imageSubresourceRange.aspectMask, m.imageSubresourceRange.aspectMask);
        CMP(imageSubresourceRange.baseMipLevel, m.imageSubresourceRange.baseMipLevel);
        CMP(imageSubresourceRange.levelCount, m.imageSubresourceRange.levelCount);
        CMP(imageSubresourceRange.baseArrayLayer, m.imageSubresourceRange.baseArrayLayer);
        CMP(imageSubresourceRange.layerCount, m.imageSubresourceRange.layerCount);
        return false;
    }

    void to_string(std::ostream &str) const
    {
        str << "{";

        switch (type)
        {
        case INVALID:
            str << " INVALID";
            break;
        case GLOBAL:
            str << " GLOBAL";
            break;
        case BUFFER:
            str << " BUFFER";
            str << " " << (void *)buffer;
            str << " offset=" << bufferOffset;
            str << " range=" << bufferRange;
            break;
        case IMAGE:
            str << " IMAGE";
            str << " " << (void *)image;
            str << " aspectMask=0x" << std::hex << imageSubresourceRange.aspectMask << std::dec;
            str << " baseMipLevel=" << imageSubresourceRange.baseMipLevel;
            str << " levelCount=" << imageSubresourceRange.levelCount;
            str << " baseArrayLayer=" << imageSubresourceRange.baseArrayLayer;
            str << " layerCount=" << imageSubresourceRange.layerCount;
            break;
        }

        str << " }";
    }
};

struct SyncNode
{
    enum ENodeType
    {
        INVALID,
        ACTION_CMD_STAGE,
        SYNC_CMD_SRC_STAGE,
        SYNC_CMD_DST_STAGE,
        SYNC_CMD_SRC,
        SYNC_CMD_DST,
        SYNC_CMD_POST_TRANS,
        SYNC_CMD_PRE_TRANS,
        TRANSITION,
        MEM_READ,
        MEM_WRITE,
        MEM_FLUSH,
        MEM_INVALIDATE,
    };

    ENodeType type;

    CommandId commandId;
    VkPipelineStageFlagBits stage;

    VkAccessFlagBits access;
    MemRegion memory;

    SyncNode() : type(INVALID), stage((VkPipelineStageFlagBits)0), access((VkAccessFlagBits)0)
    {
    }

    bool operator<(const SyncNode &n) const
    {
        CMP(type, n.type);
        CMP(commandId, n.commandId);
        CMP(stage, n.stage);
        CMP(access, n.access);
        CMP(memory, n.memory);
        return false;
    }

    void to_string(std::ostream &str) const
    {
        str << "{";

        str << " ";
        switch (type)
        {
        case INVALID: str << "INVALID"; break;
        case ACTION_CMD_STAGE: str << "ACTION_CMD_STAGE"; break;
        case SYNC_CMD_SRC_STAGE: str << "SYNC_CMD_SRC_STAGE"; break;
        case SYNC_CMD_DST_STAGE: str << "SYNC_CMD_DST_STAGE"; break;
        case SYNC_CMD_SRC: str << "SYNC_CMD_SRC"; break;
        case SYNC_CMD_DST: str << "SYNC_CMD_DST"; break;
        case SYNC_CMD_POST_TRANS: str << "SYNC_CMD_POST_TRANS"; break;
        case SYNC_CMD_PRE_TRANS: str << "SYNC_CMD_PRE_TRANS"; break;
        case TRANSITION: str << "TRANSITION"; break;
        case MEM_READ: str << "MEM_READ"; break;
        case MEM_WRITE: str << "MEM_WRITE"; break;
        case MEM_FLUSH: str << "MEM_FLUSH"; break;
        case MEM_INVALIDATE: str << "MEM_INVALIDATE"; break;
        }

        str << " {";
        str << " queueId=" << commandId.queueId;
        if (commandId.subpassId == CommandId::SUBPASS_NONE)
            str << " subpassId=NONE";
        else
            str << " subpassId=" << commandId.queueId;
        str << " sequenceId=" << commandId.sequenceId;
        str << " }";

        switch (type)
        {
        case ACTION_CMD_STAGE:
        case SYNC_CMD_SRC_STAGE:
        case SYNC_CMD_DST_STAGE:
        case MEM_READ:
        case MEM_WRITE:
        case MEM_FLUSH:
        case MEM_INVALIDATE:
            switch (stage)
            {
#define X(n) case VK_PIPELINE_STAGE_##n##_BIT: str << " " #n; break;
                X(TOP_OF_PIPE);
                X(DRAW_INDIRECT);
                X(VERTEX_INPUT);
                X(VERTEX_SHADER);
                X(TESSELLATION_CONTROL_SHADER);
                X(TESSELLATION_EVALUATION_SHADER);
                X(GEOMETRY_SHADER);
                X(FRAGMENT_SHADER);
                X(EARLY_FRAGMENT_TESTS);
                X(LATE_FRAGMENT_TESTS);
                X(COLOR_ATTACHMENT_OUTPUT);
                X(COMPUTE_SHADER);
                X(TRANSFER);
                X(BOTTOM_OF_PIPE);
                X(HOST);
#undef X
            default: str << " " << std::hex << stage << std::dec; break;
            }
            break;
        default:
            break;
        }

        switch (type)
        {
        case MEM_READ:
        case MEM_WRITE:
        case MEM_FLUSH:
        case MEM_INVALIDATE:
            switch (access)
            {
#define X(n) case VK_ACCESS_##n##_BIT: str << " " #n; break;
                X(INDIRECT_COMMAND_READ);
                X(INDEX_READ);
                X(VERTEX_ATTRIBUTE_READ);
                X(UNIFORM_READ);
                X(INPUT_ATTACHMENT_READ);
                X(SHADER_READ);
                X(SHADER_WRITE);
                X(COLOR_ATTACHMENT_READ);
                X(COLOR_ATTACHMENT_WRITE);
                X(DEPTH_STENCIL_ATTACHMENT_READ);
                X(DEPTH_STENCIL_ATTACHMENT_WRITE);
                X(TRANSFER_READ);
                X(TRANSFER_WRITE);
                X(HOST_READ);
                X(HOST_WRITE);
                X(MEMORY_READ);
                X(MEMORY_WRITE);
#undef X
            default: str << " " << std::hex << stage << std::dec; break;
            }

            str << " ";
            memory.to_string(str);
            break;

        default:
            break;
        }

        str << " }";
    }

};

struct SyncEdge
{
    SyncNode a;
    SyncNode b;

    SyncEdge(SyncNode a, SyncNode b) : a(a), b(b) { }

    bool operator<(const SyncEdge &e) const
    {
        CMP(a, e.a);
        CMP(b, e.b);
        return false;
    }
};

#undef CMP


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

    CommandId nextCommandId;
    nextCommandId.queueId = 0;
    nextCommandId.subpassId = CommandId::SUBPASS_NONE;
    nextCommandId.sequenceId = 0;
    uint64_t nextSubpassId = 0;

    std::set<SyncNode> memNodes;
    std::set<SyncEdge> edges;

    for (auto &cmd : buf.commands)
    {
        // TODO: reset state on vkCmdExecuteCommands

        if (cmd->as_begin_render_pass() || cmd->as_next_subpass())
        {
            nextCommandId.subpassId = nextSubpassId++;
        }

        if (cmd->as_end_render_pass())
        {
            nextCommandId.subpassId = CommandId::SUBPASS_NONE;
        }

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

        // XXX: handle vkCmdBindIndexBuffer

        if (cmd->is_draw())
        {
            CommandId commandId = nextCommandId;
            nextCommandId.sequenceId++;

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

            {
                SyncNode n1, n2, n3;
                n1.type = n2.type = n3.type = SyncNode::ACTION_CMD_STAGE;
                n1.commandId = n2.commandId = n3.commandId = commandId;
                n1.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                n3.stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                for (VkPipelineStageFlagBits stage : {
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                    VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
                    VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
                    VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                })
                {
                    n2.stage = stage;
                    edges.insert(SyncEdge(n1, n2));
                    edges.insert(SyncEdge(n2, n3));
                }
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

                    std::vector<VkPipelineStageFlagBits> pipelineStages;
                    if (binding.stageFlags & VK_SHADER_STAGE_VERTEX_BIT)
                        pipelineStages.push_back(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
                    if (binding.stageFlags & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
                        pipelineStages.push_back(VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT);
                    if (binding.stageFlags & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
                        pipelineStages.push_back(VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT);
                    if (binding.stageFlags & VK_SHADER_STAGE_GEOMETRY_BIT)
                        pipelineStages.push_back(VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT);
                    if (binding.stageFlags & VK_SHADER_STAGE_FRAGMENT_BIT)
                        pipelineStages.push_back(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                    // TODO: should limit this to stages where the SPIR-V actually uses these descriptors

                    str << "      Set " << setIdx << ", binding " << bindingIdx << ", stageFlags 0x" << std::hex << binding.stageFlags << std::dec << ":\n";
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

                            SyncNode node;
                            node.commandId = commandId;
                            node.memory.type = MemRegion::IMAGE;
                            node.memory.image = imageView->second.image;
                            node.memory.imageSubresourceRange = imageView->second.subresourceRange;

                            SyncNode stageNode;
                            stageNode.commandId = commandId;
                            stageNode.type = SyncNode::ACTION_CMD_STAGE;

                            for (auto stage : pipelineStages)
                            {
                                node.stage = stageNode.stage = stage;

                                // XXX: handle the access types

                                node.type = SyncNode::MEM_READ;
                                node.access = VK_ACCESS_SHADER_READ_BIT;
                                memNodes.insert(node);
                                edges.insert(SyncEdge(stageNode, node));
                                edges.insert(SyncEdge(node, stageNode));

                                if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                {
                                    node.type = SyncNode::MEM_WRITE;
                                    node.access = VK_ACCESS_SHADER_WRITE_BIT;
                                    memNodes.insert(node);
                                    edges.insert(SyncEdge(stageNode, node));
                                    edges.insert(SyncEdge(node, stageNode));
                                }
                            }

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

                            SyncNode node;
                            node.commandId = commandId;
                            node.memory.type = MemRegion::BUFFER;
                            node.memory.buffer = bufferView->second.buffer;
                            node.memory.bufferOffset = bufferView->second.offset;
                            node.memory.bufferRange = bufferView->second.range;

                            for (auto stage : pipelineStages)
                            {
                                node.stage = stage;

                                // XXX: handle the access types properly
                                // (TODO: is UNIFORM_TEXEL_BUFFER using ACCESS_UNIFORM_READ?)

                                node.type = SyncNode::MEM_READ;
                                node.access = VK_ACCESS_SHADER_READ_BIT;
                                memNodes.insert(node);

                                if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
                                {
                                    node.type = SyncNode::MEM_WRITE;
                                    node.access = VK_ACCESS_SHADER_WRITE_BIT;
                                    memNodes.insert(node);
                                }
                            }

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

                            SyncNode node;
                            node.commandId = commandId;
                            node.memory.type = MemRegion::BUFFER;
                            node.memory.buffer = bufferInfo.buffer;
                            node.memory.bufferOffset = bufferInfo.offset;
                            node.memory.bufferRange = bufferInfo.range;

                            // XXX: handle pDynamicOffsets

                            for (auto stage : pipelineStages)
                            {
                                node.stage = stage;

                                // XXX: handle the access types properly

                                node.type = SyncNode::MEM_READ;
                                node.access = VK_ACCESS_SHADER_READ_BIT;
                                memNodes.insert(node);

                                if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                                {
                                    node.type = SyncNode::MEM_WRITE;
                                    node.access = VK_ACCESS_SHADER_WRITE_BIT;
                                    memNodes.insert(node);
                                }
                            }

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

    for (const SyncNode &node : memNodes)
    {
        std::stringstream str;
        node.to_string(str);
        if (LOG_INFO(QUEUE, queue, SYNC_MSG_NONE, "Mem node: %s", str.str().c_str()))
            return true;
    }

    for (const SyncEdge &edge : edges)
    {
        std::stringstream str;
        str << "    src: ";
        edge.a.to_string(str);
        str << "\n    dst: ";
        edge.b.to_string(str);
        if (LOG_INFO(QUEUE, queue, SYNC_MSG_NONE, "Edge:\n%s", str.str().c_str()))
            return true;
    }

    return false;
}
