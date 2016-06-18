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

void sync_cmd_bind_pipeline::to_string(std::ostream &str)
{
    str << "vkCmdBindPipeline {";
    str << " pipelineBindPoint=" << pipelineBindPoint;
    str << " pipeline=" << (void *)pipeline;
    str << " }";
}

void sync_cmd_set_viewport::to_string(std::ostream &str)
{
    str << "vkCmdSetViewport {";
    str << " firstViewport=" << firstViewport;

    str << " viewports=[";
    for (auto &v : viewports)
    {
        str << " {";
        str << " x=" << v.x;
        str << " y=" << v.y;
        str << " width=" << v.width;
        str << " height=" << v.height;
        str << " minDepth=" << v.minDepth;
        str << " maxDepth=" << v.maxDepth;
        str << " }";
    }
    str << " ]";

    str << " }";
}

void sync_cmd_set_scissor::to_string(std::ostream &str)
{
    str << "vkCmdSetScissor {";
    str << " firstScissor =" << firstScissor;

    str << " scissors=[";
    for (auto &s : scissors)
    {
        str << " {";
        str << " offset=(" << s.offset.x << ", " << s.offset.y << ")";
        str << " extent=(" << s.extent.width << ", " << s.extent.height << ")";
        str << " }";
    }
    str << " ]";
    str << " }";
}

void sync_cmd_bind_descriptor_sets::to_string(std::ostream &str)
{
    str << "vkCmdBindDescriptorSets {";
    str << " pipelineBindPoint=" << pipelineBindPoint;
    str << " layout=" << (void *)layout;
    str << " firstSet=" << firstSet;

    str << " descriptorSets=[";
    for (auto &set : descriptorSets)
        str << " " << (void *)set;
    str << " ]";

    str << " dynamicOffsets=[";
    for (auto &offset : dynamicOffsets)
        str << " " << offset;
    str << " ]";

    str << " }";
}

void sync_cmd_bind_vertex_buffers::to_string(std::ostream &str)
{
    str << "vkCmdBindVertexBuffers {";
    str << " ...";
    str << " }";
}

void sync_cmd_draw::to_string(std::ostream &str)
{
    str << "vkCmdDraw {";
    str << " vertexCount=" << vertexCount;
    str << " instanceCount=" << instanceCount;
    str << " firstVertex=" << firstVertex;
    str << " firstInstance=" << firstInstance;
    str << " }";
}

void sync_cmd_draw_indexed::to_string(std::ostream &str)
{
    str << "vkCmdDrawIndexed {";
    str << " indexCount=" << indexCount;
    str << " instanceCount=" << instanceCount;
    str << " firstIndex=" << firstIndex;
    str << " vertexOffset=" << vertexOffset;
    str << " firstInstance=" << firstInstance;
    str << " }";
}

void sync_cmd_copy_image::to_string(std::ostream &str)
{
    str << "vkCmdCopyImage {";
    str << " ...";
    str << " }";
}

void sync_cmd_pipeline_barrier::to_string(std::ostream &str)
{
    str << "vkCmdPipelineBarrier {";
    str << " srcStageMask=0x" << std::hex << srcStageMask << std::dec;
    str << " dstStageMask=0x" << std::hex << dstStageMask << std::dec;
    str << " dependencyFlags=0x" << std::hex << dependencyFlags << std::dec;

    str << " memoryBarriers=[";
    for (auto &b : memoryBarriers)
    {
        str << " {";
        str << " srcAccessMask=0x" << std::hex << b.srcAccessMask << std::dec;
        str << " dstAccessMask=0x" << std::hex << b.dstAccessMask << std::dec;
        str << " }";
    }
    str << " ]";

    str << " bufferMemoryBarriers=[";
    for (auto &b : bufferMemoryBarriers)
    {
        str << " {";
        str << " srcAccessMask=0x" << std::hex << b.srcAccessMask << std::dec;
        str << " dstAccessMask=0x" << std::hex << b.dstAccessMask << std::dec;
        str << " srcQueueFamilyIndex=" << b.srcQueueFamilyIndex;
        str << " dstQueueFamilyIndex=" << b.dstQueueFamilyIndex;
        str << " buffer=" << (void *)b.buffer;
        str << " offset=" << b.offset;
        str << " size=" << b.size;
        str << " }";
    }
    str << " ]";

    str << " imageMemoryBarriers=[";
    for (auto &b : imageMemoryBarriers)
    {
        str << " {";
        str << " srcAccessMask=0x" << std::hex << b.srcAccessMask << std::dec;
        str << " dstAccessMask=0x" << std::hex << b.dstAccessMask << std::dec;
        str << " oldLayout=" << b.oldLayout;
        str << " newLayout=" << b.newLayout;
        str << " srcQueueFamilyIndex=" << b.srcQueueFamilyIndex;
        str << " dstQueueFamilyIndex=" << b.dstQueueFamilyIndex;
        str << " image=" << (void *)b.image;
        str << " subresourceRange={";
        str << " aspectMask=0x" << std::hex << b.subresourceRange.aspectMask << std::dec;
        str << " baseMipLevel=" << b.subresourceRange.baseMipLevel;
        str << " levelCount=" << b.subresourceRange.levelCount;
        str << " baseArrayLayer=" << b.subresourceRange.baseArrayLayer;
        str << " layerCount=" << b.subresourceRange.layerCount;
        str << " }";
        str << " }";
    }
    str << " ]";

    str << " }";
}

void sync_cmd_begin_render_pass::to_string(std::ostream &str)
{
    str << "vkCmdBeginRenderPass {";
    str << " renderPass=" << (void *)renderPass;
    str << " framebuffer=" << (void *)framebuffer;
    str << " renderArea={";
    str << " offset=(" << renderArea.offset.x << ", " << renderArea.offset.y << ")";
    str << " extent=(" << renderArea.extent.width << ", " << renderArea.extent.height << ")";
    str << " }";

    str << " clearValues=[";
    for (auto &v : clearValues)
    {
        // Correct interpretation depends on attachment type, which we
        // don't know here, so just print all possibilities
        str << " {";
        str << " (" << v.color.float32[0] << ", " << v.color.float32[1] << ", " << v.color.float32[2] << ", " << v.color.float32[3] << ")";
        str << " |";
        str << " (" << v.color.int32[0] << ", " << v.color.int32[1] << ", " << v.color.int32[2] << ", " << v.color.int32[3] << ")";
        str << " |";
        str << " (" << v.color.uint32[0] << ", " << v.color.uint32[1] << ", " << v.color.uint32[2] << ", " << v.color.uint32[3] << ")";
        str << " |";
        str << " (" << v.depthStencil.depth << ", " << v.depthStencil.stencil << ")";
        str << " }";
    }
    str << " ]";

    str << " contents=" << contents;
    str << " }";
}

void sync_cmd_next_subpass::to_string(std::ostream &str)
{
    str << "vkCmdNextSubpass {";
    str << " contents=" << contents;
    str << " }";
}

void sync_cmd_end_render_pass::to_string(std::ostream &str)
{
    str << "vkCmdEndRenderPass {";
    str << " }";
}


void sync_descriptor_set::to_string(std::ostream &str)
{
    str << "VkDescriptorSet " << (void *)descriptor_set << " {";
    str << " descriptor_pool=" << (void *)descriptor_pool;
    str << " setLayout=" << (void *)setLayout;
    str << " bindings={";
    for (auto &binding : bindings)
    {
        str << " " << binding.first << ": {";
        str << " type=" << binding.second.descriptorType;
        str << " descriptors=[";
        for (auto &descriptor : binding.second.descriptors)
        {
            if (!descriptor.valid)
            {
                str << " invalid";
            }
            else
            {
                switch (binding.second.descriptorType)
                {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    str << " {";
                    str << " sampler=" << (void *)descriptor.imageInfo.sampler;
                    str << " imageView=" << (void *)descriptor.imageInfo.imageView;
                    str << " imageLayout=" << descriptor.imageInfo.imageLayout;
                    str << " }";
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                    str << " {";
                    str << " bufferView=" << (void *)descriptor.bufferView;
                    str << " }";
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    str << " {";
                    str << " buffer=" << (void *)descriptor.bufferInfo.buffer;
                    str << " offset=" << descriptor.bufferInfo.offset;
                    str << " range=" << descriptor.bufferInfo.range;
                    str << " }";
                    break;
                default:
                    str << " (INVALID TYPE)";
                    break;
                }
            }
        }
        str << " ]";
        str << " }";
    }
    str << " }";
    str << " }";
}

void sync_descriptor_set_layout::to_string(std::ostream &str)
{
    str << "VkDescriptorSetLayout " << (void *)descriptor_set_layout << " {";
    str << " flags=0x" << std::hex << flags << std::dec;
    str << " bindings=[";
    for (auto &binding : bindings)
    {
        str << " {";
        str << " binding=" << binding.binding;
        str << " descriptorType=" << binding.descriptorType;
        str << " descriptorCount=" << binding.descriptorCount;
        str << " stageFlags=0x" << std::hex << binding.stageFlags << std::dec;
        str << " immutableSamplers=[";
        for (auto &sampler : binding.immutableSamplers)
        {
            str << " " << (void *)sampler;
        }
        str << " ]";
        str << " }";
    }
    str << " ]";
    str << " }";
}

void sync_pipeline_layout::to_string(std::ostream &str)
{
    str << "VkPipelineLayout " << (void *)pipeline_layout << " {";
    str << " flags=0x" << std::hex << flags << std::dec;
    str << " setLayouts=[";
    for (auto &set : setLayouts)
    {
        str << " " << (void *)set;
    }
    str << " ]";
    str << " pushConstantRanges=[";
    for (auto &range : pushConstantRanges)
    {
        str << " {";
        str << " stageFlags=0x" << std::hex << range.stageFlags << std::dec;
        str << " offset=" << range.offset;
        str << " size=" << range.size;
        str << " }";
    }
    str << " ]";
    str << " }";
}

void sync_graphics_pipeline::to_string(std::ostream &str)
{
    str << "VkPipeline " << (void *)pipeline << " {";
    str << " flags=0x" << std::hex << flags << std::dec;
    str << " stages=[";
    for (auto &stage : stages)
    {
        str << " {";
        str << " flags=0x" << std::hex << stage.flags << std::dec;
        str << " stage=0x" << std::hex << stage.stage << std::dec;
        str << " module=" << (void *)stage.module;
        str << " name=\"" << stage.name << "\"";
        str << " }";
    }
    str << " ]";
    str << " ..."; // TODO
    str << " layout=" << (void *)layout;
    str << " renderPass=" << (void *)renderPass;
    str << " subpass=" << subpass;
    str << "}";
}
