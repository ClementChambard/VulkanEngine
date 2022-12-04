#ifndef PIPELINE_H_
#define PIPELINE_H_

#include "Device.hpp"

#include <string>
#include <vector>

struct PipelineConfigInfo {
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineViewportStateCreateInfo viewportInfo;
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    VkPipelineRasterizationStateCreateInfo rasterizationInfo;
    VkPipelineMultisampleStateCreateInfo multisampleInfo;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineColorBlendStateCreateInfo colorBlendInfo;
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
    VkPipelineLayout pipelineLayout = nullptr;
    VkRenderPass renderPass = nullptr;
    uint32_t subpass = 0;
};

class Pipeline {
public:
    Pipeline(Device& device, std::string const& vertFilePath, std::string const& fragFilePath, PipelineConfigInfo& configInfo);
    ~Pipeline();

    Pipeline(Pipeline const&) = delete;
    void operator=(Pipeline const&) = delete;

    static PipelineConfigInfo defaultPipelineConfigInfo(uint32_t width, uint32_t height);

private:
    static std::vector<char> readFile(std::string const& filepath);

    void createGraphicsPipeline(std::string const& vertFilePath, std::string const& fragFilePath, PipelineConfigInfo& configInfo);

    void createShaderModule(std::vector<char> const& code, VkShaderModule* shaderModule);

    Device& m_device;
    VkPipeline m_graphicsPipeline;
    VkShaderModule m_vertShaderModule;
    VkShaderModule m_fragShaderModule;
};

#endif // PIPELINE_H_
