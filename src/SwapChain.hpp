#ifndef SWAPCHAIN_H_
#define SWAPCHAIN_H_

#include "Device.hpp"

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

class SwapChain {

public:
    static constexpr int MAX_FRAME_IN_FLIGHT = 2;

    SwapChain(Device& device, VkExtent2D windowExtent);
    ~SwapChain();

    SwapChain(SwapChain const&) = delete;
    void operator=(SwapChain const&) = delete;

    VkFramebuffer getFramebuffer(int index) const { return m_framebuffers[index]; }
    VkRenderPass getRenderPass() const { return m_renderPass; }
    VkImageView getImageView(int index) const { return m_imageViews[index]; }
    size_t imageCount() const { return m_images.size(); }
    VkFormat getImageFormat() const { return m_imageFormat; }
    VkExtent2D getExtent() const { return m_extent; }
    size_t getCurrentFrame() const { return m_currentFrame; }
    uint32_t width() const { return m_extent.width; }
    uint32_t height() const { return m_extent.height; }
    VkFence& getFence(int index) { return m_inFlightFences[index]; }

    float extentAspectRatio() { return static_cast<float>(m_extent.width) / static_cast<float>(m_extent.height); }
    VkFormat findDepthFormat();

    VkResult acquireNextImage(uint32_t *imageIndex);
    VkResult submitCommandBuffers(const VkCommandBuffer *buffers, uint32_t *imageIndex);

    void recreate();
    void cleanup();

private:
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createColorResources();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(std::vector<VkPresentModeKHR> const& availablePresentModes);
    VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR const& capabilities);

    VkFormat m_imageFormat;
    VkExtent2D m_extent;

    std::vector<VkFramebuffer> m_framebuffers;
    VkRenderPass m_renderPass;

    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
    VkImage m_depthImage;
    VkImageView m_depthImageView;
    VkDeviceMemory m_depthImageMemory;
    VkImage m_colorImage;
    VkImageView m_colorImageView;
    VkDeviceMemory m_colorImageMemory;

    Device& m_device;
    VkExtent2D m_windowExtent;

    VkSwapchainKHR m_swapChain;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    size_t m_currentFrame = 0;
};

#endif // SWAPCHAIN_H_
