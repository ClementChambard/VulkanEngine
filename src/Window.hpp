#ifndef WINDOW_H_
#define WINDOW_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>

class Window {

public:
    Window(int width, int height, std::string title);
    ~Window();

    void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);
    bool shouldClose() { return glfwWindowShouldClose(window); }
    void getFramebufferSize(int& width, int& height) { glfwGetFramebufferSize(window, &width, &height); this->width = width, this->height = height; }

    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }

    bool hasBeenResized() { return resized; }
    void resizeHandled() { resized = false; }

    void waitMinimized();

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    int width, height;

    GLFWwindow* window;
    bool resized = false;
};

#endif // WINDOW_H_
