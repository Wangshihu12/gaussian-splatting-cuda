#include "gui/panels/scene_panel.hpp"
#include "gui/windows/image_preview.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <imgui.h>
#include <print>
#include <ranges>

namespace gs::gui {

    // ScenePanel Implementation
    ScenePanel::ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager) : m_trainer_manager(trainer_manager) {
        // Create image preview window
        m_imagePreview = std::make_unique<ImagePreview>();
        setupEventHandlers();
    }

    ScenePanel::~ScenePanel() {
        // Cleanup handled automatically
    }

    void ScenePanel::setupEventHandlers() {
        // Subscribe to events using the new event system
        events::state::SceneLoaded::when([this](const auto& event) {
            handleSceneLoaded(event);
        });

        events::state::SceneCleared::when([this](const auto&) {
            handleSceneCleared();
        });
    }

    void ScenePanel::handleSceneLoaded(const events::state::SceneLoaded& event) {
        // Load the image list from the dataset
        if (!event.path.empty()) {
            loadImageCams(event.path);
        }
    }

    void ScenePanel::handleSceneCleared() {
        // Clear the image list
        m_imagePaths.clear();
        m_selectedImageIndex = -1;
    }

    void ScenePanel::render(bool* p_open) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        if (!ImGui::Begin("Scene", p_open)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        // Make buttons smaller to fit the narrow panel
        float button_width = ImGui::GetContentRegionAvail().x;

        if (ImGui::Button("打开文件浏览器", ImVec2(button_width, 0))) {
            // Request to show file browser
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "打开文件浏览器...",
                .source = "ScenePanel"}
                .emit();

            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
        }

        if (ImGui::Button("刷新", ImVec2(button_width * 0.48f, 0))) {
            if (!m_currentDatasetPath.empty()) {
                loadImageCams(m_currentDatasetPath);
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("清空", ImVec2(button_width * 0.48f, 0))) {
            // Clear the image list
            m_imagePaths.clear();
            m_selectedImageIndex = -1;
            m_currentDatasetPath.clear();

            // Also clear the actual scene data
            events::cmd::ClearScene{}.emit();

            // Log the action
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "场景清空",
                .source = "ScenePanel"}
                .emit();
        }

        ImGui::Separator();

        // Image list view
        ImGui::BeginChild("ImageList", ImVec2(0, 0), true);

        if (!m_imagePaths.empty()) {
            ImGui::Text("图片 (%zu):", m_imagePaths.size());
            ImGui::Separator();

            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                const auto& imagePath = m_imagePaths[i];
                std::string filename = imagePath.filename().string();

                // Create unique ID for ImGui by combining filename with index
                std::string unique_id = std::format("{}##{}", filename, i);

                // Check if this item is selected
                bool is_selected = (m_selectedImageIndex == static_cast<int>(i));

                if (ImGui::Selectable(unique_id.c_str(), is_selected)) {
                    m_selectedImageIndex = static_cast<int>(i);
                    onImageSelected(imagePath);
                }

                // Handle double-click to open image preview
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    onImageDoubleClicked(i);
                }

                // Context menu for right-click - use unique ID
                std::string context_menu_id = std::format("context_menu_{}", i);
                if (ImGui::BeginPopupContextItem(context_menu_id.c_str())) {
                    if (ImGui::MenuItem("转到相机视图")) {
                        // Get the camera data for this image
                        auto cam_data_it = m_PathToCamId.find(imagePath);
                        if (cam_data_it != m_PathToCamId.end()) {
                            // Emit the new GoToCamView command event with camera data
                            events::cmd::GoToCamView{
                                .cam_id = cam_data_it->second}
                                .emit();

                            // Log the action
                            events::notify::Log{
                                .level = events::notify::Log::Level::Info,
                                .message = std::format("转到相机视图: {} (相机ID: {})",
                                                       imagePath.filename().string(),
                                                       cam_data_it->second),
                                .source = "ScenePanel"}
                                .emit();
                        } else {
                            // Log warning if camera data not found
                            events::notify::Log{
                                .level = events::notify::Log::Level::Warning,
                                .message = std::format("相机数据未找到: {}", imagePath.filename().string()),
                                .source = "ScenePanel"}
                                .emit();
                        }
                    }
                    ImGui::EndPopup();
                }

                // Tooltip with full path
                // if (ImGui::IsItemHovered()) {
                //     ImGui::BeginTooltip();
                //     ImGui::Text("Path: %s", imagePath.string().c_str());
                //     ImGui::EndTooltip();
                // }
            }
        } else {
            ImGui::Text("没有加载图片.");
            ImGui::Text("使用 '打开文件浏览器' 加载数据集.");
        }

        ImGui::EndChild();

        ImGui::End();
        ImGui::PopStyleColor();

        // Render image preview window if open
        if (m_showImagePreview && m_imagePreview) {
            m_imagePreview->render(&m_showImagePreview);
        }
    }

    void ScenePanel::loadImageCams(const std::filesystem::path& path) {

        m_currentDatasetPath = path;
        m_imagePaths.clear();
        m_PathToCamId.clear();
        m_selectedImageIndex = -1;

        if (!m_trainer_manager) {
            std::cerr << "m_trainer_manager 未设置" << std::endl;
            return;
        }

        auto cams = m_trainer_manager->getCamList();

        for (const auto& cam : cams) {
            m_imagePaths.emplace_back(cam->image_path());
            m_PathToCamId[cam->image_path()] = cam->uid();
        }

        // Sort paths for consistent ordering
        std::ranges::sort(m_imagePaths, [](const auto& a, const auto& b) {
            return a.filename() < b.filename();
        });

        // Log the action
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("加载 {} 张图片从数据集: {}", m_imagePaths.size(), path.string()),
            .source = "ScenePanel"}
            .emit();
    }

    void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
        m_onDatasetLoad = std::move(callback);
    }

    void ScenePanel::onImageSelected(const std::filesystem::path& imagePath) {
        // Log selection
        events::notify::Log{
            .level = events::notify::Log::Level::Debug,
            .message = std::format("选中图片: {}", imagePath.filename().string()),
            .source = "ScenePanel"}
            .emit();

        // Publish NodeSelectedEvent for other components to react
        events::ui::NodeSelected{
            .path = imagePath.string(),
            .type = "Images",
            .metadata = {{"filename", imagePath.filename().string()}, {"path", imagePath.string()}}}
            .emit();
    }

    void ScenePanel::onImageDoubleClicked(size_t imageIndex) {
        if (imageIndex >= m_imagePaths.size()) {
            return;
        }

        const auto& imagePath = m_imagePaths[imageIndex];

        // Open the image preview with all images and current index
        if (m_imagePreview) {
            m_imagePreview->open(m_imagePaths, imageIndex);
            m_showImagePreview = true;
        }

        // Log the action
        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("打开图片预览: {}", imagePath.filename().string()),
            .source = "ScenePanel"}
            .emit();
    }

} // namespace gs::gui