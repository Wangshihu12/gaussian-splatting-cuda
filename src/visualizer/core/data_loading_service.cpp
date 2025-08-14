#include "core/data_loading_service.hpp"
#include "core/viewer_state_manager.hpp"
#include "scene/scene_manager.hpp"
#include <print>

namespace gs::visualizer {

    DataLoadingService::DataLoadingService(SceneManager* scene_manager,
                                           ViewerStateManager* state_manager)
        : scene_manager_(scene_manager),
          state_manager_(state_manager) {
        setupEventHandlers();
    }

    DataLoadingService::~DataLoadingService() = default;

    void DataLoadingService::setupEventHandlers() {
        using namespace events;

        // Listen for file load commands
        cmd::LoadFile::when([this](const auto& cmd) {
            handleLoadFileCommand(cmd);
        });
    }

    void DataLoadingService::handleLoadFileCommand(const events::cmd::LoadFile& cmd) {
        if (cmd.is_dataset) {
            loadDataset(cmd.path);
        } else {
            loadPLY(cmd.path);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadPLY(const std::filesystem::path& path) {
        try {
            std::println("加载PLY文件: {}", path.string());

            // Load through scene manager
            scene_manager_->loadPLY(path);

            // Update state
            state_manager_->setPLYPath(path);

            // Emit success event
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("成功加载PLY: {}", path.filename().string()),
                .source = "DataLoadingService"}
                .emit();

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("加载PLY时出错: {}", e.what());

            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", path.string())}
                .emit();

            return std::unexpected(error_msg);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadDataset(const std::filesystem::path& path) {
        try {
            std::println("加载数据集: {}", path.string());

            // Validate parameters
            if (params_.dataset.data_path.empty() && path.empty()) {
                throw std::runtime_error("未指定数据集路径");
            }

            // Load through scene manager
            scene_manager_->loadDataset(path, params_);

            // Update state
            state_manager_->setDatasetPath(path);

            // Emit success event
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("成功加载数据集: {}", path.filename().string()),
                .source = "DataLoadingService"}
                .emit();

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("加载数据集时出错: {}", e.what());

            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", path.string())}
                .emit();

            return std::unexpected(error_msg);
        }
    }

    void DataLoadingService::clearScene() {
        try {
            scene_manager_->clearScene();
            state_manager_->reset();

            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "场景已清除",
                .source = "DataLoadingService"}
                .emit();
        } catch (const std::exception& e) {
            events::notify::Error{
                .message = "清除场景时出错",
                .details = e.what()}
                .emit();
        }
    }

} // namespace gs::visualizer
