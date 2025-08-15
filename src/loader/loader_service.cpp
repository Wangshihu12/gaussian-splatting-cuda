#include "loader/loader_service.hpp"
#include "loader/loaders/blender_loader.hpp"
#include "loader/loaders/colmap_loader.hpp"
#include "loader/loaders/ply_loader.hpp"
#include <format>
#include <print>

namespace gs::loader {

    LoaderService::LoaderService()
        : registry_(std::make_unique<DataLoaderRegistry>()) {

        // Register default loaders
        registry_->registerLoader(std::make_unique<PLYLoader>());
        registry_->registerLoader(std::make_unique<ColmapLoader>());
        registry_->registerLoader(std::make_unique<BlenderLoader>());

        std::println("LoaderService 初始化完成，共有 {} 个加载器",
                     registry_->size());
    }

    std::expected<LoadResult, std::string> LoaderService::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        // Find appropriate loader
        auto* loader = registry_->findLoader(path);
        if (!loader) {
            // Build detailed error message
            std::string error_msg = std::format(
                "没有找到加载器: {}\n", path.string());

            // Try all loaders to get diagnostic info
            error_msg += "Tried loaders:\n";
            for (const auto& info : registry_->getLoaderInfo()) {
                error_msg += std::format("  - {}: ", info.name);

                // Get specific loader to check
                auto loaders = registry_->findAllLoaders(path);
                bool can_load = false;
                for (auto* l : loaders) {
                    if (l->name() == info.name) {
                        can_load = true;
                        break;
                    }
                }

                if (!can_load) {
                    if (info.extensions.empty()) {
                        error_msg += "directory-based format not detected\n";
                    } else {
                        error_msg += "extension not supported\n";
                    }
                }
            }

            return std::unexpected(error_msg);
        }

        std::println("使用 {} 加载器加载: {}", loader->name(), path.string());

        // Perform the load
        try {
            return loader->load(path, options);
        } catch (const std::exception& e) {
            return std::unexpected(std::format(
                "{} 加载器加载失败: {}", loader->name(), e.what()));
        }
    }

    std::vector<std::string> LoaderService::getAvailableLoaders() const {
        std::vector<std::string> names;
        for (const auto& info : registry_->getLoaderInfo()) {
            names.push_back(info.name);
        }
        return names;
    }

    std::vector<std::string> LoaderService::getSupportedExtensions() const {
        return registry_->getAllSupportedExtensions();
    }

} // namespace gs::loader