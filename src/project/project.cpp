#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "core/logger.hpp"
#include "project/project.hpp"

namespace gs::management {

    // Static member definitions
    const Version Project::CURRENT_VERSION(0, 0, 1);
    const std::string Project::FILE_HEADER = "LichtFeldStudio Project File";
    const std::string Project::EXTENSION = ".ls"; // LichtFeldStudio file

    // Version implementation
    Version::Version(const std::string& versionStr) {
        std::istringstream ss(versionStr);
        std::string token;

        std::getline(ss, token, '.');
        major = std::stoi(token);

        std::getline(ss, token, '.');
        minor = std::stoi(token);

        std::getline(ss, token);
        patch = std::stoi(token);
    }

    std::string Version::toString() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }

    bool Version::operator>=(const Version& other) const {
        if (major != other.major)
            return major > other.major;
        if (minor != other.minor)
            return minor > other.minor;
        return patch >= other.patch;
    }

    bool Version::operator<(const Version& other) const {
        return !(*this >= other);
    }

    bool Version::operator<=(const Version& other) const {
        return *this < other || *this == other;
    }

    bool Version::operator>(const Version& other) const {
        return !(*this <= other);
    }

    bool Version::operator==(const Version& other) const {
        return major == other.major && minor == other.minor && patch == other.patch;
    }

    bool Version::operator!=(const Version& other) const {
        return !(*this == other);
    }

    // MigratorRegistry implementation
    void MigratorRegistry::registerMigrator(std::unique_ptr<ProjectMigrator> migrator) {
        migrators_.push_back(std::move(migrator));
    }

    nlohmann::json MigratorRegistry::migrateToVersion(const nlohmann::json& data, const Version& from, const Version& to) const {
        nlohmann::json current = data;
        Version currentVersion = from;

        while (currentVersion < to) {
            bool migrationFound = false;
            for (const auto& migrator : migrators_) {
                if (migrator->canMigrate(currentVersion, to)) {
                    current = migrator->migrate(current, currentVersion, to);
                    currentVersion = to; // For now, assume direct migration
                    migrationFound = true;
                    break;
                }
            }

            if (!migrationFound) {
                throw std::runtime_error("No migration path found from version " +
                                         currentVersion.toString() + " to " + to.toString());
            }
        }

        return current;
    }

    bool IsColmapData(const std::filesystem::path& path) {
        if (!std::filesystem::is_directory(path)) {
            return false;
        }
        // Check for sparse reconstruction
        std::filesystem::path sparse_path;
        if (std::filesystem::exists(path / "sparse" / "0")) {
            sparse_path = path / "sparse" / "0";
        } else if (std::filesystem::exists(path / "sparse")) {
            sparse_path = path / "sparse";
        } else {
            return false;
        }

        return true;
    }

    DataSetInfo::DataSetInfo(const param::DatasetConfig& data_config) : DatasetConfig(data_config) {
        if (data_path.empty()) {
            data_type = "";
        } else {
            data_type = IsColmapData(data_path) ? "Colmap" : "Blender";
        }
    }

    // LichtFeldProject implementation
    Project::Project(bool update_file_on_change) : update_file_on_change_(update_file_on_change) {
        project_data_.version = CURRENT_VERSION;
        project_data_.project_creation_time = generateCurrentTimeStamp();
        initializeMigrators();

        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    void Project::setProjectFileName(const std::filesystem::path& path) {
        if (std::filesystem::is_directory(path)) {
            std::string project_file_name = project_data_.project_name.empty() ? "project" : project_data_.project_name;
            project_file_name += EXTENSION;
            output_file_name_ = path / project_file_name;
        } else if (std::filesystem::is_regular_file(path)) {
            if (path.extension() != EXTENSION) {
                throw std::runtime_error(std::format("LichtFeldProjectFile: {} expected file extension to be {}", path.string(), EXTENSION));
            }
        }
        output_file_name_ = path;
    }

    Project::Project(const ProjectData& initialData, bool update_file_on_change)
        : project_data_(initialData),
          update_file_on_change_(update_file_on_change) {
        initializeMigrators();
    }

    void Project::initializeMigrators() {
        // Register migration classes for future versions
        // Example: migrator_registry_.registerMigrator(std::make_unique<Version001To002Migrator>());
    }

    bool Project::readFromFile(const std::filesystem::path& filepath) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        try {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                LOG_ERROR("Cannot open file for reading: {}", filepath.string());
                return false;
            }

            nlohmann::json doc;
            file >> doc;

            if (!validateJsonStructure(doc)) {
                LOG_ERROR("Invalid JSON structure in file: {}", filepath.string());
                return false;
            }

            // Check version and migrate if necessary
            Version fileVersion(doc["version"].get<std::string>());

            nlohmann::json processedDoc = doc;
            if (fileVersion < CURRENT_VERSION) {
                LOG_INFO("Migrating from version {} to {}", fileVersion.toString(), CURRENT_VERSION.toString());
                processedDoc = migrator_registry_.migrateToVersion(doc, fileVersion, CURRENT_VERSION);
            }

            project_data_ = parseProjectData(processedDoc);
            output_file_name_ = filepath;

            return true;

        } catch (const std::exception& e) {
            LOG_ERROR("Error reading project file: {}", e.what());
            return false;
        }
    }

    bool Project::writeToFile(const std::filesystem::path& filepath) {
        std::lock_guard<std::mutex> lock(io_mutex_);

        std::filesystem::path targetPath = filepath.empty() ? output_file_name_ : filepath;
        if (targetPath.empty()) {
            LOG_ERROR("LichtFeldProjectFile::writeToFile - no output file was set");
            return false;
        }

        if (std::filesystem::is_directory(targetPath)) {
            LOG_ERROR("LichtFeldProjectFile: {} is directory and not a file", targetPath.string());
            return false;
        }

        if (!std::filesystem::is_directory(targetPath.parent_path())) {
            LOG_ERROR("LichtFeldProjectFile: {} parent directory does not exist {}", targetPath.parent_path().string(), targetPath.string());
            return false;
        }

        if (targetPath.extension() != EXTENSION) {
            LOG_ERROR("LichtFeldProjectFile: {} expected file extension to be {}", targetPath.string(), EXTENSION);
            return false;
        }

        project_data_.project_last_update_time = generateCurrentTimeStamp();

        try {
            std::ofstream file(targetPath);
            if (!file.is_open()) {
                LOG_ERROR("Cannot open file for writing: {}", targetPath.string());
                return false;
            }

            // Serialize and write JSON
            nlohmann::ordered_json doc = serializeProjectData(project_data_);
            file << doc.dump(4) << std::endl; // Pretty print with 4-space indentation

            return true;

        } catch (const std::exception& e) {
            LOG_ERROR("Error writing project file: {}", e.what());
            return false;
        }
    }

    bool Project::validateJsonStructure(const nlohmann::json& json) const {
        // Basic validation - check required fields
        bool contains_basics = json.contains("project_info") &&
                               json.contains("version") &&
                               json.contains("project_name") &&
                               json.contains("project_creation_time") &&
                               json.contains("project_last_update_time") &&
                               json.contains("project_output_folder") &&
                               json.contains("data") &&
                               json.contains("outputs");
        if (!contains_basics) {
            return false;
        }

        const auto& dataJson = json["data"];
        bool contains_data = dataJson.contains("data_path") &&
                             dataJson.contains("images") &&
                             dataJson.contains("resize_factor") &&
                             dataJson.contains("test_every") &&
                             dataJson.contains("data_type");

        return contains_data;
    }

    ProjectData Project::parseProjectData(const nlohmann::json& json) const {
        ProjectData data;

        data.version = Version(json["version"].get<std::string>());
        data.project_name = json["project_name"].get<std::string>();
        data.project_creation_time = json["project_creation_time"].get<std::string>();
        data.project_last_update_time = json["project_last_update_time"].get<std::string>();
        data.data_set_info.output_path = std::filesystem::path(json["project_output_folder"].get<std::string>());

        // Parse data section
        const auto& dataJson = json["data"];
        data.data_set_info.data_path = dataJson["data_path"].get<std::string>();
        data.data_set_info.images = dataJson["images"].get<std::string>();
        data.data_set_info.resize_factor = dataJson["resize_factor"].get<int>();
        data.data_set_info.test_every = dataJson["test_every"].get<int>();
        data.data_set_info.data_type = dataJson["data_type"].get<std::string>();

        if (json.contains("training") && json["training"].contains("optimization")) {
            data.optimization = param::OptimizationParameters::from_json(json["training"]["optimization"]);
        }

        // Parse outputs section
        const auto& outputsJson = json["outputs"];
        if (outputsJson.contains("plys") && outputsJson["plys"].is_array()) {
            for (const auto& plyJson : outputsJson["plys"]) {
                PlyData plyData;
                plyData.is_imported = plyJson["is_imported"].get<bool>();
                plyData.ply_path = plyJson["ply_path"].get<std::string>();
                plyData.ply_training_iter_number = plyJson["ply_training_iter_number"].get<int>();
                plyData.ply_name = plyJson["ply_name"].get<std::string>();
                data.outputs.plys.push_back(plyData);
            }
        }

        // Store any additional fields for future compatibility
        data.additional_fields = json;
        // Remove known fields to keep only unknown ones
        data.additional_fields.erase("project_info");
        data.additional_fields.erase("version");
        data.additional_fields.erase("project_name");
        data.additional_fields.erase("project_creation_time");
        data.additional_fields.erase("project_last_update_time");
        data.additional_fields.erase("data");
        data.additional_fields.erase("outputs");

        return data;
    }

    nlohmann::ordered_json Project::serializeProjectData(const ProjectData& data) const {
        nlohmann::ordered_json json;

        // Add project info as the first field
        json["project_info"] = FILE_HEADER;
        json["version"] = data.version.toString();
        json["project_name"] = data.project_name;
        json["project_creation_time"] = data.project_creation_time;
        json["project_last_update_time"] = data.project_last_update_time;
        json["project_output_folder"] = data.data_set_info.output_path;

        // Data section
        json["data"]["data_path"] = data.data_set_info.data_path;
        json["data"]["data_type"] = data.data_set_info.data_type;
        json["data"]["resize_factor"] = data.data_set_info.resize_factor;
        json["data"]["test_every"] = data.data_set_info.test_every;
        json["data"]["images"] = data.data_set_info.images;

        // training optimization
        json["training"]["optimization"] = data.optimization.to_json();

        // Outputs section
        json["outputs"]["plys"] = nlohmann::ordered_json::array();
        for (const auto& ply : data.outputs.plys) {
            nlohmann::ordered_json plyJson;
            plyJson["is_imported"] = ply.is_imported;
            plyJson["ply_path"] = ply.ply_path.string();
            plyJson["ply_training_iter_number"] = ply.ply_training_iter_number;
            plyJson["ply_name"] = ply.ply_name;
            json["outputs"]["plys"].push_back(plyJson);
        }

        // Merge any additional fields
        if (!data.additional_fields.empty()) {
            json.update(data.additional_fields);
        }

        return json;
    }

    std::string Project::generateCurrentTimeStamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }

    // Convenience methods
    void Project::setProjectName(const std::string& name) {
        project_data_.project_name = name;
    }

    void Project::setDataInfo(const param::DatasetConfig& data_config) {
        project_data_.data_set_info = DataSetInfo(data_config);
        std::string datatype = IsColmapData(project_data_.data_set_info.data_path) ? "Colmap" : "Blender";

        project_data_.data_set_info.data_type = datatype;

        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    bool Project::addPly(const PlyData& ply_to_be_added) {

        for (const auto& ply : project_data_.outputs.plys) {
            if (ply.ply_name == ply_to_be_added.ply_name) {
                LOG_ERROR("can not insert two plys with the same name");
                return false;
            }
        }

        project_data_.outputs.plys.push_back(ply_to_be_added);

        if (update_file_on_change_ && !output_file_name_.empty()) {
            if (!writeToFile()) {
                return false;
            }
        }
        return true;
    }

    std::vector<PlyData> Project::getPlys() const {
        return project_data_.outputs.plys;
    }

    void Project::removePly(size_t index) {
        if (index < project_data_.outputs.plys.size()) {
            project_data_.outputs.plys.erase(project_data_.outputs.plys.begin() + index);
        }

        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    void Project::clearPlys() {
        project_data_.outputs.plys.clear();
        if (update_file_on_change_ && !output_file_name_.empty()) {
            writeToFile();
        }
    }

    bool Project::isCompatible(const Version& fileVersion) const {
        return fileVersion <= CURRENT_VERSION;
    }

    bool Project::validateProjectData() const {
        return !project_data_.project_name.empty() &&
               !project_data_.data_set_info.data_path.empty() &&
               !project_data_.data_set_info.data_type.empty();
    }

    bool Project::portProjectToDir(const std::filesystem::path& dst_dir) {

        namespace fs = std::filesystem;

        if (!std::filesystem::is_directory(dst_dir)) {
            LOG_ERROR("PortProjectToDir: Directory does not exists {}", dst_dir.string());
            return false;
        }

        const fs::path src_project_dir = getProjectOutputFolder();

        setProjectOutputFolder(dst_dir);
        if (!fs::is_regular_file(output_file_name_)) {
            LOG_ERROR("PortProjectToDir: {} orig path does not exists", output_file_name_.string());
        }
        const std::string proj_filename = output_file_name_.filename().string();
        const fs::path dst_project_file_path = dst_dir / proj_filename;

        setProjectFileName(dst_project_file_path);

        // Copy all ply files into new directory
        for (auto& ply : project_data_.outputs.plys) {
            try {
                if (!fs::exists(ply.ply_path)) {
                    LOG_ERROR("PortProjectToDir: ply file does not exist: {}", ply.ply_path.string());
                    return false;
                }

                // Keep same filename, copy to dst_dir
                fs::path dst_ply_path = dst_dir / ply.ply_path.filename();

                // Overwrite if already exists
                fs::copy_file(ply.ply_path, dst_ply_path, fs::copy_options::overwrite_existing);

                // Update metadata path
                ply.ply_path = dst_ply_path;

            } catch (const fs::filesystem_error& e) {
                LOG_ERROR("PortProjectToDir: failed to copy ply {} -> {}. reason: {}",
                          ply.ply_path.string(), (dst_dir / ply.ply_path.filename()).string(), e.what());
                return false;
            }
        }

        // Finally, write updated project file
        if (!writeToFile()) {
            LOG_ERROR("PortProjectToDir: failed to write updated project file to {}", dst_project_file_path.string());
            return false;
        }

        LOG_INFO("Project was successfully ported to {}", dst_project_file_path.string());

        return true;
    }

    /**
     * [功能描述]：创建新的项目实例，设置项目名称、路径、输出目录、数据集信息和优化参数。
     * @param data [参数说明]：数据集配置，包含数据路径、输出路径和项目路径等信息。
     * @param opt [参数说明]：优化参数配置，包含训练和渲染相关的参数设置。
     * @param project_name [参数说明]：项目名称，用于标识项目。
     * @param update_file_on_change [参数说明]：是否在项目变更时自动更新文件，true表示自动更新。
     * @return [返回值说明]：返回创建的项目智能指针，如果创建失败则返回nullptr。
     */
    std::shared_ptr<Project> CreateNewProject(const gs::param::DatasetConfig& data,
                                              const param::OptimizationParameters& opt,
                                              const std::string& project_name,
                                              bool update_file_on_change) {

        // 创建新的项目实例，传入自动更新标志
        auto project = std::make_shared<gs::management::Project>(update_file_on_change);

        // 设置项目名称
        project->setProjectName(project_name);
        
        // 验证输出路径不能为空
        if (data.output_path.empty()) {
            LOG_ERROR("output_path is empty");
            return nullptr;
        }
        
        // 处理项目文件路径
        std::filesystem::path project_path = data.project_path;
        if (project_path.empty()) {
            // 如果项目路径为空，则在输出目录下创建默认项目文件
            project_path = data.output_path / ("project" + Project::EXTENSION);
            LOG_INFO("project_path is empty - creating new project{} file", Project::EXTENSION);
        }

        // 验证项目文件扩展名必须正确
        if (project_path.extension() != Project::EXTENSION) {
            LOG_ERROR("project_path must be {} file: {}", Project::EXTENSION, project_path.string());
            return nullptr;
        }
        
        // 验证项目文件路径必须包含父目录
        if (project_path.parent_path().empty()) {
            LOG_ERROR("project_path must have parent directory: project_path: {} ", project_path.string());
            return nullptr;
        }

        try {
            // 设置项目文件名
            project->setProjectFileName(project_path);
            // 设置项目输出目录
            project->setProjectOutputFolder(data.output_path);
            // 设置数据集信息
            project->setDataInfo(data);
            // 设置优化参数
            project->setOptimizationParams(opt);
        } catch (const std::exception& e) {
            // 捕获并记录设置项目文件时的异常
            LOG_ERROR("Error writing project file: {}", e.what());
            return nullptr;
        }

        // 返回成功创建的项目实例
        return project;
    }

    std::filesystem::path FindProjectFile(const std::filesystem::path& directory) {
        if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
            return {};
        }

        std::filesystem::path foundPath;
        int count = 0;

        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".ls") {
                ++count;
                if (count == 1) {
                    foundPath = entry.path();
                } else {
                    LOG_ERROR("Multiple .ls files found in {}", directory.string());
                    return {};
                }
            }
        }

        if (count == 0) {
            return {};
        }
        return foundPath;
    }

    void clear_directory(const std::filesystem::path& path) {
        namespace fs = std::filesystem;
        for (const auto& entry : fs::directory_iterator(path)) {
            fs::remove_all(entry);
        }
    }

    std::shared_ptr<Project> CreateTempNewProject(const gs::param::DatasetConfig& data,
                                                  const param::OptimizationParameters& opt,
                                                  const std::string& project_name, bool udpdate_file_on_change) {
        namespace fs = std::filesystem;
        gs::param::DatasetConfig data_with_temp_output = data;

        auto temp_path = fs::temp_directory_path() / "LichtFeldStudio";
        if (fs::exists(temp_path)) {
            clear_directory(temp_path);
            LOG_INFO("Project temoprary directory exists removing its contenet {}", temp_path.string());
        } else {
            try {
                bool success = fs::create_directories(temp_path);
                if (!success) {
                    LOG_ERROR("failed to create temporary directory {}", temp_path.string());
                    return nullptr;
                }
                LOG_INFO("Project created temoprary directory successfuly: {}", temp_path.string());

            } catch (const fs::filesystem_error& e) {
                LOG_ERROR("failed to create temporary directory {}. reason: {}", temp_path.string(), e.what());
                return nullptr;
            }
        }
        data_with_temp_output.output_path = temp_path;
        auto project = CreateNewProject(data_with_temp_output, opt, project_name, udpdate_file_on_change);

        if (project) {
            project->setIsTempProject(true);
        }

        return project;
    }

} // namespace gs::management