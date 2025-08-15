#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include <imgui.h>
#include "visualizer_impl.hpp"

/**
 * [文件描述]：训练控制面板实现文件
 * 功能：实现训练过程的GUI控制界面，包括开始、暂停、停止训练等操作
 * 用途：为用户提供直观的训练控制界面和实时状态监控
 */

namespace gs::gui::panels {

    /**
     * [功能描述]：绘制训练控制面板的主函数
     * @param ctx：UI上下文对象，包含GUI状态和访问器（当前未使用但保留参数）
     * 
     * 该函数负责渲染完整的训练控制界面，包括：
     * - 训练状态查询和显示
     * - 根据当前状态显示相应的控制按钮
     * - 检查点保存功能和反馈
     * - 训练进度和损失信息显示
     * - 训练时间监控（新增功能）
     */
    void DrawTrainingControls([[maybe_unused]] const UIContext& ctx) {
        // =============================================================================
        // 面板标题和分隔线
        // =============================================================================
        ImGui::Text("训练控制");    // 面板标题
        ImGui::Separator();                 // 视觉分隔线

        // =============================================================================
        // 获取训练面板状态（单例模式）
        // =============================================================================
        // TrainingPanelState管理面板的内部状态，如保存进度反馈等
        auto& state = TrainingPanelState::getInstance();

        // =============================================================================
        // 通过事件系统查询训练器状态
        // =============================================================================
        // 使用事件驱动的方式查询当前训练器状态，避免直接耦合
        events::query::TrainerState response;  // 用于接收响应的对象
        bool has_response = false;              // 响应接收标志

        // 设置一次性事件处理器来接收查询响应
        // 使用lambda表达式捕获响应对象和标志
        [[maybe_unused]] auto handler = events::query::TrainerState::when([&response, &has_response](const auto& r) {
            response = r;           // 保存接收到的响应
            has_response = true;    // 标记已接收到响应
        });

        // 发送训练器状态查询事件
        events::query::GetTrainerState{}.emit();

        // TODO: 在实际应用中，这应该是异步的
        // 当前实现假设响应是立即的，但在真实场景中需要异步处理
        // 未来可能需要重构为异步回调模式

        // =============================================================================
        // 处理无响应情况
        // =============================================================================
        if (!has_response) {
            // 如果没有收到响应，说明训练器尚未加载或连接失败
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "未加载训练器");
            return;  // 提前返回，不显示控制按钮
        }

        // =============================================================================
        // 根据训练器状态渲染相应的控制界面
        // =============================================================================
        switch (response.state) {
        case events::query::TrainerState::State::Idle:
            // 训练器空闲状态：未加载任何训练数据
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "未加载训练器");
            break;

        case events::query::TrainerState::State::Ready:
            // 训练器就绪状态：已加载数据，可以开始训练
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));        // 绿色按钮
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));  // 悬停时的亮绿色
            if (ImGui::Button("开始训练", ImVec2(-1, 0))) {  // 全宽按钮
                events::cmd::StartTraining{}.emit();              // 发送开始训练命令
            }
            ImGui::PopStyleColor(2);  // 恢复默认按钮颜色
            break;

        case events::query::TrainerState::State::Running:
            // 训练器运行状态：正在进行训练
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));        // 橙色按钮
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));  // 悬停时的亮橙色
            if (ImGui::Button("暂停", ImVec2(-1, 0))) {          // 暂停按钮
                events::cmd::PauseTraining{}.emit();             // 发送暂停训练命令
            }
            ImGui::PopStyleColor(2);
            break;

        case events::query::TrainerState::State::Paused:
            // 训练器暂停状态：训练已暂停，可以恢复或停止
            
            // 恢复训练按钮
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));        // 绿色
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("恢复", ImVec2(-1, 0))) {
                events::cmd::ResumeTraining{}.emit();            // 发送恢复训练命令
            }
            ImGui::PopStyleColor(2);

            // 永久停止按钮
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));        // 红色
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));  // 悬停时的亮红色
            if (ImGui::Button("永久停止", ImVec2(-1, 0))) {
                events::cmd::StopTraining{}.emit();              // 发送停止训练命令
            }
            ImGui::PopStyleColor(2);
            break;

        case events::query::TrainerState::State::Completed:
            // 训练完成状态：训练已成功完成
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "训练完成!");
            break;

        case events::query::TrainerState::State::Error:
            // 训练错误状态：训练过程中发生了错误
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "训练错误!");
            
            // 如果有错误消息，显示详细信息
            if (response.error_message) {
                ImGui::TextWrapped("%s", response.error_message->c_str());  // 自动换行显示错误详情
            }
            break;
        }

        // =============================================================================
        // 检查点保存功能（仅在训练进行时可用）
        // =============================================================================
        if (response.state == events::query::TrainerState::State::Running ||
            response.state == events::query::TrainerState::State::Paused) {

            // 保存检查点按钮
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));        // 蓝色按钮
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));  // 悬停时的亮蓝色
            if (ImGui::Button("保存检查点", ImVec2(-1, 0))) {
                events::cmd::SaveCheckpoint{}.emit();            // 发送保存检查点命令
                
                // 启动保存进度反馈
                state.save_in_progress = true;                   // 标记保存正在进行
                state.save_start_time = std::chrono::steady_clock::now();  // 记录保存开始时间
            }
            ImGui::PopStyleColor(2);
        }

        // =============================================================================
        // 保存操作反馈显示
        // =============================================================================
        if (state.save_in_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.save_start_time)     // 计算已经过的时间
                               .count();
            
            if (elapsed < 2000) {  // 显示反馈信息2秒钟
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "检查点保存成功!");
            } else {
                state.save_in_progress = false;  // 2秒后自动隐藏反馈信息
            }
        }

        // =============================================================================
        // 训练状态信息显示区域
        // =============================================================================
        ImGui::Separator();  // 分隔线，将控制按钮和状态信息分开
        
        // 显示当前训练器状态的文字描述
        ImGui::Text("状态: %s", widgets::GetTrainerStateString(static_cast<int>(response.state)));
        
        // 显示当前训练迭代次数
        ImGui::Text("迭代次数: %d", response.current_iteration);
        
        // 显示当前损失值（保留6位小数精度）
        ImGui::Text("损失值: %.6f", response.current_loss);

        // =============================================================================
        // 训练时间显示（修改后的实现）
        // =============================================================================
        if (auto* viewer = ctx.viewer) {  // 检查UIContext是否有viewer指针
            if (auto training_info = viewer->getTrainingInfo()) {  // 获取训练信息对象
                
                // 根据训练状态显示相应的时间信息
                std::string time_text;
                std::string status_text;
                ImVec4 status_color;
                
                switch (response.state) {
                case events::query::TrainerState::State::Idle:
                case events::query::TrainerState::State::Ready:
                    // 未开始训练：显示00:00
                    time_text = "00:00";
                    status_text = "未开始";
                    status_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // 灰色
                    break;
                    
                case events::query::TrainerState::State::Running:
                    // 正在训练：显示当前训练时间
                    {
                        double current_seconds = training_info->getCurrentTrainingTimeSeconds();
                        if (current_seconds > 0) {
                            time_text = training_info->formatTrainingTime();
                        } else {
                            time_text = "00:00";
                        }
                        status_text = "● 训练中";
                        status_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);  // 绿色
                    }
                    break;
                    
                case events::query::TrainerState::State::Paused:
                    // 暂停状态：显示暂停时的累计时间
                    {
                        double total_seconds = training_info->getTotalTrainingTimeSeconds();
                        if (total_seconds > 0) {
                            // 使用自定义格式化函数显示暂停时的时间
                            int hours = static_cast<int>(total_seconds / 3600);
                            int minutes = static_cast<int>((total_seconds - hours * 3600) / 60);
                            int secs = static_cast<int>(total_seconds) % 60;
                            
                            if (hours > 0) {
                                time_text = std::format("{:02d}:{:02d}:{:02d}", hours, minutes, secs);
                            } else {
                                time_text = std::format("{:02d}:{:02d}", minutes, secs);
                            }
                        } else {
                            time_text = "00:00";
                        }
                        status_text = "⏸ 已暂停";
                        status_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);  // 黄色
                    }
                    break;
                    
                case events::query::TrainerState::State::Completed:
                    // 训练完成：显示最终总训练时间
                    {
                        // 使用最终时间，不包含当前正在运行的时间
                        double total_seconds = training_info->getTotalTrainingTimeSeconds();
                        if (total_seconds > 0) {
                            // 使用专门的格式化方法显示最终时间
                            time_text = training_info->formatFinalTrainingTime();
                        } else {
                            time_text = "00:00";
                        }
                        status_text = "✓ 已完成";
                        status_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // 绿色
                    }
                    break;
                    
                case events::query::TrainerState::State::Error:
                    // 训练错误：显示错误前的累计时间
                    {
                        // 使用最终时间，不包含当前正在运行的时间
                        double total_seconds = training_info->getTotalTrainingTimeSeconds();
                        if (total_seconds > 0) {
                            time_text = training_info->formatFinalTrainingTime();
                        } else {
                            time_text = "00:00";
                        }
                        status_text = "✗ 错误";
                        status_color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // 红色
                    }
                    break;
                }
                
                // 显示训练时间（一直显示）
                ImGui::Text("训练时间: %s", time_text.c_str());
                
                // 在同一行显示训练状态指示器
                ImGui::SameLine();
                ImGui::TextColored(status_color, "%s", status_text.c_str());
            }
        }
    }
} // namespace gs::gui::panels