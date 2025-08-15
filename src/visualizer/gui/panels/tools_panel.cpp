#include "gui/panels/tools_panel.hpp"
#include "core/events.hpp"
#include "tools/tool_base.hpp"
#include "tools/tool_manager.hpp"
#include "visualizer_impl.hpp"
#include <format>
#include <imgui.h>
#include <string>

namespace gs::gui::panels {

    void DrawToolsPanel(const UIContext& ctx) {
        if (!ctx.viewer) {
            return;
        }

        auto* tool_manager = ctx.viewer->getToolManager();
        if (!tool_manager) {
            return;
        }

        // Tools section header
        ImGui::Text("工具");
        ImGui::Separator();

        // Get all active tools
        auto tools = tool_manager->getActiveTools();

        if (tools.empty()) {
            ImGui::TextDisabled("没有可用的工具");
            ImGui::Spacing();
            return;
        }

        // Draw each tool
        for (auto* tool : tools) {
            detail::DrawToolUI(ctx, tool);
        }

        ImGui::Spacing();

        // Remove the Tool Manager button - not needed
    }

    namespace detail {

        void DrawToolUI(const UIContext& ctx, gs::visualizer::ToolBase* tool) {
            if (!tool) {
                return;
            }

            ImGui::PushID(tool);

            // Draw tool header with checkbox
            bool enabled_changed = DrawToolHeader(tool);

            if (enabled_changed) {
                // Emit enable/disable events
                if (tool->isEnabled()) {
                    events::tools::ToolEnabled{
                        .tool_name = std::string(tool->getName())}
                        .emit();

                    events::notify::Log{
                        .level = events::notify::Log::Level::Info,
                        .message = std::format("启用工具: {}", tool->getName()),
                        .source = "ToolsPanel"}
                        .emit();
                } else {
                    events::tools::ToolDisabled{
                        .tool_name = std::string(tool->getName())}
                        .emit();

                    events::notify::Log{
                        .level = events::notify::Log::Level::Info,
                        .message = std::format("禁用工具: {}", tool->getName()),
                        .source = "ToolsPanel"}
                        .emit();
                }
            }

            // Show tool-specific UI only if enabled
            if (tool->isEnabled()) {
                // Indent the tool's UI
                ImGui::Indent();

                // Let the tool render its own UI
                bool dummy_open = true;
                tool->renderUI(ctx, &dummy_open);

                ImGui::Unindent();
            }

            ImGui::PopID();

            // Add some spacing between tools
            ImGui::Spacing();
        }

        bool DrawToolHeader(gs::visualizer::ToolBase* tool) {
            bool enabled = tool->isEnabled();
            bool changed = false;

            // Tool icon (for future use)
            const char* icon = GetToolIcon(tool->getName());

            // Draw checkbox with tool name
            if (icon) {
                std::string label = std::format("{} {}", icon, tool->getName());
                if (ImGui::Checkbox(label.c_str(), &enabled)) {
                    tool->setEnabled(enabled);
                    changed = true;
                }
            }

            // Remove this: ImGui::PopStyleColor();

            // Show description as tooltip when hovering
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                ImGui::TextUnformatted(tool->getDescription().data());
                ImGui::PopTextWrapPos();

                // Add additional info
                ImGui::Separator();
                ImGui::TextDisabled("状态: %s", enabled ? "启用" : "禁用");

                ImGui::EndTooltip();
            }

            // Remove the settings button - keep it simple

            return changed;
        }

        const char* GetToolIcon(const std::string_view& tool_name) {
            // Map tool names to icons
            // Using simple text icons for now, but could use FontAwesome or similar
            if (tool_name == "裁剪框") {
                return "[□]"; // Box icon
            } else if (tool_name == "世界坐标系变换") {
                return "[⊕]"; // Transform icon
            } else if (tool_name == "背景") {
                return "[◐]"; // Background/color icon
            }

            // No warning needed - just return default icon
            return "[?]"; // Default icon
        }

    } // namespace detail

} // namespace gs::gui::panels