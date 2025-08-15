#pragma once

#include "core/events.hpp"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <thread>

namespace gs {

    class MemoryMonitor {
    public:
        MemoryMonitor() = default;

        ~MemoryMonitor() {
            stop();
        }

        void start(std::chrono::milliseconds interval = std::chrono::milliseconds(1000)) {
            if (running_)
                return;

            running_ = true;
            monitor_thread_ = std::thread([this, interval]() {
                while (running_) {
                    updateMemoryStats();
                    std::this_thread::sleep_for(interval);
                }
            });
        }

        void stop() {
            if (running_) {
                running_ = false;
                if (monitor_thread_.joinable()) {
                    monitor_thread_.join();
                }
            }
        }

        void updateMemoryStats() {
            // GPU Memory
            size_t gpu_free, gpu_total;
            cudaMemGetInfo(&gpu_free, &gpu_total);
            size_t gpu_used = gpu_total - gpu_free;
            float gpu_percent = (float)gpu_used / gpu_total * 100.0f;

            // RAM (simplified - platform specific implementation needed)
            size_t ram_used = 0, ram_total = 0;
            float ram_percent = 0.0f;
            // TODO: Implement platform-specific RAM monitoring

            // Publish memory usage event
            events::state::MemoryUsage{
                .gpu_used = gpu_used,
                .gpu_total = gpu_total,
                .gpu_percent = gpu_percent,
                .ram_used = ram_used,
                .ram_total = ram_total,
                .ram_percent = ram_percent}
                .emit();

            // Check for warnings
            if (gpu_percent > 90.0f && !gpu_warning_sent_) {
                events::notify::MemoryWarning{
                    .type = events::notify::MemoryWarning::Type::GPU,
                    .usage_percent = gpu_percent,
                    .message = "GPU 内存使用严重! 考虑减少批量大小或高斯数量."}
                    .emit();
                gpu_warning_sent_ = true;
            } else if (gpu_percent < 85.0f) {
                gpu_warning_sent_ = false;
            }
        }

    private:
        std::thread monitor_thread_;
        std::atomic<bool> running_{false};
        bool gpu_warning_sent_ = false;
    };

} // namespace gs