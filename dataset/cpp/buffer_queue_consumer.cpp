// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2014 The Android Open Source Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Parts of this implementation were based on:
// https://cs.android.com/android/platform/superproject/+/android-5.1.1_r38:frameworks/native/libs/gui/BufferQueueConsumer.cpp

#include "common/logging/log.h"
#include "core/hle/service/nvdrv/core/nvmap.h"
#include "core/hle/service/nvnflinger/buffer_item.h"
#include "core/hle/service/nvnflinger/buffer_queue_consumer.h"
#include "core/hle/service/nvnflinger/buffer_queue_core.h"
#include "core/hle/service/nvnflinger/producer_listener.h"
#include "core/hle/service/nvnflinger/ui/graphic_buffer.h"

namespace Service::android {

BufferQueueConsumer::BufferQueueConsumer(std::shared_ptr<BufferQueueCore> core_,
                                         Service::Nvidia::NvCore::NvMap& nvmap_)
    : core{std::move(core_)}, slots{core->slots}, nvmap(nvmap_) {}

BufferQueueConsumer::~BufferQueueConsumer() = default;

Status BufferQueueConsumer::AcquireBuffer(BufferItem* out_buffer,
                                          std::chrono::nanoseconds expected_present) {
    std::scoped_lock lock{core->mutex};

    // Check that the consumer doesn't currently have the maximum number of buffers acquired.
    const s32 num_acquired_buffers{
        static_cast<s32>(std::count_if(slots.begin(), slots.end(), [](const auto& slot) {
            return slot.buffer_state == BufferState::Acquired;
        }))};

    if (num_acquired_buffers >= core->max_acquired_buffer_count + 1) {
        LOG_ERROR(Service_Nvnflinger, "max acquired buffer count reached: {} (max {})",
                  num_acquired_buffers, core->max_acquired_buffer_count);
        return Status::InvalidOperation;
    }

    // Check if the queue is empty.
    if (core->queue.empty()) {
        return Status::NoBufferAvailable;
    }

    auto front(core->queue.begin());

    // If expected_present is specified, we may not want to return a buffer yet.
    if (expected_present.count() != 0) {
        constexpr auto MAX_REASONABLE_NSEC = 1000000000LL; // 1 second

        // The expected_present argument indicates when the buffer is expected to be presented
        // on-screen.
        while (core->queue.size() > 1 && !core->queue[0].is_auto_timestamp) {
            const auto& buffer_item{core->queue[1]};

            // If entry[1] is timely, drop entry[0] (and repeat).
            const auto desired_present = buffer_item.timestamp;
            if (desired_present < expected_present.count() - MAX_REASONABLE_NSEC ||
                desired_present > expected_present.count()) {
                // This buffer is set to display in the near future, or desired_present is garbage.
                LOG_DEBUG(Service_Nvnflinger, "nodrop desire={} expect={}", desired_present,
                          expected_present.count());
                break;
            }

            LOG_DEBUG(Service_Nvnflinger, "drop desire={} expect={} size={}", desired_present,
                      expected_present.count(), core->queue.size());

            if (core->StillTracking(*front)) {
                // Front buffer is still in mSlots, so mark the slot as free
                slots[front->slot].buffer_state = BufferState::Free;
            }

            core->queue.erase(front);
            front = core->queue.begin();
        }

        // See if the front buffer is ready to be acquired.
        const auto desired_present = front->timestamp;
        if (desired_present > expected_present.count() &&
            desired_present < expected_present.count() + MAX_REASONABLE_NSEC) {
            LOG_DEBUG(Service_Nvnflinger, "defer desire={} expect={}", desired_present,
                      expected_present.count());
            return Status::PresentLater;
        }

        LOG_DEBUG(Service_Nvnflinger, "accept desire={} expect={}", desired_present,
                  expected_present.count());
    }

    const auto slot = front->slot;
    *out_buffer = *front;

    LOG_DEBUG(Service_Nvnflinger, "acquiring slot={}", slot);

    // If the buffer has previously been acquired by the consumer, set graphic_buffer to nullptr to
    // avoid unnecessarily remapping this buffer on the consumer side.
    if (out_buffer->acquire_called) {
        out_buffer->graphic_buffer = nullptr;
    }

    core->queue.erase(front);

    // We might have freed a slot while dropping old buffers, or the producer  may be blocked
    // waiting for the number of buffers in the queue to decrease.
    core->SignalDequeueCondition();

    return Status::NoError;
}

Status BufferQueueConsumer::ReleaseBuffer(s32 slot, u64 frame_number, const Fence& release_fence) {
    if (slot < 0 || slot >= BufferQueueDefs::NUM_BUFFER_SLOTS) {
        LOG_ERROR(Service_Nvnflinger, "slot {} out of range", slot);
        return Status::BadValue;
    }

    std::shared_ptr<IProducerListener> listener;
    {
        std::scoped_lock lock{core->mutex};

        // If the frame number has changed because the buffer has been reallocated, we can ignore
        // this ReleaseBuffer for the old buffer.
        if (frame_number != slots[slot].frame_number) {
            return Status::StaleBufferSlot;
        }

        // Make sure this buffer hasn't been queued while acquired by the consumer.
        auto current(core->queue.begin());
        while (current != core->queue.end()) {
            if (current->slot == slot) {
                LOG_ERROR(Service_Nvnflinger, "buffer slot {} pending release is currently queued",
                          slot);
                return Status::BadValue;
            }
            ++current;
        }

        slots[slot].buffer_state = BufferState::Free;

        nvmap.FreeHandle(slots[slot].graphic_buffer->BufferId(), true);

        listener = core->connected_producer_listener;

        LOG_DEBUG(Service_Nvnflinger, "releasing slot {}", slot);

        core->SignalDequeueCondition();
    }

    // Call back without lock held
    if (listener != nullptr) {
        listener->OnBufferReleased();
    }

    return Status::NoError;
}

Status BufferQueueConsumer::Connect(std::shared_ptr<IConsumerListener> consumer_listener,
                                    bool controlled_by_app) {
    if (consumer_listener == nullptr) {
        LOG_ERROR(Service_Nvnflinger, "consumer_listener may not be nullptr");
        return Status::BadValue;
    }

    LOG_DEBUG(Service_Nvnflinger, "controlled_by_app={}", controlled_by_app);

    std::scoped_lock lock{core->mutex};

    if (core->is_abandoned) {
        LOG_ERROR(Service_Nvnflinger, "BufferQueue has been abandoned");
        return Status::NoInit;
    }

    core->consumer_listener = std::move(consumer_listener);
    core->consumer_controlled_by_app = controlled_by_app;

    return Status::NoError;
}

Status BufferQueueConsumer::GetReleasedBuffers(u64* out_slot_mask) {
    if (out_slot_mask == nullptr) {
        LOG_ERROR(Service_Nvnflinger, "out_slot_mask may not be nullptr");
        return Status::BadValue;
    }

    std::scoped_lock lock{core->mutex};

    if (core->is_abandoned) {
        LOG_ERROR(Service_Nvnflinger, "BufferQueue has been abandoned");
        return Status::NoInit;
    }

    u64 mask = 0;
    for (int s = 0; s < BufferQueueDefs::NUM_BUFFER_SLOTS; ++s) {
        if (!slots[s].acquire_called) {
            mask |= (1ULL << s);
        }
    }

    // Remove from the mask queued buffers for which acquire has been called, since the consumer
    // will not receive their buffer addresses and so must retain their cached information
    auto current(core->queue.begin());
    while (current != core->queue.end()) {
        if (current->acquire_called) {
            mask &= ~(1ULL << current->slot);
        }
        ++current;
    }

    LOG_DEBUG(Service_Nvnflinger, "returning mask {}", mask);
    *out_slot_mask = mask;
    return Status::NoError;
}

} // namespace Service::android
