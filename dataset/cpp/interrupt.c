#include "arm/interrupt.h"
#include "kernel/interrupt.h"
#include "arm/vmm.h"
#include "libc/string.h"
#include "kernel/log.h"

#define IRQ_NUMS 96

Tick *tick_init(Tick *tick, TickHandler handler, const char *name) {
    tick->handler = handler;
    memset(tick->name, 0, sizeof(tick->name));
    memcpy(tick->name, name, sizeof(tick->name) - 1);
    tick->node.next = nullptr;
    tick->node.prev = nullptr;
    return tick;
}

void interrupt_manager_default_register_tick(InterruptManager *manager, Tick *tick) {
    if (manager->ticks == nullptr) {
        manager->ticks = tick;
    } else {
        klist_append(&manager->ticks->node, &tick->node);
    }
    LogInfo("[Interrupt]: tick '%s' registered.\n", tick->name)
}

void interrupt_manager_default_un_register_tick(InterruptManager *manager, Tick *tick) {
    klist_remove_node(&tick->node);
}

void interrupt_manager_default_register(InterruptManager *manager, Interrupt interrupt) {
    manager->interrupts[interrupt.interruptNumber].interruptNumber = interrupt.interruptNumber;
    manager->interrupts[interrupt.interruptNumber].handler = interrupt.handler;
    manager->interrupts[interrupt.interruptNumber].clearHandler = interrupt.clearHandler;
    manager->interrupts[interrupt.interruptNumber].enableHandler = interrupt.enableHandler;
    manager->interrupts[interrupt.interruptNumber].disableHandler = interrupt.disableHandler;
    memcpy(manager->interrupts[interrupt.interruptNumber].name, interrupt.name, sizeof(interrupt.name));

    manager->registed[interrupt.interruptNumber] = 1;

    manager->interrupts[interrupt.interruptNumber].enableHandler();
}

void interrupt_manager_default_unregister(InterruptManager *manager, Interrupt interrupt) {
    manager->interrupts[interrupt.interruptNumber].disableHandler();
    manager->interrupts[interrupt.interruptNumber].interruptNumber = 0;
    manager->interrupts[interrupt.interruptNumber].handler = nullptr;
    manager->interrupts[interrupt.interruptNumber].clearHandler = nullptr;
    manager->interrupts[interrupt.interruptNumber].disableHandler = nullptr;
    manager->registed[interrupt.interruptNumber] = 0;
}

void interrupt_manager_default_enable(InterruptManager *manager) {
    if (!arch_is_interrupt_enabled()) {
        arch_enable_interrupt();
        LogInfo("[Interrupt]: enable\n");
    }
}

void interrupt_manager_default_disable(InterruptManager *manager) {
    if (arch_is_interrupt_enabled()) {
        arch_disable_interrupt();
        LogInfo("[Interrupt]: disable\n");
    }
}

void interrupt_manager_default_init(InterruptManager *manager) {
    LogInfo("[InterruptManager]: init\n")
    if(manager->physicalInit != nullptr){
        manager->physicalInit();
    }
}

void interrupt_manager_default_tick(InterruptManager *manager) {
    Tick *tick = manager->ticks;
    if (tick != nullptr) {
        LogInfo("[InterruptManager]: tick '%s' triggered\n", tick->name);
        tick->handler();
        while (tick->node.next != nullptr) {
            Tick *next = getNode(tick->node.next, Tick, node);
            LogInfo("[InterruptManager]: tick '%s' triggered\n", tick->name);
            next->handler();
            tick = next;
        }
    } else {
        LogError("[InterruptManager]: no tick registered\n");
    }
}


void interrupt_manager_default_interrupt(InterruptManager *manager) {
    for (uint32_t interrupt_no = 0; interrupt_no < IRQ_NUMS; interrupt_no++) {
        if (manager->registed[interrupt_no] == 1 /* &&  manager->interrupts[interrupt_no].pendingHandler()*/) {
            LogInfo("[Interrupt]: interrupt '%s' triggered.\n", manager->interrupts[interrupt_no].name);
            if (manager->interrupts[interrupt_no].clearHandler != nullptr) {
                manager->interrupts[interrupt_no].clearHandler();
            }
            manager->interrupts[interrupt_no].handler();
        }
    }
}

void interrupt_manager_default_register_physical_init(struct InterruptManager *manager, InterruptManagerPhysicalInit init){
    manager->physicalInit = init;
}

InterruptManager *interrupt_manager_create(InterruptManager *manger) {
    manger->operation.init = (InterruptManagerOperationInit) interrupt_manager_default_init;
    manger->operation.registerInterrupt = (InterruptManagerOperationRegister) interrupt_manager_default_register;
    manger->operation.unRegisterInterrupt = (InterruptManagerOperationUnRegister) interrupt_manager_default_unregister;
    manger->operation.enableInterrupt = (InterruptManagerOperationEnableInterrupt) interrupt_manager_default_enable;
    manger->operation.disableInterrupt = (InterruptManagerOperationDisableInterrupt) interrupt_manager_default_disable;
    manger->operation.registerTick = (InterruptManagerOperationRegisterTick) interrupt_manager_default_register_tick;
    manger->operation.unRegisterTick = (InterruptManagerOperationUnRegisterTick) interrupt_manager_default_un_register_tick;
    manger->operation.interrupt = (InterruptManagerOperationInterrupt) interrupt_manager_default_interrupt;
    manger->operation.tick = (InterruptManagerOperationTick) interrupt_manager_default_tick;
    manger->operation.registerPhysicalInit = (InterruptManagerPhysicalInit) interrupt_manager_default_register_physical_init;
    manger->physicalInit = nullptr;

    manger->ticks = nullptr;
    memset((char *) manger->registed, 0, IRQ_NUMS * sizeof(uint32_t));
    manger->operation.disableInterrupt(manger);

    LogInfo("[InterruptManager]: created\n")

    return manger;
}
