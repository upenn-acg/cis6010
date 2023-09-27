#pragma once

enum BarrierFlavor { KERNEL_LAUNCH_BARRIER, SPIN_BARRIER, TWO_LEVEL_BARRIER };

extern void barrierTest(const BarrierFlavor);
extern const char* barrierName(const BarrierFlavor);