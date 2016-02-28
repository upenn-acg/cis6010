#pragma once

enum BarrierFlavor { SPIN_BARRIER, TWO_LEVEL_BARRIER };

extern void barrierTest(const BarrierFlavor);