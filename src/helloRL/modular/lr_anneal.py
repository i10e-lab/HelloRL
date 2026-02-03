from helloRL.modular.foundation import *

@dataclass
class LRScheduleLinearAnneal(LRSchedule):
    start_lr: float = 0.001
    end_lr: float = 0.0

    def get_lr(self, step: int, total_steps: int) -> float:
        safe_total = max(1, int(total_steps))
        frac = float(step) / float(safe_total)
        frac = min(max(frac, 0.0), 1.0)

        # linear interpolation: start -> end
        return self.start_lr + ((self.end_lr - self.start_lr) * frac)