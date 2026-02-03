from dataclasses import field

from helloRL.modular.foundation import *
from helloRL.modular.monte_carlo import RolloutMethodMonteCarlo

@dataclass
class Params:
    reward_transform: RewardTransform = field(
        default_factory=RewardTransformNone
    )
    rollout_method: RolloutMethod = field(
        default_factory=RolloutMethodMonteCarlo
    )
    advantage_method: AdvantageMethod = field(
        default_factory=AdvantageMethodStandard
    )
    advantage_transform: AdvantageTransform = field(
        default_factory=AdvantageTransformNone
    )
    data_load_method: DataLoadMethod = field(
        default_factory=DataLoadMethodSingle
    )
    gradient_transform: GradientTransform = field(
        default_factory=GradientTransformNone
    )
    lr_schedule: LRSchedule = field(
        default_factory=LRScheduleConstant
    )
    gamma: float = 0.99