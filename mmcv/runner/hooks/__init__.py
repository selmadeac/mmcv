# Copyright (c) OpenMMLab. All rights reserved.
from .checkpoint import CheckpointHook
from .closure import ClosureHook
from .ema import EMAHook
from .evaluation import DistEvalHook, EvalHook
from .hook import HOOKS, Hook
from .iter_timer import IterTimerHook
from .logger import (ClearMLLoggerHook, DvcliveLoggerHook, LoggerHook,
                     MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook,
                     SegmindLoggerHook, TensorboardLoggerHook, TextLoggerHook,
                     WandbLoggerHook)
from .lr_updater import (ReserveSigmoidLrUpdaterHook,DampedCosineAnnealingRestartLrUpdaterHook,CenterformerStepCycLrUpdaterHook,CosineAnnealingLrUpdaterHook,
                         CosineRestartLrUpdaterHook, CyclicLrUpdaterHook,
                         ExpLrUpdaterHook, FixedLrUpdaterHook,CosineAnnealingRestartsWithDecayLrUpdaterHook,
                         FlatCosineAnnealingLrUpdaterHook, InvLrUpdaterHook,
                         LinearAnnealingLrUpdaterHook, LrUpdaterHook,
                         OneCycleLrUpdaterHook, PolyLrUpdaterHook,CenterformerCosAnLrUpdaterHook,
                         StepLrUpdaterHook,FixMultiplyLrUpdaterHook)
from .memory import EmptyCacheHook
from .momentum_updater import (CosineAnnealingMomentumUpdaterHook,
                               CenterformerStepCycMomentumUpdaterHook,CyclicMomentumUpdaterHook,
                               LinearAnnealingMomentumUpdaterHook,
                               MomentumUpdaterHook,
                               OneCycleMomentumUpdaterHook,
                               StepMomentumUpdaterHook,CenterformerCosAnMomentumUpdaterHook)
from .optimizer import (Fp16OptimizerHook, GradientCumulativeFp16OptimizerHook,
                        GradientCumulativeOptimizerHook, OptimizerHook)
from .profiler import ProfilerHook
from .sampler_seed import DistSamplerSeedHook
from .sync_buffer import SyncBuffersHook

__all__ = [
    'ReserveSigmoidLrUpdaterHook','HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'FixedLrUpdaterHook', 'StepLrUpdaterHook', 'ExpLrUpdaterHook',
    'PolyLrUpdaterHook', 'InvLrUpdaterHook', 'CosineAnnealingLrUpdaterHook',
    'FlatCosineAnnealingLrUpdaterHook', 'CosineRestartLrUpdaterHook',
    'CyclicLrUpdaterHook', 'OneCycleLrUpdaterHook', 'OptimizerHook',
    'Fp16OptimizerHook', 'IterTimerHook', 'DistSamplerSeedHook',
    'EmptyCacheHook', 'LoggerHook', 'MlflowLoggerHook', 'PaviLoggerHook',
    'TextLoggerHook', 'TensorboardLoggerHook', 'NeptuneLoggerHook','CosineAnnealingRestartsWithDecayLrUpdaterHook',
    'WandbLoggerHook', 'DvcliveLoggerHook', 'MomentumUpdaterHook','DampedCosineAnnealingRestartLrUpdaterHook',
    'StepMomentumUpdaterHook', 'CosineAnnealingMomentumUpdaterHook',
    'CyclicMomentumUpdaterHook', 'OneCycleMomentumUpdaterHook','CenterformerStepCycMomentumUpdaterHook',
    'SyncBuffersHook', 'EMAHook','CenterformerStepCycLrUpdaterHook', 'EvalHook', 'DistEvalHook', 'ProfilerHook',
    'GradientCumulativeOptimizerHook', 'GradientCumulativeFp16OptimizerHook',
    'SegmindLoggerHook', 'LinearAnnealingLrUpdaterHook','CenterformerCosAnLrUpdaterHook','CenterformerCosAnMomentumUpdaterHook'
    'LinearAnnealingMomentumUpdaterHook', 'ClearMLLoggerHook','FixMultiplyLrUpdaterHook',
]
