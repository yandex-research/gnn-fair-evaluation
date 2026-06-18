import bisect
from collections.abc import Sequence
from typing import Any

import optuna
import optuna.distributions
import optuna.samplers
import optuna.trial


class MultiStageSampler(optuna.samplers.BaseSampler):
    """A sampler that delegates to different child samplers based on trial count.

    Each stage specifies a sampler and a number of trials. Once the cumulative
    trial count reaches a stage boundary, the next stage's sampler takes over.
    The new sampler sees all prior trials as warm-up history.
    """

    def __init__(self, stages: list[dict[str, Any]], seed: int) -> None:
        self._samplers: list[optuna.samplers.BaseSampler] = []
        self._boundaries: list[int] = []
        cumulative = 0
        for stage in stages:
            sampler_type = stage.get('sampler_type', 'TPESampler')
            sampler_kwargs = stage.get('sampler', {})
            sampler = getattr(optuna.samplers, sampler_type)(
                **sampler_kwargs, seed=seed
            )
            self._samplers.append(sampler)
            cumulative += stage['n_trials']
            self._boundaries.append(cumulative)

    def _get_current_sampler(self, study: optuna.Study) -> optuna.samplers.BaseSampler:
        idx = bisect.bisect_right(self._boundaries, len(study.trials))
        idx = min(idx, len(self._samplers) - 1)
        return self._samplers[idx]

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._get_current_sampler(study).infer_relative_search_space(
            study, trial
        )

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        return self._get_current_sampler(study).sample_relative(
            study, trial, search_space
        )

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        return self._get_current_sampler(study).sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        self._get_current_sampler(study).before_trial(study, trial)

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._get_current_sampler(study).after_trial(study, trial, state, values)


OPTUNA_SAMPLERS: dict[str, type[optuna.samplers.BaseSampler]] = {
    'MultiStageSampler': MultiStageSampler,
}
