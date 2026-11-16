"""Reader-score budgets; forecasts require measured aggregate throughput."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostPlan:
    steps: int
    group_size: int
    probe_every: int
    cube_dimensions: int
    wall_hours: float
    reserve_fraction: float = 0.2
    relation_cubes: int = 1
    validation_every: int = 0
    validation_examples: int = 0
    validation_methods: int = 3

    def __post_init__(self):
        if (self.steps < 0 or self.group_size < 2 or self.probe_every < 0
                or not 0 <= self.cube_dimensions <= 3 or self.wall_hours <= 0
                or not 0 <= self.reserve_fraction < 1 or self.relation_cubes < 0
                or self.validation_every < 0 or self.validation_examples < 0
                or self.validation_methods < 1):
            raise ValueError("invalid cost plan")

    @property
    def probe_steps(self):
        return math.ceil(self.steps / self.probe_every) if self.probe_every else 0

    @property
    def training_score_requests(self):
        # One shared bank, not a separate bank per sampled structure. Does not
        # credit cache hits, identical corners or reduced dimensions near K=0/N.
        return (self.steps * self.group_size
                + self.probe_steps * self.relation_cubes * 2 ** self.cube_dimensions)

    @property
    def validation_score_requests(self):
        rounds = self.steps // self.validation_every if self.validation_every else 0
        return rounds * self.validation_examples * self.validation_methods

    @property
    def score_requests(self):
        return self.training_score_requests + self.validation_score_requests

    @property
    def required_aggregate_rate(self):
        return self.score_requests / (self.wall_hours * 3600 * (1 - self.reserve_fraction))

    def estimated_hours(self, aggregate_scores_per_second):
        if aggregate_scores_per_second <= 0:
            raise ValueError("measured scoring throughput must be positive")
        return self.score_requests / aggregate_scores_per_second / 3600 / (1 - self.reserve_fraction)

    def report(self, aggregate_scores_per_second=None):
        result = dict(score_requests_upper_bound=self.score_requests,
                      training_score_requests_upper_bound=self.training_score_requests,
                      validation_score_requests_upper_bound=self.validation_score_requests,
                      probe_steps=self.probe_steps, wall_hours=self.wall_hours,
                      reserve_fraction=self.reserve_fraction,
                      required_aggregate_scores_per_second=self.required_aggregate_rate,
                      throughput_is_measured=False)
        if aggregate_scores_per_second is not None:
            result["assumed_aggregate_scores_per_second"] = aggregate_scores_per_second
            result["scenario_hours"] = self.estimated_hours(aggregate_scores_per_second)
        return result
