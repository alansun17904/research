from .discovery import (
    DiscoveryResult,
    DiscoverySearch,
    SearchConfig,
    check_seed,
    family_priority,
)
from .validation import (
    ValidationResult,
    check_validation_inputs,
    evaluate_baselines,
    evaluate_trial,
)


def discover(
    initial,
    budget,
    thresholds,
    search: SearchConfig,
    seed_sequence=None,
    progress=lambda _: None,
    trace=None,
) -> DiscoveryResult:
    check_seed(initial.config.n, seed_sequence)
    run = DiscoverySearch(initial, budget, thresholds, search, progress, trace)
    families = []
    for restart in range(search.restarts):
        seed = seed_sequence if restart == 0 else None
        families.extend(run.run_restart(restart, seed))
    families.sort(key=family_priority)
    return DiscoveryResult(families, sorted(run.seen), run.evaluations)


def validate(
    config, budget, support, test, seeds, device="cpu", progress=lambda _: None
) -> ValidationResult:
    seeds = list(seeds)
    check_validation_inputs(config, support, test, seeds)
    baselines = evaluate_baselines(support, test, budget)
    trials = []
    for seed in seeds:
        trial = evaluate_trial(config, budget, support, test, seed, device, progress)
        trials.append(trial)
    return ValidationResult(trials, baselines)
