from dataclasses import dataclass

import torch

from .core import BinaryTransformer, adapt, score, simple_baseline, summarize


@dataclass(frozen=True)
class CoreEvaluation:
    initial_test: dict[str, dict[str, float]]
    support: dict[str, dict[str, float]]
    test: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ValidationTrial:
    core_seed: int
    transformer: CoreEvaluation
    identity: CoreEvaluation


@dataclass(frozen=True)
class ValidationResult:
    trials: list[ValidationTrial]
    baselines: dict[str, dict[str, dict[str, float]]]


def check_validation_inputs(config, support, test, seeds):
    if not support or not test or not seeds:
        raise ValueError("Validation requires nonempty support, test, and core seeds")
    if set(support) & set(test):
        raise ValueError("Validation support and test must be disjoint")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Validation core seeds must be distinct")
    if any(len(s) != config.n or set(s) - {"0", "1"} for s in (*support, *test)):
        raise ValueError(
            f"Validation sequences must contain exactly {config.n} binary digits"
        )


def sample_support_stream(support, budget):
    """Replay adaptation's minibatch RNG so the count controls see the same data."""
    generator = torch.Generator().manual_seed(budget.sampling_seed)
    observed = []
    for _ in range(budget.steps):
        indices = torch.randint(len(support), (budget.batch_size,), generator=generator)
        observed.extend(support[i] for i in indices.tolist())
    return observed


def evaluate_baselines(support, test, budget):
    observed = sample_support_stream(support, budget)
    return {
        "unigram": simple_baseline(observed, test, order=0),
        "bigram": simple_baseline(observed, test, order=1),
    }


def evaluate_trial(config, budget, support, test, seed, device, progress):
    transformer = evaluate_core(
        config, budget, support, test, seed, "transformer", device, progress
    )
    identity = evaluate_core(
        config, budget, support, test, seed, "identity", device, progress
    )
    return ValidationTrial(seed, transformer=transformer, identity=identity)


def evaluate_core(config, budget, support, test, seed, core, device, progress):
    initial = BinaryTransformer(config, seed, core).to(device)
    initial_metrics = score(initial, test)
    trained = adapt(initial, support, budget)
    support_metrics = score(trained, support)
    test_metrics = score(trained, test)
    summary = summarize(test_metrics)
    progress(
        f"Seed {seed}, {core}: test loss {summary['mean_loss']:.4f}, accuracy {summary['mean_accuracy']:.3f}"
    )
    return CoreEvaluation(initial_metrics, support_metrics, test_metrics)
