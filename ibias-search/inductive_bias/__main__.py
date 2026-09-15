from dataclasses import asdict
from pathlib import Path
import json
import statistics

import click
import torch

from .core import (
    POSITION_ENCODING,
    BinaryTransformer,
    Budget,
    ModelConfig,
    Thresholds,
    summarize,
)
from .pipeline import SearchConfig, discover, validate
from .sequences import Rule, hypotheses


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def protocol(config, budget, thresholds):
    return {
        "model": asdict(config),
        "budget": asdict(budget),
        "thresholds": asdict(thresholds),
        "architecture": "decoder_only_transformer",
        "task": "next_bit_prediction",
        "scored_positions": list(range(1, config.n)),
        "loss_units": "mean_natural_log_cross_entropy_per_scored_token",
        "trainable_parameters": ["embedding.weight", "unembedding.weight"],
        "trainable_parameter_count": 4 * config.width,
        "positions": POSITION_ENCODING,
        "identity_positions": "none",
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "training_examples_per_evaluation": budget.steps * budget.batch_size,
        "scored_training_tokens_per_evaluation": budget.steps
        * budget.batch_size
        * (config.n - 1),
        "training_sampling": "uniform_with_replacement_from_support",
        "reset": "same_initial_model_optimizer_and_minibatch_rng_for_every_evaluation",
    }


def evaluation_report(evaluation, thresholds):
    summary = summarize(evaluation.test)
    return {
        "support": evaluation.support,
        "test": evaluation.test,
        "summary": summary,
        "initial_test": evaluation.initial_test,
        "mean_transfer_gain": summarize(evaluation.initial_test)["mean_loss"]
        - summary["mean_loss"],
        "support_passes_thresholds": thresholds.accepts(evaluation.support),
        "test_passes_thresholds": thresholds.accepts(evaluation.test),
        "failure_cases": [
            {"sequence": sequence, **metrics}
            for sequence, metrics in sorted(
                evaluation.test.items(), key=lambda item: -item[1]["loss"]
            )
            if not thresholds.accepts({sequence: metrics})
        ][:10],
    }


def validation_report(result, thresholds):
    baselines = {
        name: {"test": metrics, "summary": summarize(metrics)}
        for name, metrics in result.baselines.items()
    }
    trials = []
    for trial in result.trials:
        row = {
            "core_seed": trial.core_seed,
            "transformer": evaluation_report(trial.transformer, thresholds),
            "identity": evaluation_report(trial.identity, thresholds),
        }
        transformer_loss = row["transformer"]["summary"]["mean_loss"]
        row["loss_advantage_over_controls"] = {
            "identity": row["identity"]["summary"]["mean_loss"] - transformer_loss,
            **{
                name: baseline["summary"]["mean_loss"] - transformer_loss
                for name, baseline in baselines.items()
            },
        }
        trials.append(row)
    aggregate = {}
    for core in ("transformer", "identity"):
        aggregate[core] = {}
        for metric in ("mean_loss", "mean_accuracy", "worst_loss", "worst_accuracy"):
            values = [trial[core]["summary"][metric] for trial in trials]
            aggregate[core][metric] = {
                "mean": statistics.mean(values),
                "std_across_seeds": statistics.pstdev(values),
            }
        aggregate[core]["all_seeds_pass_test_thresholds"] = all(
            trial[core]["test_passes_thresholds"] for trial in trials
        )
    return {"trials": trials, "baselines": baselines, "aggregate": aggregate}


@click.group()
def cli():
    """Discover candidate binary families, then validate an explicitly frozen rule.

    Run experiments through SLURM with inductive_bias.sbatch.
    """


@cli.command("discover")
@click.option("--n", default=16, show_default=True)
@click.option("--width", default=16, show_default=True)
@click.option("--heads", default=2, show_default=True)
@click.option("--layers", default=1, show_default=True)
@click.option("--ff-width", default=32, show_default=True)
@click.option("--steps", default=100, show_default=True)
@click.option("--batch-size", default=16, show_default=True)
@click.option("--learning-rate", default=0.2, show_default=True)
@click.option(
    "--max-loss",
    default=0.65,
    show_default=True,
    help="Required for EVERY sequence, in nats per target bit.",
)
@click.option("--min-accuracy", default=0.6, show_default=True)
@click.option("--rounds", default=4, show_default=True)
@click.option("--proposals", default=24, show_default=True)
@click.option("--shortlist", "shortlist_size", default=4, show_default=True)
@click.option(
    "--low-transfer",
    default=1,
    show_default=True,
    help="Reserve this many shortlist slots for lowest-transfer proposals. Increasing this number results in more exploration, decreasing it results in more exploitation.",
)
@click.option("--beam-width", default=2, show_default=True)
@click.option("--restarts", default=3, show_default=True)
@click.option("--seed-mutations", default=1, show_default=True)
@click.option("--novelty-weight", default=0.05, show_default=True)
@click.option("--seed", default=42, show_default=True, help="Proposal RNG seed.")
@click.option(
    "--core-seed",
    default=0,
    show_default=True,
    help="Frozen core and initial E/U seed.",
)
@click.option(
    "--sampling-seed", default=0, show_default=True, help="Reset minibatch RNG seed."
)
@click.option(
    "--seed-sequence",
    help="Optional binary seed for the first restart; must have length n.",
)
@click.option("--device", default="cuda", show_default=True)
@click.option(
    "--debug",
    is_flag=True,
    help="Include proposal and verification traces in the report.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="results/binary-discovery.json",
    show_default=True,
)
def discover_command(
    n,
    width,
    heads,
    layers,
    ff_width,
    steps,
    batch_size,
    learning_rate,
    max_loss,
    min_accuracy,
    rounds,
    proposals,
    shortlist_size,
    low_transfer,
    beam_width,
    restarts,
    seed_mutations,
    novelty_weight,
    seed,
    core_seed,
    sampling_seed,
    seed_sequence,
    device,
    debug,
    output,
):
    """Grow candidate families using shared E/U and a frozen decoder-only transformer."""
    config = ModelConfig(n, width, heads, layers, ff_width)
    budget = Budget(steps, batch_size, learning_rate, sampling_seed)
    thresholds = Thresholds(max_loss, min_accuracy)
    search = SearchConfig(
        rounds=rounds,
        proposals=proposals,
        shortlist=shortlist_size,
        low_transfer=low_transfer,
        beam_width=beam_width,
        restarts=restarts,
        seed_mutations=seed_mutations,
        novelty_weight=novelty_weight,
        seed=seed,
    )
    initial = BinaryTransformer(config, core_seed).to(device)
    trace = []

    def record_trace(stage, family, iteration=None, candidates=(), additions=()):
        trace.append(
            {
                "stage": stage,
                "restart": family.restart,
                "iteration": iteration,
                "members": family.members,
                "metrics": family.metrics,
                "accepted": thresholds.accepts(family.metrics),
                "candidates": [asdict(candidate) for candidate in candidates],
                "verified_additions": [
                    {
                        "sequence": addition.members[-1],
                        "accepted": thresholds.accepts(addition.metrics),
                        "metrics": addition.metrics,
                    }
                    for addition in additions
                ],
            }
        )

    result = discover(
        initial,
        budget,
        thresholds,
        search,
        seed_sequence=seed_sequence,
        progress=click.echo,
        trace=record_trace if debug else None,
    )
    report = {
        "schema_version": 1,
        "kind": "discovery",
        "status": "unvalidated",
        **protocol(config, budget, thresholds),
        "core_seed": core_seed,
        "search": asdict(search),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "total_parameter_count": sum(p.numel() for p in initial.parameters()),
        "adaptation_evaluations": result.adaptation_evaluations,
        "total_search_training_examples": result.adaptation_evaluations
        * budget.steps
        * budget.batch_size,
        "families": [
            {
                "status": "candidate_family",
                "restart": family.restart,
                "members": family.members,
                "metrics": family.metrics,
                "summary": summarize(family.metrics),
                "hypotheses": [
                    {
                        "description": rule.describe(),
                        "rule": asdict(rule),
                        "status": "unvalidated_hypothesis",
                    }
                    for rule in hypotheses(family.members)
                ],
            }
            for family in result.families
        ],
        "trace": trace,
        "discovery_sequences": result.discovery_sequences,
        "explanation_method": "descriptive_rule_templates_for_explicit_selection; no_LLM_used",
        "interpretation": "All seeds, proposals and verified additions are discovery data. Template explanations and training success are not held-out validation.",
    }
    write_json(output, report)
    click.echo(
        f"Saved {len(report['families'])} candidate families to {output}; held-out validation is still required."
    )


@cli.command("validate")
@click.option(
    "--discovery", "discovery_path", type=click.Path(path_type=Path), required=True
)
@click.option(
    "--rule-file",
    type=click.Path(path_type=Path),
    help="JSON Rule object, as found in a discovery hypothesis's 'rule' field.",
)
@click.option(
    "--period", type=int, help="Repeat a prefix of this length, truncating to n."
)
@click.option("--mask", help="Required bits and ? wildcards; length must equal n.")
@click.option(
    "--ones", type=(int, int), help="Inclusive minimum and maximum number of ones."
)
@click.option("--max-transitions", type=int, help="Maximum adjacent bit changes.")
@click.option(
    "--support-size", type=click.IntRange(min=1), default=16, show_default=True
)
@click.option("--test-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option(
    "--core-seed",
    "core_seeds",
    multiple=True,
    default=(100, 101, 102),
    show_default=True,
    help="Repeat for fresh initializations.",
)
@click.option("--data-seed", default=1000, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="results/binary-validation.json",
    show_default=True,
)
def validate_command(
    discovery_path,
    rule_file,
    period,
    mask,
    ones,
    max_transitions,
    support_size,
    test_size,
    core_seeds,
    data_seed,
    device,
    output,
):
    """Freeze a selected rule, then test fresh examples across fresh random cores.

    Rule flags form a conjunction. For example, --ones 0 4 selects sequences
    with at most four ones. Support/test examples exclude ALL discovery data.
    """
    frozen_path = output.with_suffix(".protocol.json")
    if output.exists() or frozen_path.exists() or frozen_path == output:
        raise click.ClickException(
            "Use a new output path for the results and frozen protocol."
        )
    discovery = json.loads(discovery_path.read_text())
    config = ModelConfig(**discovery["model"])
    budget = Budget(**discovery["budget"])
    thresholds = Thresholds(**discovery["thresholds"])
    rule = (
        Rule(**json.loads(rule_file.read_text()))
        if rule_file
        else Rule(
            n=config.n,
            period=period,
            mask=mask,
            min_ones=ones[0] if ones else 0,
            max_ones=ones[1] if ones else None,
            max_transitions=max_transitions,
        )
    )
    if rule.n != config.n:
        raise click.ClickException("The rule length must match the discovery model.")
    if len(set(core_seeds)) != len(core_seeds) or discovery["core_seed"] in core_seeds:
        raise click.ClickException(
            "Validation requires distinct core seeds unused in discovery."
        )
    excluded = set(discovery["discovery_sequences"])
    fresh = rule.draw(support_size + test_size, excluded, data_seed)
    support, test = fresh[:support_size], fresh[support_size:]
    frozen = {
        "schema_version": 1,
        "kind": "frozen_validation_protocol",
        **protocol(config, budget, thresholds),
        "rule": asdict(rule),
        "description": rule.describe(),
        "core_seeds": list(core_seeds),
        "data_seed": data_seed,
        "support": support,
        "test": test,
        "excluded_discovery_count": len(excluded),
        "sampling": "uniform_without_replacement_within_rule_excluding_all_discovery_sequences",
        "controls": ["identity_core", "unigram", "bigram"],
        "device": str(device),
        "torch_version": str(torch.__version__),
        "baseline_budget": "Analytic Laplace-smoothed counts fit on exactly the same sampled support-example stream as adaptation.",
        "discovery_path": str(discovery_path.resolve()),
    }
    # Persist the rule, thresholds, seeds and complete split before any fit
    # or test evaluation. No model results influence sample selection.
    write_json(frozen_path, frozen)
    result = validate(
        config, budget, support, test, core_seeds, device, progress=click.echo
    )
    report = {
        **frozen,
        "kind": "validation",
        "status": "evaluated_frozen_rule",
        **validation_report(result, thresholds),
        "interpretation": "Held-out performance for one frozen rule and split across fresh random initializations. A successful family alone does not establish a transformer-specific bias; inspect control comparisons. Reusing this test split to select rules invalidates it as held-out evidence.",
    }
    write_json(output, report)
    click.echo(f"Saved held-out results and control comparisons to {output}")


if __name__ == "__main__":
    cli()
