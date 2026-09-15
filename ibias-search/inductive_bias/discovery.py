from dataclasses import dataclass
import random

from .core import adapt, score, summarize
from .sequences import edit_distance, propose


@dataclass(frozen=True)
class SearchConfig:
    rounds: int = 4
    proposals: int = 24
    shortlist: int = 4
    low_transfer: int = 1
    beam_width: int = 2
    restarts: int = 3
    seed_mutations: int = 1
    novelty_weight: float = 0.05
    seed: int = 0

    def __post_init__(self):
        for name in (
            "rounds",
            "proposals",
            "shortlist",
            "low_transfer",
            "seed_mutations",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.beam_width < 1 or self.restarts < 1:
            raise ValueError("beam_width and restarts must be positive")


@dataclass(frozen=True)
class Candidate:
    sequence: str
    transfer_gain: float
    score: float


@dataclass(frozen=True)
class Family:
    restart: int
    members: list[str]
    metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DiscoveryResult:
    families: list[Family]
    discovery_sequences: list[str]
    adaptation_evaluations: int


@dataclass(frozen=True)
class Branch:
    family: Family
    population: list[str]


def rank_candidates(initial, trained, members, candidates, config):
    before, after = score(initial, candidates), score(trained, candidates)
    rows = []
    for sequence in candidates:
        gain = before[sequence]["loss"] - after[sequence]["loss"]
        novelty = min(edit_distance(sequence, s) / len(s) for s in members)
        rows.append(Candidate(sequence, gain, gain + config.novelty_weight * novelty))
    return sorted(rows, key=lambda row: (-row.score, row.sequence))


def shortlist(ranked, config):
    count = min(config.shortlist, len(ranked))
    low_count = min(config.low_transfer, max(0, count - 1))
    selected = ranked[: count - low_count]
    remaining = ranked[count - low_count :]
    return (
        selected
        + sorted(remaining, key=lambda row: (row.transfer_gain, row.sequence))[
            :low_count
        ]
    )


def check_seed(n, seed_sequence):
    if n < 2:
        raise ValueError("Discovery requires sequences of at least two bits")
    if seed_sequence is None:
        return
    if len(seed_sequence) != n or set(seed_sequence) - {"0", "1"}:
        raise ValueError(f"seed_sequence must contain exactly {n} binary digits")


def family_priority(family):
    return -len(family.members), summarize(family.metrics)["mean_loss"]


def select_branches(branches, beam_width):
    ranked = sorted(
        branches,
        key=lambda branch: (
            *family_priority(branch.family),
            sorted(branch.family.members),
        ),
    )
    unique = {}
    for branch in ranked:
        members = tuple(sorted(branch.family.members))
        unique.setdefault(members, branch)
    return list(unique.values())[:beam_width]


class DiscoverySearch:
    """One search run, with a shared RNG, discovery archive, and fit counter."""

    def __init__(self, initial, budget, thresholds, search, progress, trace):
        self.initial = initial
        self.budget = budget
        self.thresholds = thresholds
        self.search = search
        self.progress = progress
        self.trace = trace
        self.rng = random.Random(search.seed)
        self.seen = set()
        self.evaluations = 0

    def run_restart(self, restart, seed_sequence=None):
        family = self.seed_family(restart, seed_sequence)
        if not self.thresholds.accepts(family.metrics):
            return []

        branches = [Branch(family, [])]
        for iteration in range(self.search.rounds):
            branches = self.advance_beam(branches, iteration)
        return [branch.family for branch in branches]

    def seed_family(self, restart, seed_sequence):
        n = self.initial.config.n
        seed = seed_sequence
        if seed is None:
            seed = f"{self.rng.getrandbits(n):0{n}b}"
        members = [seed] + propose([seed], [], self.search.seed_mutations, self.rng)
        self.seen.update(members)
        family = self.fit_family(restart, members)
        self.record_trace("seed", family)
        status = (
            "accepted"
            if self.thresholds.accepts(family.metrics)
            else "failed thresholds"
        )
        self.progress(
            f"Restart {restart + 1}/{self.search.restarts}: seed group {status}"
        )
        return family

    def advance_beam(self, branches, iteration):
        expanded = []
        for branch in branches:
            expanded.extend(self.expand_branch(branch, iteration))
        retained = select_branches(expanded, self.search.beam_width)
        sizes = [len(branch.family.members) for branch in retained]
        self.progress(
            f"  Round {iteration + 1}/{self.search.rounds}: retained sizes {sizes}"
        )
        return retained

    def expand_branch(self, branch, iteration):
        family = branch.family
        candidates = propose(
            family.members, branch.population, self.search.proposals, self.rng
        )
        self.seen.update(candidates)
        ranked = self.rank_proposals(family, candidates)
        population = [
            candidate.sequence for candidate in ranked[: self.search.proposals // 2]
        ]
        additions = self.verify_additions(family, shortlist(ranked, self.search))
        self.record_trace(
            "growth",
            family,
            iteration=iteration,
            candidates=ranked,
            additions=additions,
        )

        # Retain the parent so an unsuccessful batch does not stop later rounds.
        expanded = [Branch(family, population)]
        for addition in additions:
            if self.thresholds.accepts(addition.metrics):
                expanded.append(Branch(addition, population))
        return expanded

    def rank_proposals(self, family, candidates):
        if not candidates:
            return []
        trained = self.fit(family.members)
        return rank_candidates(
            self.initial, trained, family.members, candidates, self.search
        )

    def verify_additions(self, family, candidates):
        additions = []
        for candidate in candidates:
            members = family.members + [candidate.sequence]
            additions.append(self.fit_family(family.restart, members))
        return additions

    def fit_family(self, restart, members):
        trained = self.fit(members)
        return Family(restart, members, score(trained, members))

    def fit(self, members):
        trained = adapt(self.initial, members, self.budget)
        self.evaluations += 1
        return trained

    def record_trace(self, stage, family, **details):
        if self.trace is not None:
            self.trace(stage, family, **details)
