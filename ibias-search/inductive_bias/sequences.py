from dataclasses import dataclass
import random


def edit_distance(a, b):
    # dp[0, j] = j
    # dp[i, 0] = i
    # dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1,j-1] + (a[i] != b[j]))
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        next_row = [i]
        for j, right in enumerate(b, 1):
            next_row.append(
                min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (left != right))
            )
        row = next_row
    return row[-1]


def mutate(sequence, rng):
    bits = list(sequence)
    n = len(bits)
    if n == 1:
        return str(1 - int(sequence))
    operation = rng.randrange(4)
    if operation == 0:  # flip a random bit
        i = rng.randrange(n)
        bits[i] = str(1 - int(bits[i]))
    elif operation == 1:  # swap two random blocks
        length = rng.randint(1, n - 1)
        source, target = rng.randrange(n - length + 1), rng.randrange(n - length + 1)
        bits[target : target + length] = bits[source : source + length]
    elif operation == 2:  # flip a random subset of bits
        for i in rng.sample(range(n), rng.randint(2, n)):
            bits[i] = str(1 - int(bits[i]))
    else:  # generate a random new sequence
        bits = list(f"{rng.getrandbits(n):0{n}b}")
    return "".join(bits)


def propose(members, population, count, rng):
    if count < 0:
        raise ValueError("Proposal count must be nonnegative")
    if count == 0:
        return []
    n = len(members[0])
    excluded = set(members)
    count = min(count, (1 << n) - len(excluded))
    proposals = list(dict.fromkeys(s for s in population if s not in excluded))[
        : count // 2
    ]
    parents = list(dict.fromkeys(members + population))
    seen = excluded | set(proposals)
    # Mutation can keep producing duplicates, especially near domain exhaustion.
    # Bound retries, then fill from a cyclic scan of the finite binary domain.
    for _ in range(max(100, count * 20)):
        if len(proposals) == count:
            return proposals
        candidate = mutate(rng.choice(parents), rng)
        if candidate not in seen:
            proposals.append(candidate)
            seen.add(candidate)
    start = rng.randrange(1 << n)
    for offset in range(1 << n):
        if len(proposals) == count:
            break
        candidate = f"{(start + offset) % (1 << n):0{n}b}"
        if candidate not in seen:
            proposals.append(candidate)
    return proposals


@dataclass(frozen=True)
class Rule:
    """An executable conjunction over fixed-length binary sequences."""

    n: int
    period: int | None = None  # repeat a prefix of length `period`
    mask: str | None = None  # `?` requires a bit
    min_ones: int = 0  # contain at least `min_ones` ones
    max_ones: int | None = None  # contain at most `max_ones` ones
    max_transitions: int | None = None  # change bit at most `max_transitions` times

    @property
    def ones_upper(self):
        return self.n if self.max_ones is None else self.max_ones

    def fixed_groups(self):
        # extracts the required FIXED bits in the sequence
        width = self.period or self.n
        return {
            i % width: bit
            for i, bit in enumerate(self.mask or "?" * self.n)
            if bit != "?"
        }

    def contains(self, sequence):
        # Check if a given sequence satisfies all of the rules
        if len(sequence) != self.n or set(sequence) - {"0", "1"}:
            return False
        if self.period and any(
            sequence[i] != sequence[i % self.period] for i in range(self.n)
        ):
            return False
        if self.mask and any(
            m != "?" and m != bit for m, bit in zip(self.mask, sequence)
        ):
            return False
        if not self.min_ones <= sequence.count("1") <= self.ones_upper:
            return False
        return (
            self.max_transitions is None
            or sum(a != b for a, b in zip(sequence, sequence[1:]))
            <= self.max_transitions
        )

    def draw(self, count, excluded, seed):
        """Uniform rejection sampling without replacement, independent of loss."""
        if count < 0:
            raise ValueError("Sample count must be nonnegative")
        if count == 0:
            return []
        rng = random.Random(seed)
        width, fixed = self.period or self.n, self.fixed_groups()
        free = [i for i in range(width) if i not in fixed]
        samples, seen = [], set(excluded)
        for _ in range(max(10000, count * 1000)):
            assignment = dict(fixed)
            assignment.update((i, str(rng.randrange(2))) for i in free)
            sequence = "".join(assignment[i % width] for i in range(self.n))
            if sequence not in seen and self.contains(sequence):
                samples.append(sequence)
                seen.add(sequence)
                if len(samples) == count:
                    return samples
        raise RuntimeError(
            f"Sampling exhausted after finding {len(samples)}/{count} fresh examples."
        )

    def describe(self):
        conditions = []
        if self.period:
            conditions.append(
                f"repeat a {self.period}-bit prefix, truncated to length {self.n}"
            )
        if self.mask:
            conditions.append(f"match mask {self.mask} (? means either bit)")
        if self.min_ones != 0 or self.ones_upper != self.n:
            conditions.append(f"contain {self.min_ones} to {self.ones_upper} ones")
        if self.max_transitions is not None:
            conditions.append(f"change bit at most {self.max_transitions} times")
        return (
            f"Length-{self.n} binary sequences"
            + (" that " + " and ".join(conditions) if conditions else "")
            + "."
        )


def hypotheses(members):
    """Generates a list of rules that are consistent with all members of the family."""
    n = len(members[0])
    rules = []
    for period in range(1, n):
        rule = Rule(n, period=period)
        if all(rule.contains(s) for s in members):
            rules.append(rule)
            break
    mask = "".join(
        column[0] if len(set(column)) == 1 else "?" for column in zip(*members)
    )
    if mask != "?" * n:
        rules.append(Rule(n, mask=mask))
    rules.append(
        Rule(
            n,
            min_ones=min(s.count("1") for s in members),
            max_ones=max(s.count("1") for s in members),
        )
    )
    rules.append(
        Rule(
            n,
            max_transitions=max(sum(a != b for a, b in zip(s, s[1:])) for s in members),
        )
    )
    return rules
