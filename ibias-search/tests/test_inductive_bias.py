from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner
import torch

from inductive_bias.__main__ import cli, protocol, validation_report
from inductive_bias.core import (
    POSITION_ENCODING,
    BinaryTransformer,
    Budget,
    ModelConfig,
    RotaryEmbedding,
    Thresholds,
    adapt,
    score,
    simple_baseline,
)
from inductive_bias.discovery import SearchConfig, rank_candidates, shortlist
from inductive_bias.pipeline import discover, validate
from inductive_bias.validation import CoreEvaluation, ValidationResult, ValidationTrial
from inductive_bias.sequences import Rule, edit_distance, mutate, propose


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = ModelConfig(n=4, width=4, heads=1, layers=1, ff_width=8)

    def test_only_embedding_and_unembedding_train_and_every_fit_resets(self):
        initial = BinaryTransformer(self.config, seed=7)
        saved = {key: value.clone() for key, value in initial.state_dict().items()}
        self.assertEqual(
            [name for name, p in initial.named_parameters() if p.requires_grad],
            ["embedding.weight", "unembedding.weight"],
        )
        budget = Budget(steps=2, batch_size=3, learning_rate=0.1)
        first = adapt(initial, ["0000", "1111"], budget)
        second = adapt(initial, ["0000", "1111"], budget)
        for name, value in initial.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)
            self.assertTrue(
                torch.equal(first.state_dict()[name], second.state_dict()[name]), name
            )
            if name in {"embedding.weight", "unembedding.weight"}:
                self.assertFalse(
                    torch.equal(first.state_dict()[name], saved[name]), name
                )
            else:
                self.assertTrue(
                    torch.equal(first.state_dict()[name], saved[name]), name
                )

    def test_causality_blocks_future_bits(self):
        # RoPE and causal attention must work in both training and evaluation.
        for heads in (1, 2):
            with self.subTest(heads=heads):
                config = ModelConfig(n=4, width=4, heads=heads, ff_width=8)
                model = BinaryTransformer(config, seed=1).eval()
                with torch.no_grad():
                    left = model(torch.tensor([[0, 1, 0]]))
                    right = model(torch.tensor([[0, 1, 1]]))
                    training_path = model.train()(torch.tensor([[0, 1, 0]]))
                torch.testing.assert_close(left[:, :2], right[:, :2], rtol=0, atol=1e-6)
                torch.testing.assert_close(left, training_path, rtol=1e-5, atol=1e-6)

    def test_shifted_loss_scores_all_and_only_n_minus_one_targets(self):
        model = BinaryTransformer(self.config, seed=1)
        with torch.no_grad():
            model.unembedding.weight.zero_()
        metrics = score(model, ["0111", "1000"])
        self.assertAlmostEqual(metrics["0111"]["loss"], math.log(2), places=6)
        self.assertEqual(metrics["0111"]["accuracy"], 0)
        self.assertEqual(metrics["1000"]["accuracy"], 1)

    def test_identity_control_has_identical_initial_adapters_and_no_position_input(
        self,
    ):
        state = torch.random.get_rng_state().clone()
        transformer = BinaryTransformer(self.config, seed=9)
        identity = BinaryTransformer(self.config, seed=9, core="identity")
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(
            torch.equal(transformer.embedding.weight, identity.embedding.weight)
        )
        self.assertTrue(
            torch.equal(transformer.unembedding.weight, identity.unembedding.weight)
        )
        tokens = torch.tensor([[0, 1, 0]])
        torch.testing.assert_close(
            identity(tokens), identity.unembedding(identity.embedding(tokens))
        )
        self.assertEqual(len(identity.blocks), 0)

    def test_rope_matches_known_rotations_and_preserves_norm(self):
        rope = RotaryEmbedding(head_dim=4, max_length=3)
        x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]] * 3]])
        expected = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0, 1.0, 0.0],
                        [math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01)],
                        [math.cos(2), math.sin(2), math.cos(0.02), math.sin(0.02)],
                    ]
                ]
            ]
        )
        rotated = rope(x)
        torch.testing.assert_close(rotated, expected)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
        self.assertEqual(list(rope.parameters()), [])

    def test_rope_attention_scores_depend_on_relative_positions(self):
        rope = RotaryEmbedding(head_dim=4, max_length=6)
        q = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 3, 6, 4)
        k = torch.tensor([-2.0, 3.0, 1.0, -4.0]).expand(2, 3, 6, 4)
        scores = rope(q) @ rope(k).transpose(-1, -2)
        torch.testing.assert_close(scores[..., :3, :3], scores[..., 2:5, 2:5])
        self.assertFalse(torch.allclose(scores[..., 0, 0], scores[..., 0, 1]))

    def test_decoder_uses_rope(self):
        model = BinaryTransformer(self.config, seed=5).eval()
        tokens = torch.tensor([[0, 1, 0], [1, 0, 1]])
        with torch.no_grad():
            rotated = model(tokens)
            with patch.object(RotaryEmbedding, "forward", lambda self, x: x):
                unrotated = model(tokens)
        self.assertFalse(torch.allclose(rotated, unrotated, rtol=1e-5, atol=1e-6))

    def test_budget_does_not_grow_with_support(self):
        initial = BinaryTransformer(self.config, seed=3)
        budget = Budget(steps=2, batch_size=3)
        forward, shapes = BinaryTransformer.forward, []

        def record(model, tokens):
            shapes.append(tuple(tokens.shape))
            return forward(model, tokens)

        with patch.object(BinaryTransformer, "forward", record):
            adapt(initial, ["0000"], budget)
            adapt(initial, ["0000", "0001", "1111"], budget)
        self.assertEqual(shapes, [(3, 3)] * 4)


class SearchTests(unittest.TestCase):
    def test_edit_novelty_and_fixed_length_mutations(self):
        self.assertEqual(edit_distance("0101", "1010"), 2)
        rng = random.Random(1)
        for _ in range(100):
            candidate = mutate("01010101", rng)
            self.assertEqual(len(candidate), 8)
            self.assertFalse(set(candidate) - {"0", "1"})
        # A finite exhausted domain terminates, with no duplicates or seeds.
        self.assertEqual(sorted(propose(["00"], [], 10, rng)), ["01", "10", "11"])

    def test_transfer_score_sign_and_low_transfer_shortlist(self):
        config = SearchConfig(
            proposals=3, shortlist=2, low_transfer=1, novelty_weight=0.1
        )
        before = {
            s: {"loss": loss}
            for s, loss in [("1111", 1.0), ("0001", 0.9), ("0010", 0.4)]
        }
        after = {
            s: {"loss": loss}
            for s, loss in [("1111", 0.5), ("0001", 0.7), ("0010", 0.8)]
        }
        with patch("inductive_bias.discovery.score", side_effect=[before, after]):
            ranked = rank_candidates(None, None, ["0000"], list(before), config)
        self.assertEqual(ranked[0].sequence, "1111")
        self.assertAlmostEqual(ranked[0].score, 0.5 + 0.1)
        self.assertEqual(
            [row.sequence for row in shortlist(ranked, config)], ["1111", "0010"]
        )

    def test_proposals_finish_when_mutations_never_produce_a_new_sequence(self):
        with patch("inductive_bias.sequences.mutate", return_value="00"):
            proposals = propose(["00"], [], 10, random.Random(1))
        self.assertEqual(sorted(proposals), ["01", "10", "11"])
        self.assertEqual(propose(["00", "01", "10", "11"], [], 2, random.Random(1)), [])

    def test_verification_rejects_a_harmed_member_and_archives_rejections(self):
        config = ModelConfig(n=4, width=4, heads=1, ff_width=8)
        initial = BinaryTransformer(config, seed=0)
        trace = []

        def record_trace(stage, family, **kwargs):
            trace.append((stage, family, kwargs))

        def fake_adapt(initial, members, budget):
            return members

        def fake_score(model, sequences):
            metrics = {s: {"loss": 0.5, "accuracy": 0.8} for s in sequences}
            if isinstance(model, BinaryTransformer):
                return {s: {"loss": 0.8, "accuracy": 0.5} for s in sequences}
            if "0010" in model:
                metrics["0000"] = {"loss": 0.75, "accuracy": 0.8}
            return metrics

        with patch(
            "inductive_bias.discovery.adapt", side_effect=fake_adapt
        ) as mocked_adapt, patch(
            "inductive_bias.discovery.score", side_effect=fake_score
        ), patch(
            "inductive_bias.discovery.propose",
            side_effect=[[], ["0001", "0010"], [], []],
        ):
            report = discover(
                initial,
                Budget(steps=1),
                Thresholds(),
                SearchConfig(
                    rounds=2,
                    proposals=2,
                    shortlist=2,
                    low_transfer=1,
                    restarts=1,
                    seed_mutations=0,
                ),
                seed_sequence="0000",
                trace=record_trace,
            )
        # The bad enlarged group's average loss is .625 (below .65), but its
        # original member fails; checking only the average would accept it.
        self.assertEqual(report.families[0].members, ["0000", "0001"])
        self.assertTrue(all("0010" not in family.members for family in report.families))
        self.assertEqual(report.discovery_sequences, ["0000", "0001", "0010"])
        self.assertEqual(report.adaptation_evaluations, mocked_adapt.call_count)
        self.assertEqual(mocked_adapt.call_count, 4)
        self.assertEqual(
            len({id(call.args[0]) for call in mocked_adapt.call_args_list}), 1
        )
        self.assertEqual(
            [stage for stage, _, _ in trace], ["seed", "growth", "growth", "growth"]
        )
        for _, family, _ in trace:
            self.assertEqual(set(family.metrics), set(family.members))
        attempts = trace[1][2]["additions"]
        self.assertEqual(
            [Thresholds().accepts(f.metrics) for f in attempts], [True, False]
        )
        self.assertEqual(trace[2][1].members, ["0000", "0001"])
        self.assertEqual(trace[2][2]["iteration"], 1)

    def test_bad_seed_is_rejected_before_adaptation(self):
        initial = BinaryTransformer(ModelConfig(n=4, width=4, heads=1), seed=0)
        with patch("inductive_bias.discovery.adapt") as adapt_mock:
            for seed in ("001", "00000", "00x0", ""):
                with self.subTest(seed=seed), self.assertRaisesRegex(
                    ValueError, "binary digits"
                ):
                    discover(
                        initial,
                        Budget(),
                        Thresholds(),
                        SearchConfig(),
                        seed_sequence=seed,
                    )
        adapt_mock.assert_not_called()

    def test_exhausted_family_is_retained_without_redundant_fits(self):
        initial = BinaryTransformer(ModelConfig(n=2, width=4, heads=1), seed=0)
        with patch("inductive_bias.discovery.adapt") as fitted, patch(
            "inductive_bias.discovery.score",
            side_effect=lambda _, sequences: {
                s: {"loss": 0.5, "accuracy": 0.8} for s in sequences
            },
        ):
            result = discover(
                initial,
                Budget(),
                Thresholds(),
                SearchConfig(rounds=2, proposals=10, seed_mutations=10, restarts=1),
                seed_sequence="00",
            )
        self.assertEqual(fitted.call_count, 1)
        self.assertEqual(result.adaptation_evaluations, 1)
        self.assertEqual(sorted(result.families[0].members), ["00", "01", "10", "11"])

    def test_failed_restart_does_not_reset_rng_or_stop_later_restarts(self):
        initial = BinaryTransformer(ModelConfig(n=4, width=4, heads=1), seed=0)

        def fake_score(_, sequences):
            return {
                s: {"loss": 0.9 if s == "0000" else 0.5, "accuracy": 0.8}
                for s in sequences
            }

        with patch("inductive_bias.discovery.adapt") as fitted, patch(
            "inductive_bias.discovery.score", side_effect=fake_score
        ):
            result = discover(
                initial,
                Budget(),
                Thresholds(),
                SearchConfig(rounds=0, seed_mutations=0, restarts=3, seed=7),
                seed_sequence="0000",
            )
        self.assertEqual([family.restart for family in result.families], [1, 2])
        self.assertEqual(
            [family.members for family in result.families], [["0101"], ["1111"]]
        )
        self.assertEqual(result.discovery_sequences, ["0000", "0101", "1111"])
        self.assertEqual(result.adaptation_evaluations, 3)
        self.assertEqual(fitted.call_count, 3)

    def test_search_keeps_trying_after_a_rejected_batch(self):
        initial = BinaryTransformer(ModelConfig(n=4, width=4, heads=1), seed=0)

        def fake_score(model, sequences):
            loss = 0.8 if model is initial or "0001" in model else 0.5
            return {s: {"loss": loss, "accuracy": 0.8} for s in sequences}

        with patch(
            "inductive_bias.discovery.adapt", side_effect=lambda _, members, __: members
        ), patch("inductive_bias.discovery.score", side_effect=fake_score), patch(
            "inductive_bias.discovery.propose", side_effect=[[], ["0001"], ["0010"]]
        ):
            result = discover(
                initial,
                Budget(),
                Thresholds(),
                SearchConfig(
                    rounds=2, proposals=1, shortlist=1, beam_width=1, restarts=1
                ),
                seed_sequence="0000",
            )
        self.assertEqual(result.families[0].members, ["0000", "0010"])
        self.assertEqual(result.discovery_sequences, ["0000", "0001", "0010"])
        self.assertEqual(result.adaptation_evaluations, 5)

    def test_cli_discovery_report_and_optional_trace(self):
        def fake_score(model, sequences):
            if isinstance(model, BinaryTransformer):
                return {s: {"loss": 0.8, "accuracy": 0.5} for s in sequences}
            metrics = {s: {"loss": 0.5, "accuracy": 0.8} for s in sequences}
            if "0010" in model:
                metrics["0000"] = {"loss": 0.75, "accuracy": 0.8}
            return metrics

        runner, reports = CliRunner(), []
        with tempfile.TemporaryDirectory(prefix="rttt-discovery-tests-") as directory:
            for debug in (False, True):
                output = Path(directory) / f"discovery-{debug}.json"
                args = [
                    "discover",
                    "--n",
                    "4",
                    "--width",
                    "4",
                    "--heads",
                    "1",
                    "--rounds",
                    "2",
                    "--restarts",
                    "1",
                    "--seed-mutations",
                    "0",
                    "--proposals",
                    "2",
                    "--shortlist",
                    "2",
                    "--seed-sequence",
                    "0000",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                ] + (["--debug"] if debug else [])
                with patch(
                    "inductive_bias.discovery.adapt",
                    side_effect=lambda _, members, __: members,
                ), patch(
                    "inductive_bias.discovery.score", side_effect=fake_score
                ), patch(
                    "inductive_bias.discovery.propose",
                    side_effect=[[], ["0001", "0010"], [], []],
                ):
                    result = runner.invoke(cli, args)
                self.assertEqual(result.exit_code, 0, result.output)
                reports.append(json.loads(output.read_text()))
        without_debug, with_debug = reports
        self.assertEqual(without_debug.pop("trace"), [])
        trace = with_debug.pop("trace")
        self.assertEqual(with_debug, without_debug)
        self.assertEqual(with_debug["kind"], "discovery")
        self.assertEqual(with_debug["adaptation_evaluations"], 4)
        self.assertEqual(with_debug["total_search_training_examples"], 4 * 100 * 16)
        self.assertEqual(with_debug["discovery_sequences"], ["0000", "0001", "0010"])
        family = with_debug["families"][0]
        self.assertEqual(family["members"], ["0000", "0001"])
        self.assertTrue(
            all(
                Rule(**hypothesis["rule"]).contains(member)
                for hypothesis in family["hypotheses"]
                for member in family["members"]
            )
        )
        growth = trace[1]
        self.assertTrue(growth["accepted"])
        self.assertEqual(
            [a["accepted"] for a in growth["verified_additions"]], [True, False]
        )
        self.assertEqual(trace[2]["members"], ["0000", "0001"])
        self.assertEqual(set(trace[2]["metrics"]), {"0000", "0001"})


class ValidationTests(unittest.TestCase):
    def discovery(self):
        return {
            "schema_version": 1,
            "kind": "discovery",
            "core_seed": 0,
            "positions": POSITION_ENCODING,
            "model": asdict(ModelConfig(n=8, width=4, heads=1, ff_width=8)),
            "budget": asdict(Budget(steps=1, batch_size=2)),
            "thresholds": asdict(Thresholds()),
            "discovery_sequences": ["00000000", "11111111", "01010101"],
        }

    def test_sampling_is_reproducible_disjoint_and_obeys_frozen_rule(self):
        discovery = self.discovery()
        rule = Rule(8, mask="0???????", max_ones=4, max_transitions=4)
        fresh = rule.draw(12, set(discovery["discovery_sequences"]), 12)
        again = rule.draw(12, set(discovery["discovery_sequences"]), 12)
        self.assertEqual(fresh, again)
        support, test = set(fresh[:4]), set(fresh[4:])
        self.assertFalse(support & test)
        self.assertFalse((support | test) & set(discovery["discovery_sequences"]))
        self.assertEqual((len(support), len(test)), (4, 8))
        self.assertTrue(all(rule.contains(s) for s in support | test))
        metadata = protocol(
            ModelConfig(**discovery["model"]),
            Budget(**discovery["budget"]),
            Thresholds(),
        )
        self.assertEqual(metadata["budget"], discovery["budget"])
        self.assertEqual(metadata["positions"], POSITION_ENCODING)
        self.assertEqual(metadata["identity_positions"], "none")

    def test_periodic_rule(self):
        rule = Rule(8, period=3)
        all_members = rule.draw(8, set(), 1)
        self.assertEqual(len(set(all_members)), 8)
        self.assertTrue(all(rule.contains(s) for s in all_members))

    def test_simple_controls_use_shifted_tokens(self):
        test = ["10101010"]
        unigram = simple_baseline(["01010101", "10101010"], test, order=0)
        bigram = simple_baseline(["01010101", "10101010"], test, order=1)
        self.assertAlmostEqual(unigram[test[0]]["loss"], math.log(2))
        self.assertLess(bigram[test[0]]["loss"], unigram[test[0]]["loss"])
        self.assertEqual(bigram[test[0]]["accuracy"], 1)

    def test_validation_reporting_with_stubbed_model_fits(self):
        discovery = self.discovery()
        config, budget = ModelConfig(**discovery["model"]), Budget(
            **discovery["budget"]
        )
        fresh = Rule(8, max_ones=4).draw(7, set(discovery["discovery_sequences"]), 12)
        support, test = fresh[:3], fresh[3:]

        def fake_score(model, sequences):
            return {s: {"loss": 0.5, "accuracy": 0.8} for s in sequences}

        with patch(
            "inductive_bias.validation.adapt", side_effect=lambda initial, *_: initial
        ) as fitted, patch("inductive_bias.validation.score", side_effect=fake_score):
            result = validate(config, budget, support, test, [10, 11])
        report = validation_report(result, Thresholds())
        self.assertEqual(fitted.call_count, 4)
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(set(report["baselines"]), {"unigram", "bigram"})
        self.assertEqual(
            report["aggregate"]["transformer"]["mean_loss"],
            {"mean": 0.5, "std_across_seeds": 0.0},
        )
        json.dumps(report, allow_nan=False)

    def test_validation_rejects_invalid_inputs_before_fitting(self):
        config = ModelConfig(n=4, width=4, heads=1)
        cases = [
            ([], ["1111"], [10]),
            (["0000"], [], [10]),
            (["0000"], ["1111"], []),
            (["0000"], ["0000"], [10]),
            (["0000"], ["1111"], [10, 10]),
            (["000"], ["1111"], [10]),
            (["0000"], ["11x1"], [10]),
        ]
        with patch("inductive_bias.validation.adapt") as fitted:
            for support, test, seeds in cases:
                with self.subTest(
                    support=support, test=test, seeds=seeds
                ), self.assertRaises(ValueError):
                    validate(config, Budget(), support, test, seeds)
        fitted.assert_not_called()

    def test_validation_controls_use_the_same_sampled_support_stream(self):
        config = ModelConfig(n=4, width=4, heads=1)
        budget = Budget(steps=3, batch_size=2, sampling_seed=7)
        support, test = ["0000", "0001", "0010"], ["1111"]
        generator = torch.Generator().manual_seed(budget.sampling_seed)
        expected = [
            support[i]
            for _ in range(3)
            for i in torch.randint(3, (2,), generator=generator)
        ]
        with patch(
            "inductive_bias.validation.simple_baseline", wraps=simple_baseline
        ) as baseline, patch(
            "inductive_bias.validation.adapt", side_effect=lambda initial, *_: initial
        ), patch(
            "inductive_bias.validation.score",
            side_effect=lambda _, sequences: {
                s: {"loss": 0.5, "accuracy": 0.8} for s in sequences
            },
        ):
            validate(config, budget, support, test, [10])
        self.assertEqual(baseline.call_count, 2)
        for call in baseline.call_args_list:
            self.assertEqual(call.args[0], expected)
            self.assertEqual(call.args[1], test)

    def test_validation_keeps_results_paired_with_their_core_and_seed(self):
        config = ModelConfig(n=4, width=4, heads=1)
        support, test = ["0000"], ["1111"]
        progress = []

        def fake_model(config, seed, core):
            model = Mock()
            model.to.return_value = (seed, core, False)
            return model

        def fake_score(model, sequences):
            seed, core, trained = model
            loss = seed / 100 + (0.1 if core == "transformer" else 0.4)
            loss += 0 if trained else 0.2
            return {s: {"loss": loss, "accuracy": 0.8} for s in sequences}

        with patch(
            "inductive_bias.validation.BinaryTransformer", side_effect=fake_model
        ) as models, patch(
            "inductive_bias.validation.adapt",
            side_effect=lambda model, *_: (*model[:2], True),
        ), patch(
            "inductive_bias.validation.score", side_effect=fake_score
        ):
            result = validate(
                config,
                Budget(steps=1),
                support,
                test,
                iter([10, 11]),
                progress=progress.append,
            )
        self.assertEqual(
            [call.args[1:] for call in models.call_args_list],
            [
                (10, "transformer"),
                (10, "identity"),
                (11, "transformer"),
                (11, "identity"),
            ],
        )
        self.assertEqual([trial.core_seed for trial in result.trials], [10, 11])
        for trial, transformer_loss, identity_loss in zip(
            result.trials, [0.2, 0.21], [0.5, 0.51]
        ):
            self.assertAlmostEqual(
                trial.transformer.test["1111"]["loss"], transformer_loss
            )
            self.assertAlmostEqual(trial.identity.test["1111"]["loss"], identity_loss)
            self.assertAlmostEqual(
                trial.transformer.initial_test["1111"]["loss"], transformer_loss + 0.2
            )
            self.assertEqual(set(trial.transformer.support), {"0000"})
        self.assertEqual(len(progress), 4)

    def test_cli_rejects_invalid_protocol_before_writing_or_evaluation(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory(prefix="rttt-validation-tests-") as directory:
            root = Path(directory)
            discovery, output, rule_file = (
                root / "discovery.json",
                root / "validation.json",
                root / "rule.json",
            )
            discovery.write_text(json.dumps(self.discovery()))
            rule_file.write_text(json.dumps(asdict(Rule(7))))
            args = ["validate", "--discovery", str(discovery), "--output", str(output)]
            invalid_options = [
                ["--support-size", "0"],
                ["--test-size", "0"],
                ["--core-seed", "0"],
                ["--core-seed", "10", "--core-seed", "10"],
                ["--rule-file", str(rule_file)],
            ]
            with patch("inductive_bias.__main__.validate") as validation:
                for options in invalid_options:
                    with self.subTest(options=options):
                        result = runner.invoke(cli, args + options)
                        self.assertNotEqual(result.exit_code, 0)
                        self.assertFalse(output.with_suffix(".protocol.json").exists())
                output.write_text("existing result\n")
                result = runner.invoke(cli, args)
                self.assertNotEqual(result.exit_code, 0)
                self.assertEqual(output.read_text(), "existing result\n")
                self.assertFalse(output.with_suffix(".protocol.json").exists())
            validation.assert_not_called()

    def test_cli_freezes_split_before_evaluation(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory(prefix="rttt-binary-tests-") as directory:
            root = Path(directory)
            discovery = root / "discovery.json"
            output = root / "validation.json"
            discovery.write_text(json.dumps(self.discovery()))

            def check_frozen(config, budget, support, test, seeds, device, **kwargs):
                frozen = json.loads((root / "validation.protocol.json").read_text())
                self.assertEqual(frozen["model"], asdict(config))
                self.assertEqual(frozen["budget"], asdict(budget))
                self.assertEqual(frozen["support"], support)
                self.assertEqual(frozen["test"], test)
                self.assertEqual(frozen["core_seeds"], list(seeds))
                self.assertEqual(frozen["device"], device)
                self.assertEqual((len(support), len(test)), (2, 3))
                self.assertFalse(set(support) & set(test))
                self.assertFalse(
                    set(support + test) & set(self.discovery()["discovery_sequences"])
                )
                rule = Rule(**frozen["rule"])
                self.assertTrue(all(rule.contains(s) for s in support + test))
                metrics = lambda sequences: {
                    s: {"loss": 0.5, "accuracy": 0.8} for s in sequences
                }
                evaluation = CoreEvaluation(
                    metrics(test), metrics(support), metrics(test)
                )
                return ValidationResult(
                    [ValidationTrial(seed, evaluation, evaluation) for seed in seeds],
                    {"unigram": metrics(test), "bigram": metrics(test)},
                )

            args = [
                "validate",
                "--discovery",
                str(discovery),
                "--ones",
                "0",
                "4",
                "--support-size",
                "2",
                "--test-size",
                "3",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
            with patch(
                "inductive_bias.__main__.validate", side_effect=check_frozen
            ) as validation:
                result = runner.invoke(cli, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(validation.call_count, 1)
            saved = json.loads(output.read_text())
            self.assertEqual(saved["kind"], "validation")
            self.assertEqual(saved["budget"], self.discovery()["budget"])
            self.assertEqual(len(saved["trials"]), 3)
            self.assertEqual(
                saved["aggregate"]["transformer"]["mean_loss"]["mean"], 0.5
            )


if __name__ == "__main__":
    unittest.main()
