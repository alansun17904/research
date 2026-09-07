from dataclasses import dataclass
import json
import math
from types import SimpleNamespace

import pytest
import torch

from rttt.attention import AttentionConfig
from rttt.benchmarks import evaluate_perplexity, generate, iter_corpus_tokens, replay_helm


class Tokenizer:
    bos_token_id = eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [int(char) for char in text if char.isdigit()]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(i) for i in ids if not skip_special_tokens or i != 0)

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


class ToyModel:
    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=4)
    max_sequence_length = None
    attention_config = AttentionConfig()
    state_bytes = 32
    cached_tokens = 1

    def __init__(self):
        self.histories = []
        self.reset()

    def reset(self):
        self.history = []
        self.histories.append(self.history)

    def set_attention(self, config):
        self.attention_config = config
        self.reset()

    def step(self, ids):
        token = int(ids.item())
        self.history.append(token)
        # Predict the next digit; depends on the actual input, catching shifts.
        logits = torch.zeros(1, 10)
        logits[0, (token + 1) % 10] = 2
        return logits

    def prefill(self, ids):
        for token in ids.unbind(1):
            logits = self.step(token)
        return logits


@pytest.fixture
def adapter():
    pytest.importorskip("lm_eval")
    from rttt.lm_eval_adapter import SequentialLM
    return SequentialLM(ToyModel(), Tokenizer())


@dataclass
class Request:
    args: tuple


def expected_loss(correct=True):
    return math.log(math.exp(2) + 9) - (2 if correct else 0)


def test_perplexity_scores_shifted_targets_exactly_once_and_bounds_iterator():
    consumed = []

    def tokens():
        for i in range(10):
            consumed.append(i)
            yield i

    model = ToyModel()
    result = evaluate_perplexity(model, tokens(), max_tokens=4, warmup_tokens=1, log_every=2)
    assert consumed == [0, 1, 2, 3, 4]
    assert model.history == [0, 1, 2, 3]
    assert result["tokens"] == 3 and result["predicted_tokens"] == 4
    assert result["mean_nll"] == pytest.approx(expected_loss())
    assert result["perplexity"] == pytest.approx(math.exp(expected_loss()))
    assert result["peak_state_bytes"] == 32
    assert len(result["trace"]) == 2


def test_explicit_resets_preserve_target_count():
    model = ToyModel()
    result = evaluate_perplexity(model, range(7), reset_interval=2)
    assert result["tokens"] == 6 and result["segments"] == 3
    assert model.histories[-3:] == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize("ids,kwargs", [([], {}), ([1], {}), ([1, 2], {"warmup_tokens": 2}),
                                       ([1, 2], {"max_tokens": 0})])
def test_empty_scoring_fails(ids, kwargs):
    with pytest.raises(ValueError):
        evaluate_perplexity(ToyModel(), ids, **kwargs)


def test_local_corpus_bos_is_explicit(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("123")
    assert list(iter_corpus_tokens(Tokenizer(), "text", text_path=path)) == [1, 2, 3]
    assert list(iter_corpus_tokens(Tokenizer(), "text", text_path=path, prepend_bos=True)) == [0, 1, 2, 3]


def test_generate_handles_multitoken_stop_and_missing_stop_without_truncation():
    model = ToyModel()
    result = generate(model, Tokenizer(), "1", max_new_tokens=5, stop=["34"])
    assert result["text"] == "2" and result["finish_reason"] == "stop"
    result = generate(model, Tokenizer(), "1", max_new_tokens=3, stop=["99"])
    assert result["text"] == "234" and result["finish_reason"] == "length"
    assert result["token_logprobs"] == pytest.approx([-expected_loss()] * 3)


def test_independent_likelihood_requests_reset_and_use_continuation_only(adapter):
    results = adapter.loglikelihood([Request(("12", "34")), Request(("56", "78"))])
    assert [value for value, _ in results] == pytest.approx([-2 * expected_loss()] * 2)
    assert all(greedy for _, greedy in results)
    assert adapter.runner.histories[-2:] == [[1, 2, 3], [5, 6, 7]]


def test_likelihood_truncation_and_rolling_token_accounting(adapter):
    result = adapter.loglikelihood([Request(("12345", "67"))])
    assert result[0][0] == pytest.approx(-2 * expected_loss())
    assert adapter.runner.history == [3, 4, 5, 6]
    assert adapter.truncated_requests == 1
    score = adapter.loglikelihood_rolling([Request(("123456789",))])[0]
    assert score == pytest.approx(-9 * expected_loss())


def test_harness_generation_leaves_room_for_decode(adapter):
    result = adapter.generate_until([Request(("123456", {"max_gen_toks": 2, "until": ["9"]}))])
    assert result == ["78"]
    assert adapter.truncated_requests == adapter.request_count == 1
    assert adapter.runner.history == [4, 5, 6, 7]
    assert adapter.generate_until([Request(("1", {"max_gen_toks": 0}))]) == [""]


def test_helm_replay_preserves_request_and_is_independent(tmp_path):
    source, dest = tmp_path / "requests.jsonl", tmp_path / "results.jsonl"
    request = {"prompt": "1", "max_tokens": 4, "temperature": 0, "top_p": 1, "n": 2, "stop": ["4"]}
    source.write_text(json.dumps({"request": request}) + "\n")
    summary = replay_helm(ToyModel(), Tokenizer(), source, dest)
    row = json.loads(dest.read_text())
    assert row["request"] == request and summary["requests"] == 1
    assert [choice["text"] for choice in row["result"]["choices"]] == ["23", "23"]
    with pytest.raises(ValueError, match="differ"):
        replay_helm(ToyModel(), Tokenizer(), source, source)


def test_real_harness_interface_with_offline_dataset(monkeypatch):
    pytest.importorskip("lm_eval")
    import datasets
    from rttt.lm_eval_adapter import run_lm_eval

    rows = datasets.Dataset.from_dict({"goal": ["1", "4"], "sol1": ["2", "5"],
                                      "sol2": ["3", "6"], "label": [0, 0]})

    def load(path, *args, **kwargs):
        assert path == "baber/piqa"
        assert kwargs["trust_remote_code"] is False
        return datasets.DatasetDict(train=rows, validation=rows)

    monkeypatch.setattr(datasets, "load_dataset", load)
    result = run_lm_eval(ToyModel(), Tokenizer(), tasks="piqa", num_fewshot=0, limit=2)
    assert result["results"]["piqa"]["acc,none"] == 1.0
    assert result["rttt_diagnostics"]["requests"] == 4
