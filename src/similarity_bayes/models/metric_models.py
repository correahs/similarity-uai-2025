from functools import reduce
from typing import Self

import numpy as np
from numpy.random import Generator, default_rng
from pytensor import function as pt_function
from pytensor import shared as pt_shared
from pytensor import tensor as pt
from pytensor.gradient import jacobian
from pymc import adam
from tqdm import tqdm  # type: ignore

from similarity_bayes.utils import pad, softmax_func
from similarity_bayes.utils.data_types import (
    SimilarityChoiceCounts,
    SimilarityQuestionInput,
)


class TSTEModel:
    def __init__(
        self: Self,
        n_items: int,
        n_dims: int,
        df: int | None = None,
        max_iter: int = 5000,
        train_with_triplets: bool = False,
        l2_reg: float = 0.01,
        rng: Generator | None = None,
    ) -> None:
        if rng is None:
            rng = default_rng()

        self.n_items = n_items
        self.n_dims = n_dims
        self.alpha = df if df is not None else n_dims - 1
        self.max_iter = max_iter
        self.train_with_triplets = train_with_triplets
        self.l2_reg = l2_reg

        self.embedding = pt_shared(
            rng.normal(size=(n_items + 1, n_dims)), shape=(n_items + 1, n_dims)
        )
        self._emb_extractor = pt_function([], self.embedding)
        self.pad_item = n_items

    def get_predict_func(
        self: Self,
        targets: np.ndarray,
        choice_sets: np.ndarray,
        n_participants: np.ndarray,
    ):
        n_participants = pt_shared(n_participants, shape=n_participants.shape)
        # mask is used to make the logits of the padding element = -inf
        mask = np.full_like(choice_sets, +np.inf, dtype="float")
        mask[np.nonzero(choice_sets == self.pad_item)] = -np.inf
        diffs = self.embedding[[[t] for t in targets]] - self.embedding[choice_sets]
        dists = (diffs**2).sum(axis=2)
        logits = pt.minimum(
            -pt.log(1 + (dists / self.alpha)) * (1 + self.alpha) / 2, mask
        )
        log_probs = pt.special.log_softmax(logits, axis=1)
        expected_counts = pt.exp(log_probs) * n_participants.reshape(
            (log_probs.shape[0], 1)
        )
        return log_probs, expected_counts

    def get_train_func(
        self: Self,
        targets: np.ndarray,
        choice_sets: np.ndarray,
        choices: np.ndarray,
    ):
        # f = open(os.devnull, "w")
        # sys.stdout = f
        # sys.stderr = f
        assert targets.shape[0] == choice_sets.shape[0]
        assert choice_sets.shape == choices.shape

        log_probs, expected_counts = self.get_predict_func(
            targets, choice_sets, choices.sum(axis=1)
        )

        choice_counts = pt_shared(choices, shape=choices.shape)

        log_likelihoods = choice_counts * log_probs
        nll = (
            -pt.switch(pt.isnan(log_likelihoods), 0, log_likelihoods).sum()
            / choice_counts.sum()
        )

        emb_norm = (self.embedding**2).sum(axis=1).mean()

        # emb_jacobian = jacobian(nll + self.l2_reg * emb_norm, self.embedding)
        train_func = pt_function(
            [],
            [expected_counts, nll],
            # updates=((self.embedding, self.embedding - 0.04 * emb_jacobian),),
            updates=adam(
                nll + self.l2_reg * emb_norm, [self.embedding], learning_rate=0.01
            ),
        )
        return train_func, nll, expected_counts

    def train(self: Self, data: list[SimilarityChoiceCounts]):
        losses = []
        if self.train_with_triplets:
            data = reduce(list.__add__, [s.to_triplets() for s in data], [])
            print("breaking down into triplets...")
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")
        choices = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)
            choices[i] = pad(counts.choice_counts, 0, max_items)
        train_func, nll, expected_counts = self.get_train_func(
            targets, choice_sets, choices
        )
        for iter in (pbar := tqdm(range(self.max_iter), desc="Fitting", unit="iter")):
            _, loss = train_func()
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
            losses.append(float(loss))

        return losses, pt_function([], expected_counts)()

    def predict(self: Self, data: list[SimilarityQuestionInput]):
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        participants = np.array([c.n_participants for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)

        embeddings = self._emb_extractor()
        target_embeddings = embeddings[targets]
        choice_set_embeddings = embeddings[choice_sets]

        diffs = np.expand_dims(target_embeddings, axis=1) - choice_set_embeddings

        dists = np.where(choice_sets == self.pad_item, np.inf, (diffs**2).sum(axis=2))
        logits = -np.log(1 + (dists / self.alpha)) * (1 + self.alpha) / 2
        probs = softmax_func(logits)

        return probs * np.expand_dims(participants, axis=1)

    def error(self, true_counts, expected_counts):
        acc = 0
        for c_true, c_hat in zip(true_counts, expected_counts):
            for t, h in zip(c_true, c_hat):
                acc += (t - h) ** 2 / h
        return acc

    def negative_ll(self: Self, data: list[SimilarityChoiceCounts]):
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")
        choices = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)
            choices[i] = pad(counts.choice_counts, 0, max_items)

        log_probs, _ = self.get_predict_func(targets, choice_sets, choices.sum(axis=1))

        log_probs = pt.maximum(log_probs, -1000000)

        nll = -(log_probs * choices).sum(axis=1).mean()
        return pt_function([], nll)()

    def accuracy(self: Self, data: list[SimilarityChoiceCounts]):
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")
        choices = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)
            choices[i] = pad(counts.choice_counts, 0, max_items)

        log_probs, _ = self.get_predict_func(targets, choice_sets, choices.sum(axis=1))

        acc = pt.eq(log_probs.argmax(axis=1), choices.argmax(axis=1)).mean()
        return pt_function([], acc)()


class FBOModel:
    def __init__(
        self: Self,
        n_items: int,
        n_dims: int,
        df: int | None = None,
        max_iter: int = 5000,
        l2_reg: float = 0.01,
        beta: float = 0.01,
        rng: Generator | None = None,
    ) -> None:
        if rng is None:
            rng = default_rng()

        self.n_items = n_items
        self.n_dims = n_dims
        self.alpha = df if df is not None else n_dims - 1
        self.beta = beta
        self.max_iter = max_iter
        self.l2_reg = l2_reg

        self.embedding = pt_shared(
            rng.normal(size=(n_items + 1, n_dims)), shape=(n_items + 1, n_dims)
        )
        self._emb_extractor = pt_function([], self.embedding)
        self.pad_item = n_items

    def get_predict_func(
        self: Self,
        targets: np.ndarray,
        choice_sets: np.ndarray,
        n_participants: np.ndarray,
    ):
        assert targets.shape[0] == choice_sets.shape[0]
        assert targets.shape == n_participants.shape

        n_participants = pt_shared(n_participants, shape=n_participants.shape)
        # mask is used to make the logits of the padding element = -inf
        mask = np.full_like(choice_sets, +np.inf, dtype="float")
        mask[np.nonzero(choice_sets == self.pad_item)] = -np.inf

        cs_emb = self.embedding[choice_sets]

        centered = cs_emb - cs_emb.mean(axis=1, keepdims=True)

        choice_set_cov = pt.einsum("qck,qcl->qkl", centered, centered.copy()) / (
            choice_sets.shape[1] - 1
        )

        identity = pt_shared(
            np.stack([np.identity(self.n_dims) for _ in range(choice_sets.shape[0])]),
            shape=(choice_sets.shape[0], self.n_dims, self.n_dims),
        )
        att_matrix = (self.beta) * identity + (1 - self.beta) * choice_set_cov
        diffs = self.embedding[[[t] for t in targets]] - self.embedding[choice_sets]

        # diffs = questions x choice_set_size x dims
        # choice_set_cov = questions x dims x dims
        # squared mahalanobis = diffs^T x choice_set_cov x diffs
        dists = pt.einsum("qck,qjk,qcj->qc", diffs, att_matrix, diffs.copy())

        # dists2 = (diffs**2).sum(axis=2)

        logits = pt.minimum(
            -pt.log(1 + (dists / self.alpha)) * (1 + self.alpha) / 2, mask
        )
        log_probs = pt.special.log_softmax(logits, axis=1)
        expected_counts = pt.exp(log_probs) * n_participants.reshape(
            (log_probs.shape[0], 1)
        )
        return log_probs, expected_counts

    def predict(self: Self, data: list[SimilarityQuestionInput]):
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        participants = np.array([c.n_participants for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)

        _, expected_counts = self.get_predict_func(targets, choice_sets, participants)

        return pt_function([], expected_counts)()

    def negative_ll(self: Self, data: list[SimilarityChoiceCounts]):
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")
        choices = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)
            choices[i] = pad(counts.choice_counts, 0, max_items)

        log_probs, _ = self.get_predict_func(targets, choice_sets, choices.sum(axis=1))

        log_probs = pt.maximum(log_probs, -1000000)

        nll = -(log_probs * choices).sum(axis=1).mean()
        return pt_function([], nll)()

    def accuracy(self: Self, data: list[SimilarityChoiceCounts]):
        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")
        choices = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)
            choices[i] = pad(counts.choice_counts, 0, max_items)

        log_probs, _ = self.get_predict_func(targets, choice_sets, choices.sum(axis=1))

        acc = pt.eq(log_probs.argmax(axis=1), choices.argmax(axis=1)).mean()
        return pt_function([], acc)()

    def get_train_func(
        self: Self,
        targets: np.ndarray,
        choice_sets: np.ndarray,
        choices: np.ndarray,
    ):
        choice_counts = pt_shared(choices, shape=choices.shape)

        log_probs, expected_counts = self.get_predict_func(
            targets, choice_sets, choices.sum(axis=1)
        )

        log_likelihoods = choice_counts * log_probs
        nll = (
            -pt.switch(pt.isnan(log_likelihoods), 0, log_likelihoods).sum()
            / choice_counts.sum()
        )

        emb_norm = (self.embedding**2).sum(axis=1).mean()

        # emb_jacobian = jacobian(nll + self.l2_reg * emb_norm, self.embedding)
        train_func = pt_function(
            [],
            [expected_counts, nll],
            # updates=((self.embedding, self.embedding - 0.04 * emb_jacobian),),
            updates=adam(
                nll + self.l2_reg * emb_norm, [self.embedding], learning_rate=0.02
            ),
        )
        return train_func, nll, expected_counts

    def train(self: Self, data: list[SimilarityChoiceCounts]):
        losses = []

        max_items = max([len(sim_counts.choice_set) for sim_counts in data])

        targets = np.array([c.target for c in data])
        choice_sets = np.empty((len(data), max_items), dtype="int")
        choices = np.empty((len(data), max_items), dtype="int")

        for i, counts in enumerate(data):
            choice_sets[i] = pad(counts.choice_set, self.pad_item, max_items)
            choices[i] = pad(counts.choice_counts, 0, max_items)
        train_func, nll, expected_counts = self.get_train_func(
            targets, choice_sets, choices
        )
        for iter in (pbar := tqdm(range(self.max_iter), desc="Fitting", unit="iter")):
            _, loss = train_func()
            losses.append(float(loss))
            pbar.set_postfix({"Loss": f"{loss:.4f}"})

        return losses, pt_function([], expected_counts)()

    def error(self, true_counts, expected_counts):
        acc = 0
        for c_true, c_hat in zip(true_counts, expected_counts):
            for t, h in zip(c_true, c_hat):
                acc += (t - h) ** 2 / h
        return acc
