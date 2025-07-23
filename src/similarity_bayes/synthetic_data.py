import numpy as np

from similarity_bayes.utils import range_without, softmax_1d, softmax_func
from similarity_bayes.utils.data_types import SimilarityChoiceCounts


def IIA_simulate(
    N_participants,
    N_questions,
    std_v,
    max_options=4,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    v = np.random.normal(loc=0, scale=std_v, size=(N_questions, max_options))
    p_full = softmax_func(v)
    res_full = rng.multinomial(N_participants, pvals=p_full)
    res_rems = []
    for i in range(max_options):
        p_rem = softmax_func(v[:, range_without(max_options, i)])
        res_rems.append(rng.multinomial(N_participants, pvals=p_rem))
    return res_full, res_rems


def additive_context_simulate(
    N_participants,
    N_questions,
    std_v,
    std_c,
    max_options=4,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    v = np.random.normal(loc=0, scale=std_v, size=(N_questions, max_options))
    p_full = softmax_func(v)
    res_full = rng.multinomial(N_participants, pvals=p_full)
    res_rems = []
    for i in range(max_options):
        p_rem = softmax_func(
            v[:, range_without(max_options, i)]
            + np.random.normal(loc=0, scale=std_c, size=(N_questions, max_options - 1))
        )
        res_rems.append(rng.multinomial(N_participants, pvals=p_rem))
    return res_full, res_rems


def simple_context_simulate(
    N_participants,
    N_questions,
    std_v,
    std_c,
    max_options=4,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    v = np.random.normal(loc=0, scale=std_v, size=(N_questions, 4))
    p_full = softmax_func(v)
    res_full = rng.multinomial(N_participants, pvals=p_full)
    res_rems = []
    for i in range(max_options):
        p_rem = softmax_func(
            v[:, range_without(max_options, i)]
            * np.random.normal(loc=1, scale=std_c, size=(N_questions, 1))
        )
        res_rems.append(rng.multinomial(N_participants, pvals=p_rem))
    return res_full, res_rems


def noise_context_simulate(
    N_participants,
    N_questions,
    std_v,
    alpha,
    max_options=4,
    rng: np.random.Generator | None = None,
):
    """perform mixture with uniform choice"""
    if rng is None:
        rng = np.random.default_rng()

    v = np.random.normal(loc=0, scale=std_v, size=(N_questions, 4))
    p_full = softmax_func(v)
    res_full = rng.multinomial(N_participants, pvals=p_full)
    res_rems = []
    p_uni = np.ones(max_options - 1) / (max_options - 1)
    for i in range(max_options):
        N_partial = np.random.binomial(N_participants, p=alpha)
        p_rem = softmax_func(v[:, range_without(max_options, i)])
        res = rng.multinomial(N_partial, pvals=p_rem) + rng.multinomial(
            N_participants - N_partial, pvals=p_uni
        )
        res_rems.append(res)
    return res_full, res_rems


def metric_choices_simulate(
    n_participants,
    n_questions,
    n_items,
    n_dims,
    std_emb,
    max_options=4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    embeddings = rng.normal(scale=std_emb, size=(n_items, n_dims))
    items = np.arange(n_items)
    targets = np.empty(n_questions, dtype=int)
    choice_sets = np.empty((n_questions, max_options), dtype=int)
    choices = np.empty((n_questions, max_options), dtype=int)
    for i in range(n_questions):
        target = np.random.choice(items)
        items_except_target = np.setdiff1d(items, [target])
        choice_set = rng.choice(items_except_target, size=max_options, replace=False)
        #  use tste to generate choices
        alpha = n_dims - 1
        diffs = embeddings[[target]] - embeddings[choice_set]
        dists = (diffs**2).sum(axis=1)
        logits = -np.log(1 + (dists / alpha)) * (1 + alpha) / 2
        probs = softmax_1d(logits)
        choice = rng.multinomial(n_participants, probs)

        targets[i] = target
        choice_sets[i] = choice_set
        choices[i] = choice

    return embeddings, targets, choice_sets, choices


def covariance_weighted_norm_choices_simulate(
    n_participants,
    n_questions,
    n_items,
    n_dims,
    std_emb,
    max_options=4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    embeddings = rng.normal(scale=std_emb, size=(n_items, n_dims))

    items = np.arange(n_items)
    targets = np.empty(n_questions, dtype=int)
    choice_sets = np.empty((n_questions, max_options), dtype=int)
    choices = np.empty((n_questions, max_options), dtype=int)

    for i in range(n_questions):
        target = np.random.choice(items)
        items_except_target = np.setdiff1d(items, [target])
        choice_set = rng.choice(items_except_target, size=max_options, replace=False)
        #  use inv mahalanobis to generate choices
        cov = np.cov(embeddings[choice_set], rowvar=False)
        diffs = embeddings[[target]] - embeddings[choice_set]
        # diffs = questions x choice_set_size x dims
        # choice_set_cov = questions x dims x dims
        # squared mahalanobis = diffs^T x choice_set_cov x diffs
        dists = np.einsum("ck,jk,cj->c", diffs, cov, diffs)

        probs = softmax_1d(-dists)
        choice = rng.multinomial(n_participants, probs)

        targets[i] = target
        choice_sets[i] = choice_set
        choices[i] = choice

    return embeddings, targets, choice_sets, choices


def metric_choices_leave_one_out_simulate(
    n_participants,
    n_questions,
    n_items,
    n_dims,
    std_emb,
    max_options=4,
    method="metric",
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    if method == "metric":
        method = metric_choices_simulate
    elif method == "cov_weighted":
        method = covariance_weighted_norm_choices_simulate
    else:
        raise Exception

    embeddings, targets, choice_sets, choices = method(
        n_participants, n_questions, n_items, n_dims, std_emb, max_options, rng
    )
    choice_counts = []
    for i in range(n_questions):
        target = targets[i]
        for j in range(max_options):
            mask = range_without(max_options, j)
            choice_set = list(choice_sets[i][mask])
            choice = list(choices[i][mask])

            if sum(choice) > 0:
                choice_counts.append(SimilarityChoiceCounts(target, choice_set, choice))
    return choice_counts, embeddings


def metric_choices_irregular_sets_simulate(
    n_participants,
    n_questions,
    n_items,
    n_dims,
    std_emb,
    max_options=4,
    method="metric",
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    if method == "metric":
        method = metric_choices_simulate
    elif method == "cov_weighted":
        method = covariance_weighted_norm_choices_simulate
    else:
        raise Exception

    embeddings, targets, choice_sets, choices = method(
        n_participants, n_questions, n_items, n_dims, std_emb, max_options, rng
    )
    choice_counts = []
    for i in range(n_questions):
        target = targets[i]
        choice_set = list(choice_sets[i])
        choice = list(choices[i])
        if rng.random() > 0.5:
            del choice_set[-1]
            del choice[-1]
        choice_counts.append(SimilarityChoiceCounts(target, choice_set, choice))
    return choice_counts
