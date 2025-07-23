import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from pandas import DataFrame
from scipy.stats import chi2  # type: ignore

from similarity_bayes.utils import range_without, softmax_1d
from similarity_bayes.utils.data_types import ChoiceCounts


def _softmax_loss_gradients(params: np.ndarray, counts: np.ndarray):
    ps = softmax_1d(params)
    grads = (counts.sum() * ps) - counts
    return grads


def _get_constant_mnl_model(choice_counts: ChoiceCounts):
    n_items, counts = choice_counts.n_items, choice_counts.masks_and_counts
    total_counts = np.zeros(n_items)
    for mask, count in counts:
        assert len(mask) == count.shape[0]
        total_counts[mask] += count
    if np.any(total_counts == 0):
        warnings.warn(
            "Unbounded problem due to unselected option(s) ",
            # f"{np.argwhere(total_counts == 0)}",
            UserWarning,
        )

    def nll_function(parameters: np.ndarray) -> float:
        ps = np.insert(parameters, 0, 0)
        ps = softmax_1d(ps)

        nll = 0
        for mask, count in counts:
            ps_rem = ps[mask] / ps[mask].sum()
            nll -= np.sum(np.log(ps_rem) * count)
        return nll

    def nll_jacobian(parameters: np.ndarray) -> np.ndarray:
        params = np.insert(parameters, 0, 0)

        gradients = np.zeros_like(params)
        for mask, count in counts:
            params_rem = params[mask]
            gradients[mask] += _softmax_loss_gradients(params_rem, count)
        return gradients[1:]

    utilities = np.zeros(n_items - 1)
    return utilities, nll_function, nll_jacobian


def gradient_descent_optimize(
    initial_solution: np.ndarray,
    objective: Callable[[np.ndarray], float],
    jacobian: Callable[[np.ndarray], np.ndarray],
    step_size: float = 0.05,
    stopping_delta: float = 0.1,
    max_iter: int = 50,
) -> tuple[np.ndarray, float]:
    running_nll = objective(initial_solution)
    running_solution = initial_solution.copy()

    for it in range(max_iter):
        running_solution -= step_size * jacobian(running_solution)
        new_nll = objective(running_solution)

        if new_nll >= running_nll - stopping_delta:
            running_nll = new_nll
            break

        running_nll = new_nll
    if it >= max_iter - 1:
        warnings.warn(
            f"Failed to converge before max iterations {max_iter}.",
            UserWarning,
        )
    return running_solution, running_nll


def _mcfadden_train_tye_test(
    choice_counts: ChoiceCounts, optimize_kwargs: dict[str, Any]
) -> list[tuple[int, float]]:
    stats = []
    full_params, full_obj, full_jac = _get_constant_mnl_model(choice_counts)
    full_sol, _ = gradient_descent_optimize(
        full_params, full_obj, full_jac, **optimize_kwargs
    )
    for i in range(choice_counts.n_items):
        red_choice_counts = choice_counts.exclude_items([i])
        red_params, red_obj, red_jac = _get_constant_mnl_model(red_choice_counts)
        red_sol, red_nll = gradient_descent_optimize(
            red_params, red_obj, red_jac, **optimize_kwargs
        )
        if i == 0:
            full_nll = red_obj(full_sol[1:] - full_sol[0])
        else:
            full_nll = red_obj(
                full_sol[range_without(red_choice_counts.n_items, i - 1)]
            )
        mtt = 2 * (full_nll - red_nll)
        stats.append((red_params.shape[0], mtt))
    return stats


def mcfadden_train_tye_tests(
    full_questions: np.ndarray,
    leave_one_out_questions: list[np.ndarray],
    optimize_kwargs=None,
) -> list[dict[str, float]]:
    if optimize_kwargs is None:
        optimize_kwargs = dict()

    chi2_stats = []

    for q in range(full_questions.shape[0]):
        choice_counts = ChoiceCounts.from_full_and_leave_one_out_questions(
            full_questions[q],
            [le_questions[q] for le_questions in leave_one_out_questions],
        ).without_unselected_options()

        if choice_counts.n_items <= 2:
            warnings.warn(f"question {q} with two or less items selected, skipping...")
            stats = [(0, 0.0)]
        else:
            stats = _mcfadden_train_tye_test(choice_counts, optimize_kwargs)
        for i, (dgf, mtt) in enumerate(stats):
            chi2_stats.append(
                {
                    "question_set": q,
                    "rem": i,
                    "stat": mtt,
                    "dgf": dgf,
                    "p-val": chi2.sf(mtt, df=dgf) if dgf > 0 else 1,
                }
            )
    return chi2_stats


def _goodness_of_fit_stat(
    choice_counts: ChoiceCounts, optimize_kwargs: dict[str, int]
) -> tuple[int, int, float]:
    params, obj, jac = _get_constant_mnl_model(choice_counts)

    opt, _ = gradient_descent_optimize(params, obj, jac, **optimize_kwargs)
    ps_full = softmax_1d(np.insert(opt, 0, 0))

    chi2_stat, n_bins = 0.0, 0
    for mask, count in choice_counts.masks_and_counts:
        ps_sub = ps_full[mask] / ps_full[mask].sum()
        n_choices = count.sum()
        expected = n_choices * ps_sub
        chi2_stat += np.sum((expected - count) ** 2 / expected)
        n_bins += count.shape[0] - 1
    dgf = n_bins - params.shape[0]
    n_params = params.shape[0]
    return dgf, n_params, chi2_stat


def goodness_of_fit_test(
    full_questions: np.ndarray,
    leave_one_out_questions: list[np.ndarray],
    optimize_kwargs=None,
) -> list[dict[str, float]]:
    if optimize_kwargs is None:
        optimize_kwargs = dict()

    stats = []
    for q in range(full_questions.shape[0]):
        choice_counts = ChoiceCounts.from_full_and_leave_one_out_questions(
            full_questions[q],
            [le_questions[q] for le_questions in leave_one_out_questions],
        ).without_unselected_options()
        dgf, n_params, stat = _goodness_of_fit_stat(
            choice_counts, optimize_kwargs=optimize_kwargs
        )
        stats.append(
            {
                "question_set": q,
                "stat": stat,
                "dgf": dgf,
                "n_params": n_params,
                "p-val": chi2.sf(stat, df=dgf),
            }
        )
    return stats


def goodness_of_fit_test_handcrafted(
    questions_a: np.ndarray,
    questions_b: np.ndarray,
    optimize_kwargs=None,
) -> list[dict[str, float]]:
    if optimize_kwargs is None:
        optimize_kwargs = dict()

    stats = []
    for q in range(questions_a.shape[0]):
        choice_counts = ChoiceCounts.from_handcrafted_survey(
            questions_a[q], questions_b[q]
        ).without_unselected_options()
        dgf, n_params, stat = _goodness_of_fit_stat(
            choice_counts, optimize_kwargs=optimize_kwargs
        )
        stats.append(
            {
                "question_set": q,
                "stat": stat,
                "dgf": dgf,
                "n_params": n_params,
                "p-val": chi2.sf(stat, df=dgf),
            }
        )
    return stats


def pop_hom_stat(df_answers: DataFrame, option_cols: list[Any]):
    empirical_probs = df_answers.groupby("question_id")[option_cols].mean()

    likelihoods = (
        df_answers[option_cols]
        * df_answers[["question_id"]].merge(
            empirical_probs, left_on="question_id", right_index=True
        )[option_cols]
    ).sum(axis=1)

    nlls = (-np.log(likelihoods)).to_frame("nll")
    nlls["user_id"] = df_answers["user_id"]
    return nlls.groupby("user_id")["nll"].sum()


def permute_answers(df_answers: DataFrame, option_cols: list[Any]) -> DataFrame:
    df_ans_p = df_answers.copy()
    for qid in df_ans_p["question_id"].unique():
        answers_shuffled = df_ans_p.loc[
            df_ans_p["question_id"] == qid, option_cols
        ].sample(frac=1)
        df_ans_p.loc[df_ans_p["question_id"] == qid, option_cols] = (
            answers_shuffled.values
        )
    return df_ans_p


def pop_hom_permutation_test(
    df_answers: DataFrame, n_options: int, n_samples: int
) -> tuple[float, list[float]]:
    option_cols = list(range(n_options))
    user_ics = pop_hom_stat(df_answers, option_cols)
    stat = user_ics.max() - user_ics.min()
    stats = []
    for n in range(n_samples):
        df_ans_p = permute_answers(df_answers, option_cols)
        user_ics = pop_hom_stat(df_ans_p, option_cols)
        stats.append(user_ics.max() - user_ics.min())
    return stat, stats
