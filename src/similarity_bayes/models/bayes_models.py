from collections.abc import Callable
from itertools import product
from typing import Self

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore
from xarray import Dataset

from similarity_bayes.utils import range_without as _range_without


class BayesChoiceModel:
    model: pm.Model
    mult_params: dict[str, pm.Distribution]

    def __init__(
        self: Self,
        n_questions: int,
        choice_set_versions: dict[str, list[int]],
        max_options: int,
    ):
        raise NotImplementedError

    def _setup(self: Self, model: pm.Model, mult_params: pm.Distribution):
        self.model = model
        self.mult_params = mult_params

    def fit(
        self: Self,
        data_dict: dict[str, np.ndarray],
        chain=4,
        tune=1000,
        draws=1000,
    ):
        with self.model:
            for ver_key, counts in data_dict.items():
                n_participants = counts.sum(axis=1)
                pm.Multinomial(
                    f"counts_{ver_key}",
                    n=n_participants,
                    p=self.mult_params[ver_key],
                    observed=counts,
                    dims=("question", f"option_{ver_key}"),
                )
            prior_pred = pm.sample_prior_predictive()
            trace = pm.sample(tune=tune, draws=draws, chains=chain)
            post_pred = pm.sample_posterior_predictive(trace=trace)
        return prior_pred, trace, post_pred


class IIAModel(BayesChoiceModel):
    def __init__(
        self: Self,
        n_questions: int,
        choice_set_versions: dict[str, list[int]],
        max_options: int,
        hp_std: float,
    ):
        with pm.Model(
            coords={
                "question": np.arange(n_questions),
                "option": np.arange(max_options),
                **{
                    f"option_{k}": np.arange(len(v))
                    for k, v in choice_set_versions.items()
                },
            }
        ) as model:
            std_sim = pm.HalfNormal("std", sigma=hp_std)
            sims = pm.Normal("sim", mu=0, sigma=std_sim, dims=["question", "option"])
            mult_params = {
                k: pm.Deterministic(
                    f"p_{k}",
                    pm.math.softmax(sims[:, v], axis=1),
                    dims=["question", f"option_{k}"],
                )
                for k, v in choice_set_versions.items()
            }
        self._setup(model, mult_params)


class AdditivePerturbationModel(BayesChoiceModel):
    def __init__(
        self: Self,
        n_questions: int,
        choice_set_versions: dict[str, list[int]],
        max_options: int,
        hp_std: float,
        hp_std_p: float,
    ):
        with pm.Model(
            coords={
                "question": np.arange(n_questions),
                "option": np.arange(max_options),
                **{
                    f"option_{k}": np.arange(len(v))
                    for k, v in choice_set_versions.items()
                },
            }
        ) as model:
            std_sim = pm.HalfNormal("std", sigma=hp_std)
            std_p = pm.HalfNormal("std_p", sigma=hp_std_p)
            sims = pm.Normal("sim", mu=0, sigma=std_sim, dims=["question", "option"])
            mult_params = {
                k: pm.Deterministic(
                    f"p_{k}",
                    pm.math.softmax(
                        sims[:, v]
                        + pm.Normal(
                            f"pert_{k}",
                            mu=0,
                            sigma=std_p,
                            dims=["question", f"option_{k}"],
                        ),
                        axis=1,
                    ),
                    dims=["question", f"option_{k}"],
                )
                for k, v in choice_set_versions.items()
            }
        self._setup(model, mult_params)


class MultiplicativePerturbationModel(BayesChoiceModel):
    def __init__(
        self: Self,
        n_questions: int,
        choice_set_versions: dict[str, list[int]],
        max_options: int,
        hp_std: float,
        hp_std_p: float,
    ):
        with pm.Model(
            coords={
                "question": np.arange(n_questions),
                "option": np.arange(max_options),
                **{
                    f"option_{k}": np.arange(len(v))
                    for k, v in choice_set_versions.items()
                },
            }
        ) as model:
            std_sim = pm.HalfNormal("std", sigma=hp_std)
            std_p = pm.HalfNormal("std_p", sigma=hp_std_p)
            sims = pm.Normal("sim", mu=0, sigma=std_sim, dims=["question", "option"])
            mult_params = {
                k: pm.Deterministic(
                    f"p_{k}",
                    pm.math.softmax(
                        sims[:, v]
                        + pm.math.stack(
                            [pm.Normal(f"pert_{k}", mu=1, sigma=std_p)] * len(v),
                            axis=1,
                        ),
                        axis=1,
                    ),
                    dims=["question", f"option_{k}"],
                )
                for k, v in choice_set_versions.items()
            }
        self._setup(model, mult_params)


def build_simple_survey_model(
    model_kind: type[BayesChoiceModel],
    n_questions: int,
    n_options: int,
    **model_params,
) -> BayesChoiceModel:
    q_versions = {"o": list(range(n_options))}
    return model_kind(n_questions, q_versions, n_options, **model_params)


def build_random_survey_model(
    model_kind: type[BayesChoiceModel],
    n_questions: int,
    max_options: int,
    **model_params,
) -> BayesChoiceModel:
    q_versions = {
        "full": list(range(max_options)),
        **{f"rem_{i}": _range_without(max_options, i) for i in range(max_options)},
    }
    return model_kind(n_questions, q_versions, max_options, **model_params)


def build_handcrafted_survey_model(
    model_kind: type[BayesChoiceModel],
    n_questions: int,
    max_options: int,
    **model_params,
) -> BayesChoiceModel:
    q_versions = {
        "A": list(range(max_options - 1)),
        "B": list(range(max_options - 2)) + [max_options - 1],
    }
    return model_kind(n_questions, q_versions, max_options, **model_params)


def squared_relative_error(
    n_participants: np.ndarray, observed: np.ndarray, ps: np.ndarray
) -> np.ndarray:
    expected = ps * n_participants.reshape(-1, 1)
    error = (expected - observed) ** 2 / expected

    return error.sum(axis=1)


def nll_error(
    n_participants: np.ndarray, observed: np.ndarray, ps: np.ndarray
) -> np.ndarray:
    expected = ps * n_participants.reshape(-1, 1)

    nll = observed * np.log(observed / expected)
    return nll.sum(axis=1)


def _posterior_predictive_check_sample(
    n_questions: int,
    data: dict[str, np.ndarray],
    posterior_sample: Dataset,
    stat_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    stat = np.zeros(n_questions)
    stat_repl = np.zeros(n_questions)

    for conf in data.keys():
        n_participants = data[conf].sum(axis=1)

        ps = posterior_sample[f"p_{conf}"].values
        data_repl = rng.multinomial(n_participants, pvals=ps)

        stat += stat_func(n_participants, data[conf], ps)
        stat_repl += stat_func(n_participants, data_repl, ps)

    df = pd.DataFrame(
        np.stack([np.arange(n_questions), stat, stat_repl], axis=-1),
        columns=["question", "stat", "stat_repl"],
    )

    return df


def posterior_predictive_check(
    n_questions: int,
    data: dict[str, np.ndarray],
    posterior: Dataset,
    stat_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    aggregate_pvals: bool = False,
    agg: str = "sum",
) -> float | pd.Series:
    rng = np.random.default_rng()

    dfs = []
    for chain in posterior.chain.values:
        for draw in posterior.draw.values:
            posterior_sample = posterior.sel(chain=chain, draw=draw)
            df = _posterior_predictive_check_sample(
                n_questions, data, posterior_sample, stat_func, rng
            )
            df["chain"] = chain
            df["draw"] = draw
            dfs.append(df)

    df_concat = pd.concat(dfs)

    if aggregate_pvals:
        df_agg_stat = df_concat.groupby(["chain", "draw"])[["stat", "stat_repl"]].agg(
            agg
        )
        return (df_agg_stat["stat_repl"] > df_agg_stat["stat"]).mean()
    else:
        df_concat["check"] = df_concat["stat_repl"] > df_concat["stat"]
        return df_concat.groupby("question")["check"].mean()


def user_ic(post_sample, user_answers):
    ps = post_sample.p.sel(question=user_answers["question_order"].values).values
    rep = user_answers[[0, 1, 2, 3]].values
    ics = -np.log((rep * ps).sum(axis=1))
    return ics.sum()


def ph_stat(post_sample, answers_df):
    ics = []

    for user in answers_df["user_id"].unique():
        ics.append(user_ic(post_sample, answers_df[answers_df["user_id"] == user]))

    return max(ics) - min(ics)


def user_ic_sim(post_sample, user_answers, rng):
    ps = post_sample.p.sel(question=user_answers["question_order"].values).values
    rep = rng.multinomial(np.ones(user_answers.shape[0]).astype(int), ps)
    ics = -np.log((rep * ps).sum(axis=1))
    return ics.sum()


def ph_stat_sim(post_sample, answers_df):
    rng = np.random.default_rng()
    ics = []

    for user in answers_df["user_id"].unique():
        ics.append(
            user_ic_sim(post_sample, answers_df[answers_df["user_id"] == user], rng)
        )

    return max(ics) - min(ics)


def ph_posterior_predictive_check(posterior, df_ans):
    rows = []
    it = list(product(posterior.draw, posterior.chain))
    for draw, chain in it:
        post_samp = posterior.sel(draw=draw, chain=chain)
        t_stat = ph_stat(post_samp, df_ans)
        t_stat_rep = ph_stat_sim(post_samp, df_ans)

        rows.append(
            {
                "chain": int(chain.values),
                "draw": int(draw.values),
                "stat": t_stat,
                "stat_rep": t_stat_rep,
            }
        )
    return pd.DataFrame(rows)
