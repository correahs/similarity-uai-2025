from typing import Any, Optional

import glob
import numpy as np
from pandas import DataFrame, Series, concat, read_csv
from scipy.stats import chi2  # type: ignore
from tqdm import tqdm  # type: ignore

from similarity_bayes.db.food100 import similarity_tuple_generator, consolidate
from similarity_bayes.utils.data_types import SimilarityChoiceCounts
from similarity_bayes.models.bayes_models import (
    IIAModel,
    AdditivePerturbationModel,
    build_handcrafted_survey_model,
    build_random_survey_model,
    posterior_predictive_check,
    squared_relative_error,
)
from similarity_bayes.models.freq_models import (
    goodness_of_fit_test,
    goodness_of_fit_test_handcrafted,
    mcfadden_train_tye_tests,
)
from similarity_bayes.synthetic_data import (
    IIA_simulate,
    additive_context_simulate,
    simple_context_simulate,
)
from similarity_bayes.utils import load_handcrafted_survey, load_two_phase_survey


def model_fit_ppc(
    full_counts: np.ndarray,
    leave_one_out_counts: list[np.ndarray],
    mcmc_params: Optional[dict[str, Any]] = None,
) -> tuple[float, Series]:
    if mcmc_params is None:
        mcmc_params = {}

    model = build_random_survey_model(
        IIAModel,
        full_counts.shape[0],
        max_options=full_counts.shape[1],
        hp_std=2,
    )
    data_dict = {
        "full": full_counts,
        **{f"rem_{i}": le for i, le in enumerate(leave_one_out_counts)},
    }
    _, trace, _ = model.fit(data_dict, **mcmc_params)

    agg_pval: float = posterior_predictive_check(  # type: ignore
        full_counts.shape[0],
        data_dict,
        trace.posterior,
        squared_relative_error,
        aggregate_pvals=True,
    )
    pvals: Series = posterior_predictive_check(  # type: ignore
        full_counts.shape[0],
        data_dict,
        trace.posterior,
        squared_relative_error,
        aggregate_pvals=False,
    )
    return agg_pval, pvals


def p_values_comparison(
    full_counts: np.ndarray, leave_one_out_counts: list[np.ndarray]
) -> tuple[DataFrame, dict[str, float]]:
    optimize_kwargs = {
        "step_size": 0.005,
        "stopping_delta": 0.0001,
        "max_iter": 10000,
    }
    mtt_results = DataFrame(
        mcfadden_train_tye_tests(
            full_counts, leave_one_out_counts, optimize_kwargs=optimize_kwargs
        )
    )
    gft_results = DataFrame(
        goodness_of_fit_test(
            full_counts, leave_one_out_counts, optimize_kwargs=optimize_kwargs
        )
    )
    ppc_pval, ppc_pvals = model_fit_ppc(
        full_counts,
        leave_one_out_counts,
        mcmc_params={"draws": 3000, "tune": 2000},
    )
    mtt_res_first = (
        mtt_results.sort_values("rem")
        .drop_duplicates("question_set")
        .set_index("question_set")
        .sort_index()
    )
    mtt_pvals = mtt_res_first["p-val"]
    mtt_stats = mtt_res_first["stat"]
    mtt_dgfs = mtt_res_first["dgf"]
    gft_pvals = gft_results["p-val"]
    gft_stats = gft_results["stat"]
    gft_dgfs = gft_results["dgf"]
    pvals = ppc_pvals.to_frame("ppc-pval")
    pvals["mtt-pval"] = mtt_pvals
    pvals["mtt-stat"] = mtt_stats
    pvals["mtt-gdf"] = mtt_dgfs
    pvals["gft-pval"] = gft_pvals
    pvals["gft-stat"] = gft_stats
    pvals["gft-dgf"] = gft_dgfs
    gft_pval = chi2.sf(gft_stats.sum(), df=gft_dgfs.sum())
    mtt_pval = chi2.sf(mtt_stats.sum(), df=mtt_dgfs.sum())
    agg_pvals = {
        "min_ppc": ppc_pvals.min(),
        "min_mtt": mtt_pvals.min(),
        "min_gft": gft_pvals.min(),
        "agg_ppc": ppc_pval,
        "agg_mtt": mtt_pval,
        "agg_gft": gft_pval,
    }
    return pvals, agg_pvals


def goodness_of_fit_type1_error():
    optimize_kwargs = {
        "step_size": 0.01,
        "stopping_delta": 0.0001,
        "max_iter": 1000,
    }
    rows = []
    for i in tqdm(range(10, 50)):
        for _ in range(2):
            m4, m3 = IIA_simulate(i, 100, 2, max_options=4)
            gft_results = goodness_of_fit_test(
                m4,
                m3,
                optimize_kwargs=optimize_kwargs,
            )
            stat, dgf, n_params = 0.0, 0, 0
            for res in gft_results:
                stat += res["stat"]
                dgf += res["dgf"]
                n_params += res["n_params"]
            rows.append(
                {
                    "n": i,
                    "stat": stat,
                    "dgf": dgf,
                    "n_params": n_params,
                    "p-val-gdf-u": chi2.sf(stat, df=dgf),
                    "p-val-gdf-l": chi2.sf(stat, df=dgf - n_params),
                }
            )
    return DataFrame(rows)


def sweep_additive_context():
    rows = []
    for std_c in tqdm(np.linspace(0, 1, 11)):
        for _ in range(10):
            m4, m3 = additive_context_simulate(30, 100, 2, std_c, max_options=4)
            _, agg_pvals = p_values_comparison(m4, m3)
            rows.append({"std_c": std_c, **agg_pvals})
            print(rows[-1])
    return DataFrame(rows)


def sweep_simple_context():
    rows = []
    for std_c in tqdm(np.linspace(0, 1, 11)):
        for _ in range(10):
            m4, m3 = simple_context_simulate(30, 100, 2, std_c, max_options=4)
            _, agg_pvals = p_values_comparison(m4, m3)
            rows.append({"std_c": std_c, **agg_pvals})
            print(rows[-1])
    return DataFrame(rows)


def fit_additive_to_simple_context():
    rows = []
    for std_c in [0.1, 0.2, 0.3]:
        for _ in range(10):
            m4, m3 = simple_context_simulate(30, 100, 2, std_c, max_options=4)

            model = build_random_survey_model(
                AdditivePerturbationModel,
                m4.shape[0],
                max_options=m4.shape[1],
                hp_std=2,
                hp_std_p=1,
            )
            data_dict = {
                "full": m4,
                **{f"rem_{i}": m for i, m in enumerate(m3)},
            }
            _, trace, _ = model.fit(data_dict, tune=4000, draws=8000)
            samples = trace.posterior["std_p"].values.flatten()
            lq, rq = np.quantile(samples, q=[0.025, 0.975])
            rows.append(
                {
                    "mult_std_p": std_c,
                    "p_2.5": lq,
                    "p_97.5": rq,
                    "mean": np.mean(samples),
                    "median": np.median(samples),
                }
            )
    return DataFrame(rows)


def simulate_IIA():
    rows = []
    for std_c in tqdm(range(10)):
        m4, m3 = IIA_simulate(30, 100, 2, max_options=4)
        _, agg_pvals = p_values_comparison(m4, m3)
        rows.append(agg_pvals)
        print(rows[-1])
    return DataFrame(rows)


def sweep_additive_context_store_all(std_cs, n_sims) -> DataFrame:
    dfs = []
    for std_c in tqdm(std_cs):
        for _ in range(n_sims):
            m4, m3 = additive_context_simulate(30, 100, 2, std_c, max_options=4)
            df_pvals, agg_pvals = p_values_comparison(m4, m3)
            df_pvals["std_c"] = std_c
            dfs.append(df_pvals.reset_index())
    return concat(dfs, ignore_index=True)


def sweep_simple_context_store_all(std_cs, n_sims) -> DataFrame:
    dfs = []
    for std_c in tqdm(std_cs):
        for _ in range(n_sims):
            m4, m3 = simple_context_simulate(30, 100, 2, std_c, max_options=4)
            df_pvals, agg_pvals = p_values_comparison(m4, m3)
            df_pvals["std_c"] = std_c
            dfs.append(df_pvals.reset_index())
    return concat(dfs, ignore_index=True)


def random_survey_analysis(
    first_phase_fpath: str, second_phase_fpath: str, read_kwargs=None
) -> tuple[DataFrame, dict[str, float]]:
    # Return p-values per question set, indexed by full question index?
    m4, m3 = load_two_phase_survey(first_phase_fpath, second_phase_fpath, read_kwargs)
    stats, agg_pvals = p_values_comparison(
        m4.to_numpy().astype(int), [m.to_numpy().astype(int) for m in m3]
    )
    stats.index = m4.index

    return stats, agg_pvals


def handcrafted_survey_analysis(
    survey_fpath: str, read_kwargs=None
) -> tuple[DataFrame, dict[str, float]]:
    optimize_kwargs = {
        "step_size": 0.005,
        "stopping_delta": 0.0001,
        "max_iter": 10000,
    }
    df1, df2 = load_handcrafted_survey(survey_fpath, read_kwargs)

    gft_results = DataFrame(
        goodness_of_fit_test_handcrafted(
            df1.to_numpy(), df2.to_numpy(), optimize_kwargs=optimize_kwargs
        )
    )

    gft_results.index = df1.index

    model = build_handcrafted_survey_model(
        IIAModel, df1.shape[0], max_options=4, hp_std=2
    )

    data_dict = {
        "A": df1.to_numpy().astype(int),
        "B": df2.to_numpy().astype(int),
    }
    _, trace, _ = model.fit(data_dict, draws=3000, tune=2000)

    ppc_agg_pval = posterior_predictive_check(
        df1.shape[0],
        data_dict,
        trace.posterior,
        squared_relative_error,
        aggregate_pvals=True,
    )
    ppc_pvals = posterior_predictive_check(
        df1.shape[0],
        data_dict,
        trace.posterior,
        squared_relative_error,
        aggregate_pvals=False,
    )

    gft_results["ppc-pval"] = (
        ppc_pvals.to_numpy() if isinstance(ppc_pvals, Series) else ppc_pvals
    )

    agg_pvals = {
        "min_ppc": (ppc_pvals.min() if isinstance(ppc_pvals, Series) else ppc_pvals),
        "min_gft": gft_results["p-val"].min(),
        "agg_ppc": ppc_agg_pval,
        "agg_gft": chi2.sf(gft_results["stat"].sum(), df=gft_results["dgf"].sum()),
    }
    return gft_results, agg_pvals


def food100_experiment(dir_path):
    # jsons = glob.glob(f"{dir_path}/*.json")
    # gen = similarity_tuple_generator(jsons, skip_multiple_selection=True)
    tuples = read_csv(f"{dir_path}/originally-single-choice-data.csv")
    dataset = [tuple(r[r.notnull()]) for _, r in tuples.iterrows()]
    counts = consolidate(dataset)
    return [
        SimilarityChoiceCounts(k[0], list(k[1]), v.astype(int).tolist())
        for k, v in counts.items()
    ]
