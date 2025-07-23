import random
import sys
from itertools import groupby, product
from pickletools import optimize

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

from similarity_bayes.db.create_db import create_and_build_db
from similarity_bayes.models.freq_models import _goodness_of_fit_stat
from similarity_bayes.models.metric_models import TSTEModel, FBOModel
from similarity_bayes.plots import plot_embeddings
from similarity_bayes.synthetic_data import (
    metric_choices_irregular_sets_simulate,
    metric_choices_leave_one_out_simulate,
)
from similarity_bayes.utils import pad_str, param_grid
from similarity_bayes.utils.data_types import (
    SimilarityChoiceCounts,
    choice_counts_from_similarity,
    graph_from_similarity,
)
from similarity_bayes.experiments import fit_additive_to_simple_context
from similarity_bayes.experiments import food100_experiment

# random surveys
# 165adf1d-fce2-4683-939d-f2c89ac166c6
# ee6c9e98-077c-4241-9cfc-80ba0a541cb4

logo = (
    "                                                      \n"
    "                 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 \n"
    "              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              \n"
    "          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          \n"
    "        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        \n"
    "      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      \n"
    "     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     \n"
    "    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    \n"
    "   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  \n"
    "  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  \n"
    " ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ \n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓                          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓           ▓▓▓▓           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓           ▓▓▓▓           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓            ▓▓            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓            ▓▓            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓  ▓▓  ▓▓▓▓    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓  ▓▓  ▓▓▓▓    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n"
    " ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ▓▓   ▓▓   ▓▓   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ \n"
    "  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ▓▓   ▓▓   ▓▓   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  \n"
    "  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  \n"
    "    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓    \n"
    "      ▓▓▓▓▓▓▓▓▓▓▓▓    L  I  R  A    ▓▓▓▓▓▓▓▓▓▓▓▓      \n"
    "      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        \n"
    "        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          \n"
    "            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓            \n"
    "               ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓               \n"
)


def print_logo():
    print(logo)


survey_responses_query = (
    "SELECT question_id, target, selected, n_participants "
    "FROM question_response_view "
    "WHERE survey_name in ("
    "'165adf1d-fce2-4683-939d-f2c89ac166c6', "
    "'ee6c9e98-077c-4241-9cfc-80ba0a541cb4')"
)

image_paths_query = "select id, image_path from item"


def image_path_dict(db_manager):
    db_manager._cursor.execute(image_paths_query)
    fetched = db_manager._cursor.fetchall()

    ret_dict = dict()
    for item, path in fetched:
        ret_dict[item] = path

    return ret_dict


def db_results_to_similarity(fetched):
    item_map = dict()
    count = 0

    dataset = []

    for question_id, data in groupby(fetched, lambda r: r[0]):
        _, target, selected, counts = zip(*data)
        items = target[:1] + selected
        m_items = []
        for i in items:
            m_i = item_map.get(i, None)
            if m_i is None:
                m_i = count
                count += 1
                item_map[i] = m_i
            m_items.append(m_i)
        c = SimilarityChoiceCounts(m_items[0], m_items[1:], list(counts))
        dataset.append(c)

    return dataset, item_map


def train_test_split(dataset, perc=0.2):
    n_train = int((1 - perc) * len(dataset))
    copied = dataset.copy()
    random.shuffle(copied)

    return copied[:n_train], copied[n_train:]


def eval_model(train, test, n_items, model_type, param_dict):
    model = model_type(n_items, **param_dict)
    loss, _ = model.train(train)
    expected_train = model.predict([c.to_input() for c in train])
    train_acc = model.accuracy(train)
    train_err = model.error([c.choice_counts for c in train], expected_train)
    train_err_norm = train_err / len(train)

    print("model", model_type, "params", param_dict)

    print("train: Chi^2", train_err, "Chi^2 normalized", train_err_norm)
    print("-" * 20)

    expected_test = model.predict([c.to_input() for c in test])

    test_err = model.error([c.choice_counts for c in test], expected_test)
    test_err_norm = test_err / len(test)
    test_nll = model.negative_ll(test)
    test_acc = model.accuracy(test)

    print("test: Chi^2", test_err, "Chi^2 norm", test_err_norm)
    print("-" * 20)

    print("\n" * 10)
    return (
        {
            "n_items": n_items,
            "n_questions": len(train) + len(test),
            **param_dict,
            "train_loss": loss[-1],
            "train_acc": train_acc,
            "train_err": train_err,
            "train_n": len(train),
            "train_err_norm": train_err_norm,
            "test_err": test_err,
            "test_acc": test_acc,
            "test_loss": test_nll,
            "test_n": len(test),
            "test_err_norm": test_err_norm,
        },
        model._emb_extractor()[:-1],
        loss,
    )


def _partial_create():
    db_manager = create_and_build_db(
        "../raw_data/", "../raw_data/All_Foodpictures_information.csv"
    )
    return db_manager


# def compare_metric_to_freq():
#
#     freq_kwargs = {
#         "step_size": 0.005,
#         "stopping_delta": 0.0001,
#         "max_iter": 10000,
#     }
#     metric_kwargs = {
#         "n_dims": 3,
#         "train_with_triplets": True,
#         "l2_reg": 0.01,
#     }
#     db_manager = _partial_create()
#     db_manager._cursor.execute(survey_responses_query)
#     fetched = db_manager._cursor.fetchall()
#
#     dataset, item_map = db_results_to_similarity(fetched)
#     choice_counts_list = choice_counts_from_similarity(dataset)
#
#     info = []
#
#     for choice_counts in choice_counts_list:
#         dgf, n_params, stat = _goodness_of_fit_stat(choice_counts, freq_kwargs)
#         for index in choice_counts.question_indices:
#             info.append(
#                 {"question": index, "dgf": dgf, "n_params": n_params, "stat": stat}
#             )
#
#     model = TSTEModel(len(item_map), **metric_kwargs)
#     loss, _ = model.train(dataset)
#     expected = model.predict([c.to_input() for c in dataset])
#
#     print("train: Chi^2", train_err, "Chi^2 normalized", train_err_norm)
#     print("-" * 20)
#
#     expected_test = model.predict([c.to_input() for c in test])
#
#     test_err = model.error([c.choice_counts for c in test], expected_test)
#     test_err_norm = test_err / len(test)
#
#     print("test: Chi^2", test_err, "Chi^2 normalized", test_err_norm)
#     print("-" * 20)
#
#     print("\n" * 10)
#
#     breakpoint()


def run_additive_on_simple_context():
    df = fit_additive_to_simple_context()
    df.to_csv("additive_on_multiplicative.csv")


def food100_tste_embeddings():
    dataset = food100_experiment("../food100")

    params = dict(n_dims=20, l2_reg=0.01, max_iter=500, df=3, beta=0.1)
    train, test = train_test_split(dataset)

    tr = eval_model(train, test, 100, FBOModel, params)
    learned_embeddings = tr[1]

    print(tr[0])

    with open("learned_emb.pkl", "wb") as f:
        pickle.dump(learned_embeddings, f)

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


def run_on_food100_data():
    dataset = food100_experiment("../food100")

    n_runs_per_config = 5

    n_dims = [20]
    l2 = [0.01, 0.05, 0.1, 0.2]
    beta = [1.0, 0.8, 0.5, 0.1]

    fbo_params = param_grid(
        n_dims=n_dims,
        l2_reg=l2,
        beta=beta,
        max_iter=[500],
    )
    fbo_res = []
    for params in fbo_params:
        for n in range(n_runs_per_config):
            train, test = train_test_split(dataset)
            tr = eval_model(train, test, 100, FBOModel, params)
            fbo_res.append({**(tr[0]), "run": n})

        df_results = pd.DataFrame(fbo_res)
        df_results.to_csv("fbo_food1002.csv")

    print("\n" * 100)
    print("-" * 50, "\n")

    print(
        df_results[
            [
                "n_items",
                "n_dims",
                "n_questions",
                "train_err_norm",
                "test_err_norm",
            ]
        ]
    )

    print("-" * 50, "\n")


def run_on_real_data():
    db_manager = _partial_create()
    db_manager._cursor.execute(survey_responses_query)
    fetched = db_manager._cursor.fetchall()

    dataset, item_map = db_results_to_similarity(fetched)

    n_runs_per_config = 5

    n_dims = [2, 3, 4, 5, 6]
    l2 = [0.01, 0.05, 0.1, 0.2, 0.3]
    beta = [1.0, 0.8, 0.5, 0.1]

    # tste_params = param_grid(
    #     n_dims=n_dims,
    #     l2_reg=l2,
    #     max_iter=[10000],
    # )

    # tste_res = []
    # for params in tste_params:
    #     for n in range(n_runs_per_config):
    #         train, test = train_test_split(dataset)
    #         tr = eval_model(train, test, len(item_map), TSTEModel, params)
    #         tste_res.append({**(tr[0]), "run": n})

    # df_results = pd.DataFrame(tste_res)
    # df_results.to_csv("tste_real.csv")

    fbo_params = param_grid(
        n_dims=n_dims,
        l2_reg=l2,
        beta=beta,
        max_iter=[6000],
    )
    fbo_res = []
    for params in fbo_params:
        for n in range(n_runs_per_config):
            train, test = train_test_split(dataset)
            tr = eval_model(train, test, len(item_map), FBOModel, params)
            fbo_res.append({**(tr[0]), "run": n})

        df_results = pd.DataFrame(fbo_res)
        df_results.to_csv("fbo_real.csv")

    print("\n" * 100)
    print("-" * 50, "\n")

    print(
        df_results[
            [
                "n_items",
                "n_dims",
                "n_questions",
                "train_err_norm",
                "test_err_norm",
            ]
        ]
    )

    print("-" * 50, "\n")


def sim_dataset(n_items, n_questions, n_dims_gt, sim_method):
    # Fixed Data params
    n_participants = 100
    std_emb = 0.5 if sim_method == "cov_weighted" else 1.0
    max_options = 4

    dataset, emb = metric_choices_leave_one_out_simulate(
        n_participants,
        n_questions,
        n_items,
        n_dims_gt,
        std_emb,
        max_options,
        sim_method,
    )
    train, test = train_test_split(dataset)

    return train, test, emb


def simulation(train, test, n_dims_gt, n_items, param_dict, model):
    ret_eval, _, _ = eval_model(train, test, n_items, model, param_dict)

    return {
        **ret_eval,
        "n_dims_gt": n_dims_gt,
    }


def mirror_pca(X):
    if min(X[:, 0]) < -max(X[:, 0]):
        X[:, 0] *= -1
    if min(X[:, 1]) < -max(X[:, 1]):
        X[:, 1] *= -1

    return X


def tste_converge():
    n_items = 20
    n_questions = 2000
    n_dims = 5

    l2_reg = 0.01

    train, test, emb = sim_dataset(n_items, n_questions, n_dims, "cov_weighted")

    orig_dict = {"n_dims": n_dims, "l2_reg": l2_reg, "max_iter": 1000}

    result_im = simulation(train, test, n_dims, n_items, orig_dict, FBOModel)

    pd.DataFrame([result_im]).to_csv("im_data_model_2.csv")

    tste_results = []
    for n_dims_model in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        param_dict = orig_dict.copy()
        param_dict["max_iter"] = 3500
        param_dict["n_dims"] = n_dims_model
        result_tste = simulation(train, test, n_dims, n_items, param_dict, TSTEModel)
        tste_results.append(result_tste)

    pd.DataFrame(tste_results).to_csv("im_data_tste_conv.csv")


def identification_compare():
    n_items = 20
    n_questions = 2000
    n_dims = 5

    l2_reg = 0.01

    train, test, emb = sim_dataset(n_items, n_questions, n_dims, "cov_weighted")

    ret_eval, learned_emb, loss = eval_model(
        train,
        test,
        n_items,
        FBOModel,
        {"n_dims": n_dims, "l2_reg": l2_reg, "max_iter": 1000},
    )
    emb_pca = mirror_pca(PCA().fit_transform(emb))
    learned_pca = mirror_pca(PCA().fit_transform(learned_emb))

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    sns.scatterplot(
        x=emb_pca[:, 0],
        y=emb_pca[:, 1],
        hue=np.arange(len(emb_pca)),  # Color by point order
        size=2,  # Optional: adjust point size
        ax=ax[0],
    )
    ax[0].set_title("Original embeddings - PCA")
    sns.scatterplot(
        x=learned_pca[:, 0],
        y=learned_pca[:, 1],
        hue=np.arange(len(learned_pca)),  # Color by point order
        size=2,  # Optional: adjust point size
        ax=ax[1],
    )
    ax[1].set_title("Learned embeddings - PCA")
    sns.lineplot(loss, ax=ax[2])
    ax[2].set_title("Training loss curve")
    breakpoint()
    plt.show()


def compare_models():
    n_items = [20]
    n_questions = [2000]
    n_dims_gt = [5]
    n_dims_model = [3, 5, 7]
    l2_reg = [0.01, 0.1]

    n_runs_per_config = 5

    # params = param_grid(
    #     n_dims=n_dims_model,
    #     # train_with_triplets=train_with_triplets,
    #     l2_reg=l2_reg,
    #     max_iter=[2500],
    # )

    # tste_results = []
    # im_results = []
    # for i, q, dgt, param_dict in product(n_items, n_questions, n_dims_gt, params):
    #     print("Running TSTE and CW with params", param_dict)
    #     for n in range(n_runs_per_config):
    #         train, test, _ = sim_dataset(i, q, dgt, "metric")
    #         result_tste = simulation(train, test, dgt, i, param_dict, TSTEModel)
    #         result_im = simulation(train, test, dgt, i, param_dict, InvMahalanobisModel)

    #         tste_results.append({**result_tste, "run": n})
    #         im_results.append({**result_im, "run": n})

    # df_results = pd.DataFrame(tste_results)
    # df_results.to_csv("tste_data_tste_model.csv")

    # df_results = pd.DataFrame(im_results)
    # df_results.to_csv("tste_data_im_model.csv")

    # print("\n" * 100)
    # print("-" * 50, "\n")

    # print(
    #     df_results[
    #         [
    #             "n_items",
    #             "n_dims_model",
    #             "n_questions",
    #             "train_err_norm",
    #             "test_err_norm",
    #         ]
    #     ]
    # )

    # print("-" * 50, "\n")

    params = param_grid(
        n_dims=n_dims_model,
        # train_with_triplets=train_with_triplets,
        l2_reg=l2_reg,
        max_iter=[2500],
    )

    tste_results = []
    im_results = []
    for i, q, dgt, param_dict in product(n_items, n_questions, n_dims_gt, params):
        param_dict_cp = param_dict.copy()
        param_dict_cp["max_iter"] = 1500
        for n in range(n_runs_per_config):
            train, test, _ = sim_dataset(i, q, dgt, "cov_weighted")
            result_tste = simulation(train, test, dgt, i, param_dict, TSTEModel)
            result_im = simulation(train, test, dgt, i, param_dict_cp, FBOModel)

            tste_results.append({**result_tste, "run": n})
            im_results.append({**result_im, "run": n})

    df_results = pd.DataFrame(tste_results)
    df_results.to_csv("im_data_tste_model.csv")

    df_results = pd.DataFrame(im_results)
    df_results.to_csv("im_data_im_model.csv")

    print("\n" * 100)
    print("-" * 50, "\n")

    print(
        df_results[
            [
                "n_items",
                "n_dims_model",
                "n_questions",
                "train_err_norm",
                "test_err_norm",
            ]
        ]
    )

    print("-" * 50, "\n")


def run_on_simulated_data():
    #  n_items = [10, 25, 50, 75, 100]
    #  n_questions = [100, 200]
    #  n_dims_gt = [10]
    #  n_dims_model = [2, 3, 5, 8, 13, 21]
    n_items = [10, 20]
    n_questions = [100]
    n_dims_gt = [10]
    n_dims_model = [2, 3, 5, 8]
    l2_reg = [0.0, 0.01, 0.1]

    params = param_grid(
        n_dims=n_dims_model,
        # train_with_triplets=train_with_triplets,
        l2_reg=l2_reg,
    )

    n_runs_per_config = 10

    simulation_results = []
    for i, q, dgt, param_dict in product(n_items, n_questions, n_dims_gt, params):
        for n in range(n_runs_per_config):
            result = simulation(i, q, dgt, param_dict, "metric", TSTEModel)
            simulation_results.append({**result, "run": n})

    df_results = pd.DataFrame(simulation_results)
    df_results.to_csv("tste_simulation_test_inv_mahala.csv")

    print("\n" * 100)
    print("-" * 50, "\n")

    print(
        df_results[
            [
                "n_items",
                "n_dims_model",
                "n_questions",
                "train_err_norm",
                "test_err_norm",
            ]
        ]
    )

    print("-" * 50, "\n")


def generate_and_plot_embeddings():
    db_manager = _partial_create()
    db_manager._cursor.execute(survey_responses_query)
    fetched_responses = db_manager._cursor.fetchall()
    item_images = image_path_dict(db_manager)

    dataset, item_map = db_results_to_similarity(fetched_responses)
    graph = graph_from_similarity(dataset)

    model = TSTEModel(len(item_map), 3, max_iter=10000)
    loss, expected_train = model.train(dataset)

    embeddings = model._emb_extractor()
    index_to_id = {v: k for k, v in item_map.items()}

    images = [item_images[index_to_id[i]] for i in range(len(item_map))]

    plot_embeddings(embeddings, graph, images)


def wrong_input_exit(flags, message):
    print(message, file=sys.stderr)
    max_len = max([len(k) for k in flags.keys()])
    print("accepted flags are:", file=sys.stderr)
    for flag, info in flags.items():
        print(f"\t{pad_str(flag, ' ', max_len)} :=   {info[0]}")
    sys.exit()


if __name__ == "__main__":
    print_logo()
    args = sys.argv

    flags = {
        "--sim": ("Run on simulated data", run_on_simulated_data),
        "--real": ("Run on survey data", run_on_real_data),
        "--food100": ("Run on food100 dataset", run_on_food100_data),
        "--db": ("Populate db", _partial_create),
        "--emb": ("Generate plot with food pictures", generate_and_plot_embeddings),
        "--comp": ("Compare models", compare_models),
        "--ind": ("Check match", identification_compare),
        "--conv": ("converge TSTE", tste_converge),
        # "--comp": ("Compare MNL to TSTE model", compare_metric_to_freq),
        "--S723": ("Question from s723", run_additive_on_simple_context),
        "--exp": ("Export embeddings", food100_tste_embeddings),
    }
    if len(args) <= 1 or len(args) > 2:
        wrong_input_exit(flags, "ERROR, accepts exactly one flag")

    info = flags.get(args[1], None)
    if info is not None:
        desc, fun = info
        print(f"performing {args[1]}: {desc}")
        fun()
    else:
        wrong_input_exit(flags, "ERROR, unrecognized flag")
