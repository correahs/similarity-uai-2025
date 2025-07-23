import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from networkx import draw_networkx_edges
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from similarity_bayes.models.bayes_models import (
    IIAModel,
    build_handcrafted_survey_model,
)
from similarity_bayes.models.freq_models import pop_hom_permutation_test, pop_hom_stat
from similarity_bayes.utils import load_handcrafted_survey, load_survey_one_hot

plt.rc("axes", labelsize=16)
matplotlib.rcParams["xtick.labelsize"] = 14
matplotlib.rcParams["ytick.labelsize"] = 14


# to be deprecated
question_pairs = [
    {
        "target": "0305",
        "main_options": ["0304", "0299"],
        "extra_options": ["0134", "0135"],
    },
    {
        "target": "0329",
        "main_options": ["0327", "0730"],
        "extra_options": ["0328", "0239"],
    },
    {
        "target": "0042",
        "main_options": ["0023", "0029"],
        "extra_options": ["0057", "0016"],
    },
    {
        "target": "0336",
        "main_options": ["0031", "0100"],
        "extra_options": ["0030", "0029"],
    },
    {
        "target": "0353",
        "main_options": ["0352", "0178"],
        "extra_options": ["0180", "0492"],
    },
    {
        "target": "0055",
        "main_options": ["0053", "0036"],
        "extra_options": ["0164", "0889"],
    },
    {
        "target": "0346",
        "main_options": ["0343", "0339"],
        "extra_options": ["0345", "0344"],
    },
    {
        "target": "0058",
        "main_options": ["0038", "0011"],
        "extra_options": ["0146", "0072"],
    },
    {
        "target": "0331",
        "main_options": ["0632", "0044"],
        "extra_options": ["0617", "0043"],
    },
    {
        "target": "0523",
        "main_options": ["0505", "0550"],
        "extra_options": ["0502", "0515"],
    },
    {
        "target": "0542",
        "main_options": ["0544", "0315"],
        "extra_options": ["0546", "0878"],
    },
    {
        "target": "0149",
        "main_options": ["0148", "0150"],
        "extra_options": ["0053", "0033"],
    },
    {
        "target": "0589",
        "main_options": ["0600", "0604"],
        "extra_options": ["0599", "0603"],
    },
    {
        "target": "0364",
        "main_options": ["0365", "0366"],
        "extra_options": ["0373", "0375"],
    },
    {
        "target": "0865",
        "main_options": ["0861", "0863"],
        "extra_options": ["0857", "0859"],
    },
    {
        "target": "0244",
        "main_options": ["0142", "0830"],
        "extra_options": ["0733", "0158"],
    },
    {
        "target": "0289",
        "main_options": ["0008", "0489"],
        "extra_options": ["0094", "0030"],
    },
    {
        "target": "0525",
        "main_options": ["0560", "0527"],
        "extra_options": ["0562", "0903"],
    },
    {
        "target": "0389",
        "main_options": ["0512", "0482"],
        "extra_options": ["0506", "0485"],
    },
]


def reduce_two_dim(embeddings, method="PCA"):
    match method:
        case "PCA":
            Transf = PCA
        case "TSNE":
            Transf = TSNE
        case _:
            raise NotImplementedError

    return Transf(n_components=2).fit_transform(embeddings)


def plot_embeddings(embeddings, graph, image_paths):
    assert embeddings.shape[1] >= 2, "needs to have at least two dimensions"
    if embeddings.shape[1] > 2:
        embeddings = reduce_two_dim(embeddings)

    x, y = embeddings[:, 0], embeddings[:, 1]

    pos = {i: (x[i], y[i]) for i in range(embeddings.shape[0])}

    scale = 0.2

    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    ax.set_zorder(1)
    draw_networkx_edges(graph, pos=pos, ax=ax, alpha=0.4, width=0.2)
    ax.set_zorder(-1)
    for x_i, y_i, path_i in zip(x, y, image_paths):
        img = Image.open(path_i)
        im = np.asarray(img.convert("RGB"))
        w = float(im.shape[1])
        h = float(im.shape[0])

        new_w = w / max(w, h)
        new_h = h / max(w, h)
        # Define the position for the image axes
        # ax_image = fig.add_axes([x_i, y_i, image_width, image_height])
        ax.imshow(
            im,
            extent=[x_i, x_i + new_w * scale, y_i, y_i + new_h * scale],
            alpha=1,
        )

        # ax_image.axis("off")

    ax.set_xlim((np.min(x) - scale, np.max(x) + scale))
    ax.set_ylim((np.min(y) - scale, np.max(y) + scale))

    fig.savefig("test.png")


def obtain_bounds_per_question_pair(handcrafted_survey_path, food_path):
    df1, df2 = load_handcrafted_survey(handcrafted_survey_path, {"index_col": 0})
    model = build_handcrafted_survey_model(
        IIAModel, df1.shape[0], max_options=4, hp_std=2
    )

    data_dict = {
        "A": df1.to_numpy().astype(int),
        "B": df2.to_numpy().astype(int),
    }
    _, trace, pred = model.fit(data_dict, draws=3000, tune=2000)

    question_pairs_dict = {
        p["target"]: (p["main_options"], p["extra_options"]) for p in question_pairs
    }

    print(df1.index)

    for i, target in enumerate(df1.index):
        counts_1 = df1.iloc[i].to_numpy()
        counts_2 = df2.iloc[i].to_numpy()

        post_pred_a1 = (
            pred.posterior_predictive.counts_A.sel(question=i, option_A=0)
            .to_numpy()
            .flatten()
        )
        post_pred_a2 = (
            pred.posterior_predictive.counts_B.sel(question=i, option_B=0)
            .to_numpy()
            .flatten()
        )
        post_pred_b1 = (
            pred.posterior_predictive.counts_A.sel(question=i, option_A=1)
            .to_numpy()
            .flatten()
        )
        post_pred_b2 = (
            pred.posterior_predictive.counts_B.sel(question=i, option_B=1)
            .to_numpy()
            .flatten()
        )

        (a, b), (c, d) = question_pairs_dict[target]

        fig = plot_question_pair_and_preds(
            f"{food_path}{target}.jpg",
            f"{food_path}{a}.jpg",
            f"{food_path}{b}.jpg",
            f"{food_path}{c}.jpg",
            f"{food_path}{d}.jpg",
            counts_1,
            counts_2,
            post_pred_a1,
            post_pred_a2,
            post_pred_b1,
            post_pred_b2,
        )
        fig.savefig(f"handcrated_example_{target}.png")


def plot_question_pair_and_preds(
    t,
    a,
    b,
    c,
    d,
    counts_1,
    counts_2,
    post_pred_a1,
    post_pred_a2,
    post_pred_b1,
    post_pred_b2,
):
    # plot it
    fig = plt.figure(figsize=(10, 7), dpi=100)

    gs = fig.add_gridspec(6, 8, wspace=1, hspace=1)

    t_ax = fig.add_subplot(gs[2:4, 0:2])
    a_ax = fig.add_subplot(gs[1:3, 2:4])
    b_ax = fig.add_subplot(gs[1:3, 4:6])
    c_ax = fig.add_subplot(gs[1:3, 6:])

    a2_ax = fig.add_subplot(gs[3:5, 2:4])
    b2_ax = fig.add_subplot(gs[3:5, 4:6])
    d_ax = fig.add_subplot(gs[3:5, 6:])

    a_pred_ax = fig.add_subplot(gs[0, 2:4])
    b_pred_ax = fig.add_subplot(gs[0, 4:6], sharey=a_pred_ax)

    a2_pred_ax = fig.add_subplot(gs[5, 2:4], sharex=a_pred_ax)
    b2_pred_ax = fig.add_subplot(gs[5, 4:6], sharex=b_pred_ax, sharey=a2_pred_ax)

    picture_axes = [t_ax, a_ax, c_ax, b_ax, a2_ax, d_ax, b2_ax]
    for ax in picture_axes:
        ax.set_xticks([])
        ax.set_yticks([])

    t_ax.set_title("target", fontsize=14)
    a_ax.set_title("$c_1$", fontsize=14)
    b_ax.set_title("$c_2$", fontsize=14)
    c_ax.set_title("$c_3$", fontsize=14)
    a2_ax.set_title("$c_1$", fontsize=14)
    b2_ax.set_title("$c_2$", fontsize=14)
    d_ax.set_title("$c_4$", fontsize=14)

    im_t, im_a, im_b, im_c, im_d = [Image.open(item) for item in [t, a, b, c, d]]
    t_ax.imshow(im_t)
    a_ax.imshow(im_a)
    a2_ax.imshow(im_a)
    b_ax.imshow(im_b)
    b2_ax.imshow(im_b)
    c_ax.imshow(im_c)
    d_ax.imshow(im_d)

    a_pred_ax.axvline(counts_1[0], c="red")
    sns.kdeplot(post_pred_a1, ax=a_pred_ax)
    a_pred_ax.set_yticks([])
    b_pred_ax.axvline(counts_1[1], c="red")
    sns.kdeplot(post_pred_b1, ax=b_pred_ax)
    b_pred_ax.set_ylabel("")
    b_pred_ax.set_yticks([])
    a2_pred_ax.axvline(counts_2[0], c="red")
    sns.kdeplot(post_pred_a2, ax=a2_pred_ax)
    a2_pred_ax.set_yticks([])
    b2_pred_ax.axvline(counts_2[1], c="red")
    sns.kdeplot(post_pred_b2, ax=b2_pred_ax)
    b2_pred_ax.set_ylabel("")
    b2_pred_ax.set_yticks([])

    return fig


def plot_population_homogeneity(random_survey_path):
    df_answers = load_survey_one_hot(random_survey_path, {"index_col": 0})
    ics = pop_hom_stat(df_answers, list(range(4)))
    #  stat, stats = pop_hom_permutation_test(df_answers, 3, 1000)
    #  stats = np.array(stats)
    #  print(stat, (stats > stat).mean())

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.set_xlabel("$I_p$ for participants $p$")

    sns.histplot(ics, ax=ax, bins=25, kde=True)
    sns.despine(trim=True)

    fig.savefig("ic_distribution.png")


def plot_population_homogeneity_2(handcrafted_survey_path):
    df_answers = load_survey_one_hot(handcrafted_survey_path, {"index_col": 0})
    ics = pop_hom_stat(df_answers, list(range(3)))
    stat, stats = pop_hom_permutation_test(df_answers, 3, 1000)
    stats = np.array(stats)
    print(stat, (stats > stat).mean())
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.set_xlabel("$I_p$ for participants $p$")

    sns.histplot(ics, ax=ax, bins=25, kde=True)
    sns.despine(trim=True)

    fig.savefig("ic_distribution_handcrafted.png")
