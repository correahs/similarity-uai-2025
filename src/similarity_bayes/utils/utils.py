from collections.abc import Iterator
from itertools import product
from typing import Any

from numpy import argwhere
from numpy import exp as expfunc
from numpy import float_, int_
from numpy.typing import NDArray
from pandas import DataFrame, concat, get_dummies, read_csv


def pad[T](seq: list[T], value: T, desired_len: int) -> list[T]:
    # craving some good pad-thai
    assert len(seq) <= desired_len
    return seq + ([value] * (desired_len - len(seq)))


def pad_str(s: str, char: str, desired_len: int) -> str:
    assert len(s) <= desired_len
    return s + char * (desired_len - len(s))


def softmax_1d(arr: NDArray[float_], add_c_term: bool = False) -> NDArray[float_]:
    exp_ = expfunc(arr)
    if add_c_term:
        return exp_ / (1 + exp_.sum())  # type: ignore
    else:
        return exp_ / exp_.sum()  # type: ignore


def softmax_func(arr: NDArray[float_]) -> Any:
    exp_ = expfunc(arr)
    return exp_ / (exp_.sum(axis=1).reshape(-1, 1))


def range_without(n_elements: int, i_to_remove: int) -> list[int]:
    l_range = list(range(n_elements))
    del l_range[i_to_remove]

    return l_range


def remove_if_exists(seq: list[int], el: int) -> None:
    found = seq.index(el)
    if found >= 0:
        del seq[found]


def target_from_id(id: str) -> str:
    return id[1:5]


def choice_set_from_id(id: str, n_options: int = 4) -> list[str]:
    items = [id[9:13]] + [id[17 + (8 * i) : 21 + (8 * i)] for i in range(n_options - 1)]
    return items


def align_datasets(
    sel_4_opt: DataFrame, sel_3_opt: DataFrame
) -> tuple[DataFrame, list[DataFrame]]:
    valid_count = 0

    matched_4_opt = []
    matched_3_opt: list[list[Any]] = [[], [], [], []]

    for qid, row in sel_4_opt.iterrows():
        qid = str(qid)
        target, choice_set = (
            target_from_id(qid),
            choice_set_from_id(qid),
        )
        leave_one_outs = []

        for i in range(4):
            mchoice_set = choice_set.copy()
            del mchoice_set[i]
            mid = (
                f"[{target}]-"
                f"['{mchoice_set[0]}', '{mchoice_set[1]}', '{mchoice_set[2]}']"
            )
            if mid in sel_3_opt.index:
                leave_one_outs.append(sel_3_opt.loc[mid])

                valid_count += 1
        if len(leave_one_outs) == 4:
            matched_4_opt.append(row)
            for i in range(4):
                matched_3_opt[i].append(leave_one_outs[i])

    print(valid_count, "matches")

    return DataFrame(matched_4_opt), [DataFrame(m3) for m3 in matched_3_opt]


def check_for_unselected_options(
    full_questions: DataFrame, leave_one_out_questions: list[DataFrame]
) -> NDArray[int_]:
    max_items = full_questions.shape[1]

    total_counts = full_questions.to_numpy()

    for i, le_questions in enumerate(leave_one_out_questions):
        rem_i = range_without(max_items, i)
        total_counts[:, rem_i] += le_questions.to_numpy()

    return argwhere((total_counts == 0).any(axis=1))


def load_survey_one_hot(
    survey_path: str, read_kwargs: dict[str, Any] | None = None
) -> DataFrame:
    if not read_kwargs:
        read_kwargs = dict()
    df = read_csv(survey_path, **read_kwargs)
    df_ans = concat(
        [
            df[["user_id", "question_id"]],
            get_dummies(df["selected"]).astype(int),
        ],
        axis=1,
    )
    return df_ans


def load_two_phase_survey(
    first_phase_fpath: str,
    second_phase_fpath: str,
    read_kwargs: dict[str, Any] | None = None,
) -> tuple[DataFrame, list[DataFrame]]:
    if not read_kwargs:
        read_kwargs = dict()
    df_4opt = read_csv(first_phase_fpath, **read_kwargs)
    df_3opt = read_csv(second_phase_fpath, **read_kwargs)

    selected_4_opt = (
        df_4opt.groupby(["question_id", "selected"]).size().unstack().fillna(0)
    )
    selected_3_opt = (
        df_3opt.groupby(["question_id", "selected"]).size().unstack().fillna(0)
    )

    return align_datasets(selected_4_opt, selected_3_opt)


def load_handcrafted_survey(
    survey_path: str,
    read_kwargs: dict[str, Any] | None = None,
) -> tuple[DataFrame, DataFrame]:
    if not read_kwargs:
        read_kwargs = dict()
    raw_df = read_csv(survey_path, **read_kwargs)
    df_group1 = raw_df[raw_df["group"] == 0]
    df_group2 = raw_df[raw_df["group"] == 1]
    df_counts_group1 = (
        df_group1.groupby(["question_id", "selected"]).size().unstack().fillna(0)
    )
    df_counts_group2 = (
        df_group2.groupby(["question_id", "selected"]).size().unstack().fillna(0)
    )
    common_index = df_counts_group1.index.intersection(df_counts_group2.index)
    return (
        df_counts_group1.loc[common_index],
        df_counts_group2.loc[common_index],
    )


def load_two_phase_survey_by_reply(
    first_phase_fpath: str,
    second_phase_fpath: str,
    read_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int], list[list[int]]]:
    if not read_kwargs:
        read_kwargs = dict()
    df_4opt = read_csv(first_phase_fpath, **read_kwargs)
    df_3opt = read_csv(second_phase_fpath, **read_kwargs)

    targets: list[int] = []
    choices: list[int] = []
    options_list: list[list[int]] = []

    appearance_order: dict[str, int] = dict()
    appearance_counter: int = 0

    for _, row in df_4opt.iterrows():
        qid = row["question_id"]
        target = target_from_id(qid)
        options = choice_set_from_id(qid, n_options=4)

        seen_items = [target] + options
        for i in seen_items:
            if i not in appearance_order:
                appearance_order[i] = appearance_counter
                appearance_counter += 1

        choice = options[row["selected"]]

        targets.append(appearance_order[target])
        choices.append(appearance_order[choice])
        options_list.append([appearance_order[opt] for opt in options])

    for _, row in df_3opt.iterrows():
        qid = row["question_id"]
        target = target_from_id(qid)
        options = choice_set_from_id(qid, n_options=3)

        seen_items = [target] + options
        for i in seen_items:
            if i not in appearance_order:
                appearance_order[i] = appearance_counter
                appearance_counter += 1

        choice = options[row["selected"]]

        targets.append(appearance_order[target])
        choices.append(appearance_order[choice])
        options_list.append([appearance_order[opt] for opt in options])

    return targets, choices, options_list


def param_grid(**param_lists: list) -> Iterator[dict]:
    keys, lists = zip(*param_lists.items())
    return map(lambda en: dict(zip(keys, en)), product(*lists))
