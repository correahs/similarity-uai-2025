import pandas as pd
import numpy as np
import json
from functools import reduce


def similarity_tuple_generator(
    json_datasets: dict[str, str],
    expand_to_triplets=False,
    skip_multiple_selection=False,
    keep_near=False,
):
    """
    Yield all triplets that we could possibly infer
    """
    for json_path in json_datasets:
        with open(json_path, "r") as f:
            json_doc = json.load(f)
            for hit in json_doc:
                for screen in hit["HIT_screens"]:
                    if not screen["is_catchtrial"]:
                        all_images = set(screen["images"])
                        near = set(screen["near_answer"])
                        far = all_images - near

                        if skip_multiple_selection and len(near) > 1:
                            continue

                        a = screen["probe"]
                        for b in near:
                            if keep_near:
                                far_ = far | (near - {b})
                            else:
                                far_ = far
                            if expand_to_triplets:
                                for c in far_:
                                    yield (a, b, c)
                            else:
                                yield (a, b) + tuple(far_)


def consolidate(tuples):
    labels = set(reduce(tuple.__add__, tuples))
    label_map = dict(zip(labels, range(len(labels))))
    counts = dict()
    for qset in tuples:
        target = label_map[qset[0]]
        choice_set = [label_map[f] for f in qset[1:]]
        indices = sorted(range(len(choice_set)), key=choice_set.__getitem__)
        selected = np.zeros(len(choice_set))
        selected[0] = 1
        key = (target, tuple([choice_set[i] for i in indices]))
        if key in counts:
            counts[key] += selected[indices]
        else:
            counts[key] = selected[indices]
        # breakpoint()
    return counts


def choice_triplets_df(json_datasets: dict[str, str]) -> pd.DataFrame:
    rows = []
    for choice in similarity_tuple_generator(json_datasets, expand_to_triplets=True):
        rows.append({"reference": choice[0], "selected": choice[1], "rest": choice[2]})
    df = pd.DataFrame(rows)
    return df


def choice_tuples_df(
    json_datasets: dict[str, str],
    skip_multiple_selection=False,
    keep_near=False,
) -> pd.DataFrame:
    rows = []
    for choice in similarity_tuple_generator(
        json_datasets,
        skip_multiple_selection=skip_multiple_selection,
        keep_near=keep_near,
    ):
        rows.append(
            {
                "reference": choice[0],
                "selected": choice[1],
                **{f"rest_{i}": image for i, image in enumerate(choice[2:])},
            }
        )
    df = pd.DataFrame(rows)
    return df
