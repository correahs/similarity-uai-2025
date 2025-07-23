from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from typing import Self

import numpy as np
from networkx import Graph

from similarity_bayes.utils import range_without


@dataclass
class SimilarityQuestionInput:
    target: int
    choice_set: list[int]
    n_participants: int


@dataclass
class SimilarityChoiceCounts:
    target: int
    choice_set: list[int]
    choice_counts: list[int]

    def to_input(self: Self) -> SimilarityQuestionInput:
        return SimilarityQuestionInput(
            self.target, self.choice_set, sum(self.choice_counts)
        )

    def to_triplets(self: Self) -> list[Self]:
        triplets = []
        for i, j in combinations(range(len(self.choice_set)), 2):
            if self.choice_counts[i] > 0 or self.choice_counts[j] > 0:
                triplets.append(
                    SimilarityChoiceCounts(
                        self.target,
                        [self.choice_set[i], self.choice_set[j]],
                        [self.choice_counts[i], self.choice_counts[j]],
                    ),
                )
        return triplets


def graph_from_similarity(questions: list[SimilarityChoiceCounts]) -> Graph:

    graph = Graph()

    for q in questions:
        if not graph.has_node(q.target):
            graph.add_node(q.target)
        for o in q.choice_set:
            if not graph.has_node(o):
                graph.add_node(o)
            graph.add_edge(q.target, o)

    return graph


@dataclass
class ChoiceCounts:
    """Models that use this class consider each item to have an
    independent relationship to the target, making it such that
    each target is dealt with by a distinct choice model"""

    n_items: int
    masks_and_counts: list[tuple[np.ndarray, np.ndarray]]
    question_indices: list[int] | None = None

    @classmethod
    def from_full_and_leave_one_out_questions(
        cls,
        full_question: np.ndarray,
        leave_one_out_questions: list[np.ndarray],
    ) -> Self:
        n_items = full_question.shape[0]
        assert n_items == len(leave_one_out_questions)

        total_counts = np.zeros(n_items)
        masks_and_counts = []
        masks_and_counts.append((np.arange(n_items), full_question))
        for i, le_question in enumerate(leave_one_out_questions):
            mask = np.array(range_without(n_items, i))
            total_counts[mask] += le_question
            masks_and_counts.append((mask, le_question))
        return cls(n_items, masks_and_counts)

    @classmethod
    def from_handcrafted_survey(
        cls, a_questions: np.ndarray, b_questions: np.ndarray
    ) -> Self:
        assert a_questions.shape == b_questions.shape
        n_items = a_questions.shape[0] + 1
        mask_a = np.array(list(range(n_items - 1)))
        mask_b = mask_a.copy()
        mask_b[-1] += 1

        return cls(n_items, [(mask_a, a_questions), (mask_b, b_questions)])

    def exclude_items(self, to_exclude: list[int]) -> Self:
        chosen_items = np.setdiff1d(np.arange(self.n_items), to_exclude)
        shift_indices = np.vectorize({k: i for i, k in enumerate(chosen_items)}.get)
        clean_masks_and_counts = []
        for mask, count in self.masks_and_counts:
            selected_indices = np.flatnonzero(np.isin(mask, chosen_items))
            clean_masks_and_counts.append(
                (
                    shift_indices(mask[selected_indices]),
                    count[selected_indices],
                )
            )
        return type(self)(chosen_items.shape[0], clean_masks_and_counts)

    def without_unselected_options(self) -> Self:
        total_counts = np.zeros(self.n_items)
        for mask, count in self.masks_and_counts:
            total_counts[mask] += count

        return self.exclude_items(np.flatnonzero(total_counts == 0).tolist())


def _dag_per_target(questions: list[SimilarityChoiceCounts], indices: list[int]):
    """Build a Dag with the relationship that a -> b if choice-set of b is contained
    in choice-set of a.
    Each connected component will be returned as a separate ChoiceCounts object.
    """

    # populate components
    components = []
    sorted_indices = sorted(
        indices, key=lambda i: len(questions[i].choice_set), reverse=True
    )
    for i in sorted_indices:
        q = questions[i]
        found = False
        for comp in components:
            for _, qc in comp:
                if set(q.choice_set) <= set(qc.choice_set):
                    comp.append((i, q))
                    found = True
                    break
        if not found:
            components.append([(i, q)])
    choice_counts_list = []
    for comp in components:
        # make map to dense indexing
        item_counter = 0
        item_map = {}

        for _, q in comp:
            for c in q.choice_set:
                if c not in item_map:
                    item_map[c] = item_counter
                    item_counter += 1

        # create ChoiceCounts
        choice_counts = ChoiceCounts(
            item_counter,
            [
                (
                    np.array([item_map[item] for item in q.choice_set]),
                    np.array(q.choice_counts),
                )
                for _, q in comp
            ],
            [t[0] for t in comp],
        )
        choice_counts_list.append(choice_counts)
    return choice_counts_list


def choice_counts_from_similarity(
    questions: list[SimilarityChoiceCounts],
) -> list[ChoiceCounts]:
    # make DAG for each target?

    per_target = {}

    for i, q in enumerate(questions):
        if q.target in per_target:
            per_target[q.target].append(i)
        else:
            per_target[q.target] = [i]

    return reduce(
        list.__add__,
        [_dag_per_target(questions, indices) for indices in per_target.values()],
    )
