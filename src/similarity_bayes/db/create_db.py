import glob
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property, partial, reduce
from pathlib import Path
from typing import Any, Self, TypeAlias

from pandas import DataFrame, read_csv
from pandas.io.sql import get_schema  # type: ignore

from ..utils.name_generator import name_generator
from ..utils.name_generator import total_names as total_cnames

RawSurveyType: TypeAlias = list[dict[str, Any]]


ABS_PATH = "/".join(__file__.split("/")[:-1])


def pad_left(string: str, c: str, max_len: int) -> str:
    return c * (max_len - len(string)) + string


def extract_item_id(img_url: str) -> str:
    return img_url.split("/")[-1].split(".")[0]


def extract_single_info(
    data_dict: dict[str, dict[str, Any]], data_key: str
) -> tuple[Any, int]:
    assert len(data_dict) >= 1
    it = iter(data_dict.values())
    val = next(it)
    assert "data" in val and "timestamp" in val
    assert isinstance(val["timestamp"], int)
    return val["data"][data_key], val["timestamp"]


def extract_infos(
    data_dict: dict[str, dict[str, Any]], data_key: str
) -> list[tuple[Any, int]]:
    infos = []
    for val in data_dict.values():
        assert "data" in val and "timestamp" in val
        assert isinstance(val["timestamp"], int)
        infos.append((val["data"][data_key], val["timestamp"]))
    return infos


def start_and_end_timestamp_from_events(events) -> tuple[int, int]:
    timestamps = [info[1] for info in extract_infos(events, "event")]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    return min_ts, max_ts


class SimilarityDBManager:
    @dataclass(frozen=True)
    class SurveyQuestion:
        target: str
        choice_set: frozenset[str]

    @cached_property
    def _connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_fpath)

    @cached_property
    def _cursor(self) -> sqlite3.Cursor:
        return self._connection.cursor()

    @cached_property
    def _item_lookup(self) -> set[str]:
        return set()

    @cached_property
    def _question_lookup(self) -> dict[SurveyQuestion, str]:
        return dict()

    def __init__(
        self: Self,
        db_path: str | None = None,
        schema_path: str | None = None,
        images_path: str | None = None,
        features_path: str | None = None,
    ):
        self.db_fpath = f"{ABS_PATH}/similarity.db" if db_path is None else db_path
        self.schema_path = (
            f"{ABS_PATH}/db_schema.sql" if schema_path is None else schema_path
        )
        self.images_path = (
            f"{ABS_PATH}/../../../food/" if images_path is None else images_path
        )

    def get_similarity_db(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_fpath}?model=ro", uri=True)

    def _setup_db(self) -> None:
        with open(self.schema_path, "r") as f:
            sql_script = f.read()

        self._cursor.executescript(sql_script)

    def _insert_item_or_nothing(self, item: str) -> None:
        insert_item = (
            "INSERT INTO item("
            "id, short_description, long_description, image_path) "
            "VALUES (?,?,?,?)"
        )
        if item in self._item_lookup:
            return
        self._cursor.execute(
            insert_item, (item, "", "", f"{self.images_path}{item}.jpg")
        )
        self._item_lookup.add(item)

    # insert items if needed,
    # insert choice set and
    # finally insert question
    def _insert_question(self, target: str, items: list[str]) -> str:
        insert_question = "INSERT INTO question(target) VALUES(?) RETURNING id"

        for i in [target] + items:
            self._insert_item_or_nothing(i)

        self._cursor.execute(insert_question, (target,))
        question_id = self._cursor.fetchall()[0][0]

        insert_choice_set = (
            "INSERT INTO choice_set_item(question_id, item_id) VALUES"
            + ", ".join(["(?,?)"] * len(items))
            + ";"
        )

        self._cursor.execute(
            insert_choice_set,
            reduce(tuple.__add__, zip([question_id] * len(items), items)),
        )
        return question_id

    def _get_or_insert_question(self, target: str, items: list[str]) -> str:
        q = SimilarityDBManager.SurveyQuestion(target, frozenset(items))
        question_id = self._question_lookup.get(q, None)
        if question_id is None:
            question_id = self._insert_question(target, items)
            self._question_lookup[q] = question_id
        return question_id

    def _add_question_and_response(self, response, participant_id):
        response_insert = (
            "INSERT INTO participant_response("
            "question_id, participant_id, response_timestamp,"
            "selected, duration, choice_set_order"
            ") values (?, ?, ?, ?, ?, ?)"
        )
        response_data = response["data"]
        response_ts = response["timestamp"]

        target = extract_item_id(response_data["images_urls"][0])
        items = [extract_item_id(i) for i in response_data["images_urls"][1:]]

        question_id = self._get_or_insert_question(target, items)
        duration = response_data["timer"]
        selected = items[response_data["answers"].index(1)]

        self._cursor.execute(
            response_insert,
            (
                question_id,
                participant_id,
                response_ts,
                selected,
                duration,
                str(items),
            ),
        )

    def _add_participant(
        self,
        participant_dict,
        survey_id: str,
        name_iter: Iterator,
    ):
        participant_insert = (
            "INSERT INTO participant("
            "name, first_access_timestamp, "
            "total_access_duration, feedback,survey_id"
            ") VALUES(?,?,?,?,?) RETURNING id"
        )
        participant_platform_id, _ = extract_single_info(
            participant_dict["prolificID"], "id"
        )
        try:
            feedback, _ = extract_single_info(participant_dict["feedback"], "feedback")
        except Exception:
            feedback = ""

        start_timestamp, end_timestamp = start_and_end_timestamp_from_events(
            participant_dict["surveyEvents"]
        )
        duration = end_timestamp - start_timestamp
        participant_cname = next(name_iter)
        print(f"Codename '{participant_cname}' for {participant_platform_id}.")
        self._cursor.execute(
            participant_insert,
            (
                participant_cname,
                start_timestamp,
                duration,
                feedback,
                survey_id,
            ),
        )
        participant_id = self._cursor.fetchall()[0][0]
        for response in participant_dict["answers"].values():
            self._add_question_and_response(response, participant_id)

    def _add_survey(
        self,
        survey_name,
        survey_responses,
        name_iter: Iterator,
    ):
        survey_insert = "INSERT INTO survey(name) VALUES(?) RETURNING id"

        self._cursor.execute(survey_insert, (survey_name,))
        survey_id = self._cursor.fetchall()[0][0]
        print(f"created survey {survey_id}.")
        print("-" * 30)
        for part_responses in survey_responses:
            self._add_participant(
                part_responses,
                survey_id,
                name_iter,
            )

    def _create_similarity_db(self, survey_dict: dict[str, RawSurveyType]):
        try:
            self._setup_db()
            name_it = name_generator()

            self._connection.execute("PRAGMA foreign_keys = 1")

            for survey_name, survey in survey_dict.items():
                self._add_survey(survey_name, survey, name_it)

            self._connection.commit()

        except Exception as e:
            self._connection.rollback()
            raise e

    def db_exists(self: Self):
        flink = Path(self.db_fpath)
        return flink if flink.exists() else None

    def create_similarity_db(self, survey_dict: dict[str, RawSurveyType]):
        flink = self.db_exists()
        if flink is not None:
            print("database already exists")
            print(f"if you want to recreate it, first delete {self.db_fpath}.")

            raise ValueError

        if not Path(self.schema_path).exists():
            print(f"schema sql file not present in {self.schema_path}")
            print("file needed for creating db")

            raise ValueError

        total_participants = sum([len(s) for s in survey_dict.values()])
        if total_participants > total_cnames():
            print("codename generator does not have enough names.")
            print(
                "generator capacity:",
                total_cnames(),
                "partipants",
                total_participants,
            )
            raise ValueError
        try:
            self._create_similarity_db(survey_dict)

        except Exception as e:
            print("failed to created db, cleaning up")
            flink.unlink()  # type:ignore
            raise e

        return self._connection

    def insert_features(self, item_features: DataFrame) -> None:
        create_item_features = (
            get_schema(item_features, "item_features")[:-1]
            + ", FOREIGN KEY (item_id) REFERENCES item(id)\n)"
        )
        self._cursor.execute(create_item_features)
        get_existing_items = "SELECT id FROM item"
        self._cursor.execute(get_existing_items)
        ids = [ret[0] for ret in self._cursor.fetchall()]
        item_features = item_features[item_features["item_id"].isin(ids)]
        item_features.to_sql(
            "item_features",
            if_exists="append",
            index=False,
            con=self._connection,
        )


def survey_name(survey_standard_fpath: str) -> str:
    survey_standard_fname = survey_standard_fpath.split("/")[-1]
    assert survey_standard_fname.startswith("survey_")
    assert survey_standard_fname.endswith(".json")
    return survey_standard_fname.split("survey_")[1].split(".json")[0]


def load_survey(fpath) -> RawSurveyType:
    with open(fpath) as f:
        survey = json.load(f)
    return survey


def load_survey_dicts(raw_data_path: str) -> dict[str, RawSurveyType]:
    if Path(raw_data_path).exists():
        if not raw_data_path.endswith("/"):
            raw_data_path += "/"
        file_paths = glob.glob(f"{raw_data_path}survey_*.json")
        # print(list(file_paths))
        return {survey_name(f): load_survey(f) for f in file_paths}
    else:
        print("Raw data directory not found")
        return {}


# First register survey: fields of start and end time can be left null
# before filling the other tables
# 1. Register survey entry
# 2. For each participant:
#     1. Generate funny name
#     2. Add to db and get their key
#     2. For each question answered
#         1. check if it was already saved in the db
#         2. if not, add it
#         3. add reply linking it to question key
# 3. Finally, update survey info.


def create_and_build_db(raw_data_path, features_path=None) -> SimilarityDBManager:
    survey_dicts = load_survey_dicts(raw_data_path)
    if len(survey_dicts) == 0:
        print(f"could not find any survey data in dir {raw_data_path}")
        raise ValueError

    db_manager = SimilarityDBManager()
    if db_manager.db_exists() is None:
        db_manager.create_similarity_db(survey_dicts)
        if features_path is not None:
            item_features = read_csv(features_path, dtype={"item_id": object})
            _pad = partial(pad_left, c="0", max_len=4)
            item_features["item_id"] = item_features["item_id"].map(_pad)
            db_manager.insert_features(item_features)
    else:
        print("DB already exists, continuing")
    return db_manager
