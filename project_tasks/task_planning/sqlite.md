# Add SQLite database for survey data

What information will be contained in the DB?



## Proposal for the DB schema
- **image_features**
    - item (item_id)
    - feature_1 (float)
    - feature_2 (float)

- **item**
    - id (id)
    - short_description (str)
    - long_description (str)
    - image_path (str)

- **question**
    - id (id)
    - target (item_id)

- **choice_set_item**
    - question (question_id)
    - item (item_id)

- **participant**
    - id (uuid)
    - username (str)
    - first_access_timestamp (timestamp)
    - total_access_duration (milliseconds)
    - feedback (str)

- **survey**
    - id (uuid)
    - name (str)
    - survey_start (timestamp) <!-- firt participant access -->
    - survey_end (timestamp) <!-- last participant finish -->

- **survey_question**
    - survey (survey_id)
    - question (question_id)

- **participant_reply**
    - survey (survey_id)
    - question (question_id)
    - participant (participant_id)
    - selected (item_id)
    - duration (milliseconds)
    - choice_set_order (list{int})

## How to read the jsons and put them into the db?


First register survey: fields of start and end time can be left null before filling the other tables

1. Register survey entry
2. For each participant:
    1. Generate funny name
    2. Add to db and get their key
    2. For each question answered
        1. check if it was already saved in the db
        2. if not, add it
        3. add reply linking it to question key
3. Finally, update survey info.