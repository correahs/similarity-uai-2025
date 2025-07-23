-- Create the 'item' table
CREATE TABLE item (
    id TEXT PRIMARY KEY, -- unique ID, reuse photo number from CROCUFID
    short_description TEXT NOT NULL,
    long_description TEXT,
    image_path TEXT
);


-- Create the 'question' table
CREATE TABLE question (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- Incremental unique ID
    target TEXT NOT NULL, -- item_id (foreign key)
    FOREIGN KEY (target) REFERENCES item(id)
);

-- Create the 'choice_set_item' table (many-to-many relationship between question and item)
CREATE TABLE choice_set_item (
    question_id INTEGER NOT NULL, -- Foreign key to question
    item_id TEXT NOT NULL, -- Foreign key to item
    FOREIGN KEY (question_id) REFERENCES question(id),
    FOREIGN KEY (item_id) REFERENCES item(id)
);

-- Create the 'participant' table
CREATE TABLE participant (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- UUID stored as TEXT 
    name TEXT NOT NULL,
    first_access_timestamp DATETIME NOT NULL,
    total_access_duration INTEGER NOT NULL, -- milliseconds
    feedback TEXT,
    survey_id INTEGER NOT NULL,
    FOREIGN KEY (survey_id) REFERENCES survey(id)
);

-- Create the 'survey' table
CREATE TABLE survey (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- UUID stored as TEXT
    name TEXT NOT NULL,
    survey_start DATETIME, -- first participant access
    survey_end DATETIME -- last participant finish
);

-- Create the 'image_features' table
-- CREATE TABLE image_features (
    -- item_id TEXT PRIMARY KEY, -- primary key, and foreign key to item (one:one)
    -- feature_1 REAL NOT NULL, -- Floating-point feature 1
    -- feature_2 REAL NOT NULL, -- Floating-point feature 2
    -- FOREIGN KEY (item_id) REFERENCES item(id)
-- );

-- Create the 'participant_response' table
CREATE TABLE participant_response (
    question_id INTEGER NOT NULL, -- Foreign key to question
    participant_id INTEGER NOT NULL, -- Foreign key to participant (UUID)
    response_timestamp DATETIME NOT NULL,
    selected TEXT NOT NULL, -- item_id (foreign key)
    duration INTEGER NOT NULL, -- milliseconds
    choice_set_order TEXT NOT NULL, -- JSON array of integers
    FOREIGN KEY (question_id) REFERENCES question(id),
    FOREIGN KEY (participant_id) REFERENCES participant(id),
    FOREIGN KEY (selected) REFERENCES item(id)
);

CREATE VIEW survey_question_view AS
SELECT DISTINCT s.id as survey_id, s.name as survey_name, q.id as question_id FROM question q
JOIN participant_response r ON question_id = q.id
JOIN participant p on p.id = r.participant_id
JOIN survey s on p.survey_id = s.id;

CREATE VIEW question_response_view AS
SELECT q.id as question_id, q.target, c.item_id as selected, COALESCE(r.n_participants, 0) as n_participants, sv.survey_name FROM question as q 
JOIN choice_set_item c ON c.question_id = q.id
JOIN survey_question_view as sv on sv.question_id = q.id
LEFT JOIN (
    SELECT question_id, selected, COUNT(*) as n_participants
    FROM participant_response 
    GROUP BY question_id, selected
) as r
ON q.id = r.question_id AND c.item_id = r.selected;
