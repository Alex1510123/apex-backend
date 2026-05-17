-- Sprint 17: Academy Progress Table
-- Run in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS academy_progress (
  id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id      UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  lesson_id    TEXT        NOT NULL,
  completed_at TIMESTAMPTZ DEFAULT NOW(),
  quiz_score   INTEGER,
  UNIQUE(user_id, lesson_id)
);

ALTER TABLE academy_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own progress"
  ON academy_progress FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own progress"
  ON academy_progress FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own progress"
  ON academy_progress FOR UPDATE
  USING (auth.uid() = user_id);
