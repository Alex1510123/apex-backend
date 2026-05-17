-- Sprint 19: Open Answer Attempts
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS open_answer_attempts (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  lesson_title text NOT NULL,
  question text NOT NULL,
  answer text NOT NULL,
  score integer NOT NULL,
  passed boolean NOT NULL DEFAULT false,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE open_answer_attempts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert own attempts"
  ON open_answer_attempts FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can read own attempts"
  ON open_answer_attempts FOR SELECT
  USING (auth.uid() = user_id);
