-- Sprint 18b: User Specializations
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS user_specializations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL,
  selected_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, track_id)
);

ALTER TABLE user_specializations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own specializations"
  ON user_specializations FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own specializations"
  ON user_specializations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own specializations"
  ON user_specializations FOR UPDATE
  USING (auth.uid() = user_id);
