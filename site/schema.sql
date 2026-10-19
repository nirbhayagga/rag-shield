-- D1 schema for design-partner applications.
--   npx wrangler d1 execute ragshield-site --remote --file=schema.sql
CREATE TABLE IF NOT EXISTS applications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  company TEXT NOT NULL,
  role TEXT NOT NULL,
  company_size TEXT NOT NULL,
  indexes TEXT NOT NULL,          -- what their assistant or RAG system indexes
  stack TEXT,                     -- framework and vector database
  timeline TEXT NOT NULL,
  wants_call INTEGER NOT NULL DEFAULT 0,
  country TEXT,                   -- from Cloudflare's request metadata
  ip_hash TEXT NOT NULL,          -- salted SHA-256, used only for rate limiting
  notified TEXT                   -- channels that accepted the notification
);
CREATE INDEX IF NOT EXISTS applications_ip_time ON applications (ip_hash, created_at);
CREATE INDEX IF NOT EXISTS applications_email ON applications (email);
