// POST /api/apply: design-partner applications.
// Turnstile check -> validation -> per-IP rate limit -> D1 insert -> notifications (Discord, ntfy, email), best effort.
//
// Bindings and variables (wrangler.toml or the Pages dashboard):
//   DB                    D1 database
//   TURNSTILE_SECRET      secret key of the Turnstile widget
//   IP_SALT               random string; IPs are stored only as salted hashes
//   DISCORD_WEBHOOK_URL   optional
//   NTFY_URL              optional, e.g. https://ntfy.sh/your-topic or https://ntfy.example.com/ragshield
//   NTFY_TOKEN            optional bearer token for a protected ntfy topic
//   RESEND_API_KEY, NOTIFY_EMAIL_TO, NOTIFY_EMAIL_FROM   optional email via Resend

import { validate } from '../../lib/validate.js';

const MAX_PER_IP_PER_HOUR = 5;

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });

async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function turnstileOk(env, token, ip) {
  if (!token) return false;
  const form = new FormData();
  form.append('secret', env.TURNSTILE_SECRET);
  form.append('response', token);
  if (ip) form.append('remoteip', ip);
  const res = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', { method: 'POST', body: form });
  const data = await res.json();
  return data.success === true;
}

function summary(a) {
  return `New RAG Shield design-partner application\n${a.name}, ${a.role} at ${a.company} (${a.company_size})\n` +
    `Email: ${a.email}\nIndexes: ${a.indexes}\nStack: ${a.stack || '-'}\nTimeline: ${a.timeline}${a.wants_call ? ' (wants a call)' : ''}`;
}

async function notify(env, application) {
  const text = summary(application);
  const jobs = [];
  if (env.DISCORD_WEBHOOK_URL) {
    jobs.push(['discord', fetch(env.DISCORD_WEBHOOK_URL, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      // allowed_mentions: none, so submitted text cannot ping @everyone
      body: JSON.stringify({ content: text.slice(0, 1900), allowed_mentions: { parse: [] } }),
    })]);
  }
  if (env.NTFY_URL) {
    const headers = { Title: 'RAG Shield design partner', Tags: 'shield', 'Content-Type': 'text/plain; charset=utf-8' };
    if (env.NTFY_TOKEN) headers.Authorization = `Bearer ${env.NTFY_TOKEN}`;
    jobs.push(['ntfy', fetch(env.NTFY_URL, { method: 'POST', headers, body: text })]);
  }
  if (env.RESEND_API_KEY && env.NOTIFY_EMAIL_TO && env.NOTIFY_EMAIL_FROM) {
    jobs.push(['email', fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, 'content-type': 'application/json' },
      body: JSON.stringify({ from: env.NOTIFY_EMAIL_FROM, to: [env.NOTIFY_EMAIL_TO], reply_to: application.email,
        subject: `Design partner: ${application.company}`, text }),
    })]);
  }
  const results = await Promise.allSettled(jobs.map(([, job]) => job));
  return jobs.filter((_, i) => results[i].status === 'fulfilled' && results[i].value.ok).map(([name]) => name);
}

export async function onRequestPost({ request, env, waitUntil }) {
  if (!(request.headers.get('content-type') || '').includes('application/json')) return json({ error: 'Expected JSON.' }, 415);
  if (Number(request.headers.get('content-length') || 0) > 8192) return json({ error: 'Request too large.' }, 413);
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: 'Invalid JSON.' }, 400);
  }
  if (body.website) return json({ ok: true }); // honeypot: bots fill hidden fields; pretend success, store nothing

  const ip = request.headers.get('CF-Connecting-IP') || '';
  if (!(await turnstileOk(env, body.turnstile, ip))) return json({ error: 'Verification failed. Please reload and try again.' }, 403);

  const checked = validate(body);
  if (checked.error) return json({ error: checked.error }, 422);
  const a = checked.value;

  const ipHash = await sha256(`${env.IP_SALT || ''}:${ip}`);
  const recent = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM applications WHERE ip_hash = ? AND created_at > strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-1 hour')",
  ).bind(ipHash).first();
  if (recent.n >= MAX_PER_IP_PER_HOUR) return json({ error: 'Too many applications from this network. Please try again later.' }, 429);

  const row = await env.DB.prepare(
    `INSERT INTO applications (name, email, company, role, company_size, indexes, stack, timeline, wants_call, country, ip_hash)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id`,
  ).bind(a.name, a.email, a.company, a.role, a.company_size, a.indexes, a.stack || null, a.timeline, a.wants_call ? 1 : 0,
    request.cf?.country || null, ipHash).first();

  // Notifications must never block or fail the submission.
  waitUntil(notify(env, a).then((channels) =>
    env.DB.prepare('UPDATE applications SET notified = ? WHERE id = ?').bind(channels.join(',') || 'none', row.id).run()));
  return json({ ok: true });
}
