# ragshield.dev

Static landing page plus one Pages Function (`POST /api/apply`) that stores design-partner applications in D1. Every
submission is checked with Turnstile and rate-limited per salted IP hash, and notifications go to Discord, ntfy and/or email.
No analytics and no third-party scripts except Turnstile (see `public/_headers` for the CSP).

```
public/            index.html, privacy.html, terms.html, app.js, styles.css, _headers
functions/api/     apply.js (Pages Function)
lib/validate.js    validation shared with the tests
schema.sql         D1 schema
```

## Local development

```bash
npm ci
cp .dev.vars.example .dev.vars           # Cloudflare's always-pass Turnstile test keys
npm run db:local
npm run dev                              # http://localhost:8788
npm test                                 # validation tests
```

`public/index.html` carries the production Turnstile site key, which doesn't render on `localhost`. For local testing,
temporarily set `data-sitekey` to Cloudflare's test key `1x00000000000000000000AA` (it pairs with the test secret in `.dev.vars`).

## Deploy (one time)

```bash
npx wrangler login
npx wrangler pages project create ragshield-site --production-branch main
npx wrangler d1 create ragshield-site    # copy database_id into wrangler.toml
npm run db:remote
```

1. **Turnstile:** Cloudflare dashboard › Turnstile › Add widget for `ragshield.dev` and `ragshield-site.pages.dev` (Managed
   mode). Put the **site key** in `public/index.html` (`data-sitekey`). With the test key in production, submissions fail
   verification. That's noisy, but it fails closed.
2. **Secrets:**
   ```bash
   npx wrangler pages secret put TURNSTILE_SECRET --project-name ragshield-site
   npx wrangler pages secret put IP_SALT --project-name ragshield-site            # e.g. openssl rand -hex 32
   npx wrangler pages secret put DISCORD_WEBHOOK_URL --project-name ragshield-site
   npx wrangler pages secret put NTFY_URL --project-name ragshield-site           # e.g. https://ntfy.example.com/ragshield
   npx wrangler pages secret put NTFY_TOKEN --project-name ragshield-site         # if the topic is protected
   ```
   Email (optional) uses Resend: `RESEND_API_KEY`, `NOTIFY_EMAIL_TO`, and `NOTIFY_EMAIL_FROM`, which must be on a domain verified in Resend.
3. **Links:** set `data-cal` (your Cal.com 15-minute event) and, once the repo is public, `data-github` on `<body>` in `index.html`.
   Both stay hidden until set.
4. **Deploy:** `npm run deploy`, then Pages project › Custom domains › add `ragshield.dev`.
5. **Inbox:** Cloudflare › Email Routing › forward `hello@ragshield.dev` to your mailbox (the page and privacy policy use it).

## Reading applications

```bash
npx wrangler d1 execute ragshield-site --remote --command \
  "select created_at, name, email, company, role, company_size, timeline, wants_call, indexes from applications order by id desc"
```

Applications are kept up to 12 months (privacy page). Delete on request with `delete from applications where email = ?`.
