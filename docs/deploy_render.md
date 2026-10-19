# Deploy the public demo on Render (free)

The console already runs locally with `.\run.ps1`. This puts a copy on the internet so anyone can open a link.
Nothing here changes local use.

## What you get and what to expect

- A link like `https://safeuav.onrender.com`, with HTTPS.
- **Free plan sleeps** after 15 minutes without visitors. The next visit takes about a minute to wake it. Drones are held in
  memory, so a sleep or a new deploy clears them. That is fine for a demo.
- Every visitor tab gets its own drone. No accounts.
- The AI is Gemini, using **your** key as a secret on Render. All visitors share a daily budget
  (`SAFEUAV_AI_DAILY_BUDGET`, default 200 calls). When it is used up, or the key is missing, the basic parser answers and the page
  shows "AI limit for today reached: basic parser". The local models (Ollama) are not offered in public mode.
- Limits per visitor: 6 chat messages a minute per drone, 40 per address, 60 flight starts an hour per address, 20 drones at once.

## Steps

1. Push the repo to GitHub (done: `CodeBrew-in/SafeUAV`, branch `main`).
2. Sign in at https://dashboard.render.com with GitHub.
3. **New +** → **Blueprint** → connect the repository `CodeBrew-in/SafeUAV` (grant Render access to that one repository only).
   Render reads `render.yaml` and proposes a service named `safeuav` on the Free plan.
4. When asked for `GEMINI_API_KEY`, paste your Google AI Studio key there. Do not put it in any file or message.
5. **Apply**. The first build takes a few minutes. When the service shows *Live*, open its `onrender.com` link.
6. Check `https://<your-service>.onrender.com/healthz` shows `ok`, then open the main page and run the demo script.

If the page shows a "403 bad host" message, the host name was not picked up: in the service's *Environment* tab add
`SAFEUAV_PUBLIC_HOST` with the exact `onrender.com` host name and redeploy.

## Keep it healthy

- **Update:** push to `main`; Render redeploys by itself.
- **Quota:** watch the chip in the page header ("AI ready · N calls left today"). Raise or lower `SAFEUAV_AI_DAILY_BUDGET` in the
  Environment tab to match your Gemini plan.
- **Keep awake for a demo:** open the link a minute before you present, so the first visitor does not wait.
- **Logs:** the Render dashboard's *Logs* tab. The server does not log requests (they can carry tokens).
- **Rotate the key:** change `GEMINI_API_KEY` in the Environment tab. Nothing else needs to change.

## What public mode changes (and what it does not)

On: the public host name is the only accepted Host, origins must be `https://<host>`, the server listens on all addresses on the
host's `PORT`, HSTS is sent, rate limits and the AI budget apply, `/healthz` answers without a token.

Not changed: the safety rules (no-fly zone, geofence, battery reserve, pad list), the shield, strict JSON, the token check, the
connection limits, the simulation-only notices. See [threat_model.md](threat_model.md).

Local mode is unchanged: without `SAFEUAV_PUBLIC=1` the console listens on 127.0.0.1 only.

## If the AI does not answer

The page header chip tells you why, and the chat message says the provider's reason in short form.

| Chip says | Meaning | Fix |
|---|---|---|
| AI not configured | no `GEMINI_API_KEY` on Render | add it in *Environment* |
| AI key rejected | Google says "API key not valid" | create a fresh key at https://aistudio.google.com/apikey and paste it again. New AI Studio keys start with `AQ.` and work. Stray spaces, quotes and line breaks are removed automatically |
| AI quota used up | the provider's free quota is spent | wait for the daily reset, or use a key from a project with quota |
| AI limit for today reached | our own daily budget (`SAFEUAV_AI_DAILY_BUDGET`) is spent | raise it in *Environment* |

While any of these is shown, the basic parser keeps answering simple orders.
