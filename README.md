# ai-discord-member

An AI-powered Discord member that reads chat, decides when to speak, and replies
in character using [Groq](https://console.groq.com/) (Llama 3.3 70B by default).
Tries to feel like an actual person in your server — short messages, casual
voice, rolling conversation memory, cooldowns so it doesn't spam.

Currently configured as **Alex**: a chill early-20s dev who loves coding and
LEGO. Edit `persona.py` to change who the bot is.

## How it behaves

- **Always** replies when you @mention it, reply to one of its messages, or
  say its persona name in chat.
- **Often** chimes in otherwise, based on `UNPROMPTED_REPLY_CHANCE`
  (default ~75%), with a per-channel cooldown so it doesn't machine-gun.
- Remembers the last ~25 messages per channel for context.
- Uses a typing indicator and a short, length-based "thinking" delay so replies
  don't appear instantly.
- Can choose to stay silent by emitting `[[SKIP]]`.

## Setup

### 1. Create a Discord bot

1. Go to <https://discord.com/developers/applications> → **New Application**.
2. Under **Bot**, click **Reset Token** and save the token (this goes in
   `DISCORD_BOT_TOKEN`).
3. Under **Bot → Privileged Gateway Intents**, turn **Message Content Intent**
   **ON**.
4. Under **OAuth2 → URL Generator**, tick scopes `bot` and
   `applications.commands`, and permissions `Send Messages`, `Read Message History`,
   `View Channels`. Open the generated URL to invite the bot to your server.

### 2. Get a Groq API key

1. Sign up at <https://console.groq.com/> (free, no credit card).
2. Create a key at <https://console.groq.com/keys>.

### 3. Run locally

```bash
cp .env.example .env
# edit .env and fill in DISCORD_BOT_TOKEN and GROQ_API_KEY

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

You should see `Connected as <bot name>`. Send a message in a channel the bot
can see, and it'll reply.

## Deploy to Fly.io (free tier)

Fly is the easiest always-on free-ish host for Discord bots. It requires a
credit card on file but a 256 MB shared-CPU worker stays in the free allotment.

```bash
# One-time: install the CLI from https://fly.io/docs/hands-on/install-flyctl/
fly auth login

# Creates the app using fly.toml (pick a unique name).
fly launch --no-deploy --copy-config --name <your-app-name>

# Store secrets on Fly (never commit real .env).
fly secrets set \
  DISCORD_BOT_TOKEN="..." \
  GROQ_API_KEY="..."

fly deploy
fly logs -a <your-app-name>
```

To change the persona later, edit `persona.py` and re-run `fly deploy`.

## Configuration

All config is via environment variables. See `.env.example` for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | *required* | Discord bot token. |
| `GROQ_API_KEY` | *required* | Groq API key. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-hosted model id. |
| `BOT_PERSONA_NAME` | `alex` | Name the bot responds to. |
| `ALLOWED_CHANNEL_IDS` | *(all)* | Comma-separated channel IDs. Blank = everywhere the bot can see. |
| `UNPROMPTED_REPLY_CHANCE` | `0.75` | 0–1. Chance it jumps into a message not directed at it. |
| `PER_CHANNEL_COOLDOWN_SECONDS` | `6` | Min gap between its own messages in one channel. |
| `CONTEXT_WINDOW_MESSAGES` | `25` | How many recent messages per channel to remember. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

## Project layout

```
bot.py           # Discord client, response policy, Groq call, sender
persona.py       # Persona description + behavior rules (edit to retheme)
requirements.txt
Dockerfile       # used by Fly.io
fly.toml         # Fly.io app config (no HTTP service)
.env.example     # template for local env vars
```

## Notes

- **Never** commit a real `.env`, paste tokens in chat, or share keys in logs.
  If a token leaks, rotate it from the Developer Portal / Groq console immediately.
- The bot does not claim to be human; if someone explicitly asks whether it's
  a bot it will deflect casually to stay in character, but if you want
  transparent disclosure, edit `BEHAVIOR_RULES` in `persona.py`.
