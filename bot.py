"""AI Discord member bot.

Reads chat in the servers it joins, decides when to speak, and responds using
Groq-hosted Llama. Tries to behave like an actual member of the server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time

import discord
import httpx
from discord.ext import commands
from dotenv import load_dotenv
from groq import AsyncGroq

from chat_log import log_event
from images import search_image
from memory import MemoryStore
from persona import PERSONA_NAME, build_system_prompt

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ai-discord-member")

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ALLOWED_CHANNEL_IDS = {
    int(x)
    for x in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",")
    if x.strip().isdigit()
}
UNPROMPTED_REPLY_CHANCE = float(os.getenv("UNPROMPTED_REPLY_CHANCE", "0.75"))
PER_CHANNEL_COOLDOWN_SECONDS = float(os.getenv("PER_CHANNEL_COOLDOWN_SECONDS", "6"))
CONTEXT_WINDOW_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MESSAGES", "25"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "220"))

# If set, the bot will mirror every message + reply it sees into a text channel
# with this name in each guild (e.g. "bot-logs"). Leave blank to disable.
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL_NAME", "bot-logs").strip()

SKIP_TOKEN = "[[SKIP]]"
IMAGE_TAG_RE = re.compile(r"\[\[\s*IMG\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)
REMEMBER_TAG_RE = re.compile(r"\[\[\s*REMEMBER\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)


def require_env() -> None:
    missing = [
        k
        for k, v in [
            ("DISCORD_BOT_TOKEN", DISCORD_BOT_TOKEN),
            ("GROQ_API_KEY", GROQ_API_KEY),
        ]
        if not v
    ]
    if missing:
        raise SystemExit(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )


# Channel history + user facts are now persisted to SQLite (see memory.py).


class ResponsePolicy:
    """Decides whether the bot should speak for a given incoming message."""

    # Common question starters (used when the message has no '?').
    _QUESTION_STARTERS = (
        "what",
        "whats",
        "whos",
        "who",
        "where",
        "when",
        "why",
        "how",
        "which",
        "can you",
        "could you",
        "will you",
        "would you",
        "do you",
        "did you",
        "are you",
        "is it",
        "wdym",
        "wym",
    )

    # Within this window after the bot replied, treat a new message from the
    # SAME user as a direct follow-up and always respond.
    FOLLOWUP_WINDOW_SECONDS = 45.0

    def __init__(self) -> None:
        self._last_spoke_at: dict[int, float] = {}
        self._last_replied_to: dict[int, int] = {}

    def mark_spoke(self, channel_id: int, replying_to_user_id: int | None) -> None:
        self._last_spoke_at[channel_id] = time.monotonic()
        if replying_to_user_id is not None:
            self._last_replied_to[channel_id] = replying_to_user_id

    def _looks_like_question(self, text: str) -> bool:
        stripped = text.strip().lower()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return True
        first_chunk = stripped.split("?", 1)[0]
        return any(first_chunk.startswith(w) for w in self._QUESTION_STARTERS)

    def should_respond(
        self, message: discord.Message, bot_user: discord.ClientUser | None
    ) -> bool:
        if bot_user is None:
            return False
        if message.author.id == bot_user.id:
            return False
        if message.author.bot:
            return False
        if not message.content or not message.content.strip():
            return False
        if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
            return False

        # Direct addressing — always respond, even during cooldown.
        if bot_user in message.mentions:
            return True
        if message.reference is not None and getattr(
            message.reference, "resolved", None
        ):
            resolved = message.reference.resolved
            if (
                isinstance(resolved, discord.Message)
                and resolved.author.id == bot_user.id
            ):
                return True
        lowered = message.content.lower()
        if PERSONA_NAME.lower() in lowered:
            return True

        # Same user, right after the bot replied to them → follow-up.
        last_replied_to = self._last_replied_to.get(message.channel.id)
        last_spoke = self._last_spoke_at.get(message.channel.id, 0.0)
        if (
            last_replied_to == message.author.id
            and (time.monotonic() - last_spoke) < self.FOLLOWUP_WINDOW_SECONDS
        ):
            return True

        # Questions, but ONLY while the bot is already active in this channel.
        # This avoids replying to random "what time is it" messages between
        # two other users who aren't talking to the bot.
        if (
            time.monotonic() - last_spoke
        ) < self.FOLLOWUP_WINDOW_SECONDS and self._looks_like_question(message.content):
            return True

        # Cooldown only applies to unprompted chiming-in, not direct engagement.
        last = self._last_spoke_at.get(message.channel.id, 0.0)
        if time.monotonic() - last < PER_CHANNEL_COOLDOWN_SECONDS:
            return False

        # Otherwise, roll the dice so the bot feels like an active member,
        # not a reply-to-every-message spammer.
        return random.random() < UNPROMPTED_REPLY_CHANCE


class AIMember(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = False
        super().__init__(
            command_prefix="!!ai_unused!!", intents=intents, help_command=None
        )
        self.memory = MemoryStore()
        self.policy = ResponsePolicy()
        self.groq = AsyncGroq(api_key=GROQ_API_KEY)
        # Note: discord.Client uses `self.http` internally, so we name ours
        # differently.
        self.http_client = httpx.AsyncClient(timeout=8.0)

    async def setup_hook(self) -> None:
        log.info("Using Groq model: %s", GROQ_MODEL)

    async def close(self) -> None:  # type: ignore[override]
        try:
            await self.http_client.aclose()
        finally:
            try:
                self.memory.close()
            finally:
                await super().close()

    async def on_ready(self) -> None:
        assert self.user is not None
        log.info(
            "Connected as %s (id=%s). In %d guild(s): %s",
            self.user,
            self.user.id,
            len(self.guilds),
            [f"{g.name} (id={g.id})" for g in self.guilds],
        )
        try:
            await self.change_presence(
                activity=discord.Game(name="messing with some code"),
                status=discord.Status.online,
            )
        except Exception as exc:  # noqa: BLE001 - cosmetic only
            log.debug("Couldn't set presence: %s", exc)

    async def on_guild_join(self, guild: discord.Guild) -> None:
        log.info("Joined guild: %s (id=%s)", guild.name, guild.id)

    def _find_log_channel(
        self, guild: discord.Guild | None
    ) -> discord.TextChannel | None:
        if guild is None or not LOG_CHANNEL_NAME:
            return None
        for ch in guild.text_channels:
            if ch.name == LOG_CHANNEL_NAME:
                return ch
        return None

    async def _mirror_to_log_channel(
        self,
        source_channel: discord.abc.GuildChannel | discord.abc.Messageable,
        guild: discord.Guild | None,
        entry: str,
    ) -> None:
        """Post a one-line summary of an event into the guild's log channel."""
        log_channel = self._find_log_channel(guild)
        if log_channel is None:
            return
        # Never mirror events that originated in the log channel itself.
        if getattr(source_channel, "id", None) == log_channel.id:
            return
        try:
            if len(entry) > 1990:
                entry = entry[:1990].rstrip() + "…"
            await log_channel.send(
                entry, allowed_mentions=discord.AllowedMentions.none()
            )
        except discord.Forbidden:
            log.warning(
                "No perms to write to #%s in guild %s",
                log_channel.name,
                guild.name if guild else "?",
            )
        except discord.HTTPException as exc:
            log.warning("Failed to mirror to log channel: %s", exc)

    async def on_message(self, message: discord.Message) -> None:
        log.info(
            "on_message from %s in #%s (guild=%s): %r",
            message.author,
            getattr(message.channel, "name", "dm"),
            getattr(message.guild, "name", None),
            (message.content or "")[:200],
        )
        channel_name = getattr(message.channel, "name", "dm")
        guild_name = getattr(message.guild, "name", None)

        # Completely ignore activity in the bot-logs channel — it's a mirror,
        # not a real convo channel. Prevents loops and memory pollution.
        if channel_name == LOG_CHANNEL_NAME:
            return

        # Record everything we can see (including our own) so context stays accurate.
        is_self = self.user is not None and message.author.id == self.user.id
        if message.content:
            guild_id = message.guild.id if message.guild else None
            self.memory.add_message(
                guild_id=guild_id,
                channel_id=message.channel.id,
                author_id=message.author.id,
                author_name=message.author.display_name,
                is_self=is_self,
                content=message.content,
            )

        # Chat log: record every inbound message (bot replies get their own
        # "reply" entry below, after generation).
        if not is_self:
            log_event(
                kind="msg",
                guild=guild_name,
                channel=channel_name,
                author=str(message.author),
                author_id=message.author.id,
                content=message.content or "",
            )
            await self._mirror_to_log_channel(
                message.channel,
                message.guild,
                f"`#{channel_name}` **{message.author.display_name}:** "
                f"{message.content or ''}",
            )

        if is_self or not self.policy.should_respond(message, self.user):
            return

        try:
            reply = await self._generate_reply(message)
        except Exception:
            log.exception("Failed to generate a reply")
            return

        if not reply or reply.strip() == SKIP_TOKEN or SKIP_TOKEN in reply:
            log.debug("Model chose to stay silent.")
            return

        cleaned = self._clean_reply(reply)
        # Pull out any [[REMEMBER: ...]] facts the bot wants to save about
        # the user it's replying to, and strip them from the outgoing message.
        remember_tags, cleaned = self._extract_remember_tags(cleaned)
        text_part, image_query = self._extract_image_request(cleaned)
        image_url: str | None = None
        if image_query:
            image_url = await search_image(image_query, client=self.http_client)

        if not text_part and not image_url:
            return

        outgoing = text_part
        if image_url:
            outgoing = f"{text_part}\n{image_url}".strip()
        # Re-cap to Discord's 2000-char limit since the URL was appended
        # after _clean_reply's truncation.
        if len(outgoing) > 1990:
            outgoing = outgoing[:1990].rstrip() + "…"

        await self._send_like_a_human(message.channel, outgoing)
        self.policy.mark_spoke(message.channel.id, message.author.id)

        # Persist any facts the bot chose to remember about the author.
        guild_id = message.guild.id if message.guild else None
        for fact in remember_tags:
            self.memory.add_fact(
                user_id=message.author.id,
                user_name=message.author.display_name,
                fact=fact,
                guild_id=guild_id,
            )

        # Record our own message in memory too, so follow-ups stay coherent.
        if self.user is not None:
            remembered = text_part or f"(sent an image of: {image_query})"
            self.memory.add_message(
                guild_id=guild_id,
                channel_id=message.channel.id,
                author_id=self.user.id,
                author_name=self.user.display_name,
                is_self=True,
                content=remembered,
            )
            log_event(
                kind="reply",
                guild=guild_name,
                channel=channel_name,
                author=str(self.user),
                author_id=self.user.id,
                content=outgoing,
                extra={
                    "in_reply_to_user": str(message.author),
                    "in_reply_to_user_id": message.author.id,
                    "in_reply_to_content": (message.content or "")[:500],
                    "image_query": image_query,
                    "remembered_facts": remember_tags,
                },
            )
            await self._mirror_to_log_channel(
                message.channel,
                message.guild,
                f"`#{channel_name}` **{self.user.display_name}** "
                f"(→ {message.author.display_name}): {outgoing}",
            )

    async def _generate_reply(self, message: discord.Message) -> str:
        history = self.memory.transcript(
            message.channel.id, limit=CONTEXT_WINDOW_MESSAGES
        )
        facts = self.memory.facts_for(message.author.id)
        extra_lines = [
            f"The current channel is #{getattr(message.channel, 'name', 'dm')}.",
            f"You go by '{PERSONA_NAME}' here. Keep replies short and in-character.",
        ]
        if facts:
            facts_block = "\n".join(f"- {f}" for f in facts)
            extra_lines.append(
                f"Things you already know about {message.author.display_name} "
                f"from past conversations:\n{facts_block}"
            )
        system_prompt = build_system_prompt(extra="\n".join(extra_lines))
        payload = [{"role": "system", "content": system_prompt}, *history]

        resp = await self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=payload,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.9,
            top_p=0.95,
            stop=None,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_remember_tags(text: str) -> tuple[list[str], str]:
        """Pull out `[[REMEMBER: ...]]` tags and return (facts, cleaned_text)."""
        facts = [m.group(1).strip() for m in REMEMBER_TAG_RE.finditer(text)]
        facts = [f for f in facts if f]
        cleaned = REMEMBER_TAG_RE.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return facts, cleaned

    @staticmethod
    def _extract_image_request(text: str) -> tuple[str, str | None]:
        """Split the model's reply into ``(text, image_query)``.

        Only the first `[[IMG: ...]]` tag is honored; any extras are stripped
        from the text. Returns the original text (minus the tags) and the
        requested query, or ``None`` if no image was asked for.
        """
        matches = list(IMAGE_TAG_RE.finditer(text))
        if not matches:
            return text.strip(), None
        query = matches[0].group(1).strip()
        stripped = IMAGE_TAG_RE.sub("", text).strip()
        # Collapse the blank line the tag leaves behind.
        stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
        return stripped, query or None

    @staticmethod
    def _clean_reply(text: str) -> str:
        text = text.strip()
        # Strip stray persona prefixes the model sometimes emits.
        for prefix in (
            f"{PERSONA_NAME}:",
            f"{PERSONA_NAME.capitalize()}:",
            "Bot:",
            "Assistant:",
        ):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].lstrip()
        # Discord hard-caps messages at 2000 chars.
        if len(text) > 1900:
            text = text[:1900].rstrip() + "…"
        return text

    @staticmethod
    async def _send_like_a_human(
        channel: discord.abc.Messageable, content: str
    ) -> None:
        # Brief "thinking" pause + typing indicator so the bot doesn't fire
        # replies back instantly and feel robotic.
        think_delay = min(1.2 + len(content) / 120.0, 4.0)
        try:
            async with channel.typing():
                await asyncio.sleep(think_delay)
                await channel.send(content)
        except discord.Forbidden:
            log.warning(
                "Missing permission to send in channel id=%s",
                getattr(channel, "id", "?"),
            )
        except discord.HTTPException as exc:
            log.warning("Discord send failed: %s", exc)


def main() -> None:
    require_env()
    bot = AIMember()
    assert DISCORD_BOT_TOKEN is not None  # guarded by require_env
    bot.run(DISCORD_BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
