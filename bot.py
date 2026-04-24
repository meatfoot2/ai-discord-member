"""AI Discord member bot.

Reads chat in the servers it joins, decides when to speak, and responds using
Groq-hosted Llama. Tries to behave like an actual member of the server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections import defaultdict, deque
from typing import Deque

import discord
from discord.ext import commands
from dotenv import load_dotenv
from groq import AsyncGroq

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

SKIP_TOKEN = "[[SKIP]]"


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


class ChannelMemory:
    """Rolling buffer of recent chat messages per channel."""

    def __init__(self, window: int = CONTEXT_WINDOW_MESSAGES) -> None:
        self._window = window
        self._buffers: dict[int, Deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def add(self, channel_id: int, author: str, content: str, is_self: bool) -> None:
        if not content:
            return
        self._buffers[channel_id].append(
            {"author": author, "content": content, "is_self": is_self}
        )

    def transcript(self, channel_id: int) -> list[dict[str, str]]:
        """Return chat history as LLM-friendly messages.

        The bot's own lines are mapped to `assistant` so the model sees its own
        voice; everyone else is `user`, with the author name prefixed so the
        model can tell speakers apart inside a single `user` turn.
        """
        out: list[dict[str, str]] = []
        for item in self._buffers[channel_id]:
            if item["is_self"]:
                out.append({"role": "assistant", "content": item["content"]})
            else:
                out.append(
                    {"role": "user", "content": f"{item['author']}: {item['content']}"}
                )
        return out


class ResponsePolicy:
    """Decides whether the bot should speak for a given incoming message."""

    def __init__(self) -> None:
        self._last_spoke_at: dict[int, float] = {}

    def mark_spoke(self, channel_id: int) -> None:
        self._last_spoke_at[channel_id] = time.monotonic()

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

        # Cooldown: avoid double-posting back-to-back.
        last = self._last_spoke_at.get(message.channel.id, 0.0)
        if time.monotonic() - last < PER_CHANNEL_COOLDOWN_SECONDS:
            return False

        # Always respond to direct addressing.
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
        self.memory = ChannelMemory()
        self.policy = ResponsePolicy()
        self.groq = AsyncGroq(api_key=GROQ_API_KEY)

    async def setup_hook(self) -> None:
        log.info("Using Groq model: %s", GROQ_MODEL)

    async def on_ready(self) -> None:
        assert self.user is not None
        log.info(
            "Connected as %s (id=%s). In %d guild(s).",
            self.user,
            self.user.id,
            len(self.guilds),
        )
        try:
            await self.change_presence(
                activity=discord.Game(name="messing with some code"),
                status=discord.Status.online,
            )
        except Exception as exc:  # noqa: BLE001 - cosmetic only
            log.debug("Couldn't set presence: %s", exc)

    async def on_message(self, message: discord.Message) -> None:
        # Record everything we can see (including our own) so context stays accurate.
        is_self = self.user is not None and message.author.id == self.user.id
        if message.content:
            self.memory.add(
                channel_id=message.channel.id,
                author=message.author.display_name,
                content=message.content,
                is_self=is_self,
            )

        if not self.policy.should_respond(message, self.user):
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
        if not cleaned:
            return

        await self._send_like_a_human(message.channel, cleaned)
        self.policy.mark_spoke(message.channel.id)
        # Record our own message in memory too, so follow-ups stay coherent.
        if self.user is not None:
            self.memory.add(
                channel_id=message.channel.id,
                author=self.user.display_name,
                content=cleaned,
                is_self=True,
            )

    async def _generate_reply(self, message: discord.Message) -> str:
        history = self.memory.transcript(message.channel.id)
        system_prompt = build_system_prompt(
            extra=(
                f"The current channel is #{getattr(message.channel, 'name', 'dm')}. "
                f"You go by '{PERSONA_NAME}' here. Keep replies short and in-character."
            )
        )
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
