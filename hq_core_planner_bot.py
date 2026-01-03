# =========================================================
# HQ CORE PLANNER BOT (Puzzle & Survival) — 11x11 + Core/Rings
# COMPLETE SINGLE-FILE BOT (with COLORS + STORAGE DEBUG + PERSISTENT DATA_FILE)
#
# Install:
#   pip install -U discord.py pillow
#
# Environment Variables:
#   DISCORD_BOT_TOKEN   = your bot token
#   DISCORD_GUILD_ID    = (optional) your server id to speed slash command sync
#   DATA_FILE           = (recommended on Railway) /app/data/hq_layouts.json
#
# Railway Persistence:
# - Attach a Railway Volume node to this service
# - Set the Volume mount path to: /app/data
# - Set DATA_FILE variable to: /app/data/hq_layouts.json
#
# Fonts (optional but recommended for symbols):
#   Create folder next to this file: ./fonts/
#   Put:
#     NotoSans-Regular.ttf
#     NotoSansSymbols2-Regular.ttf
# =========================================================

import os
import json
import string
import io
import csv
from typing import Dict, Any, Optional, List, Tuple

import discord
from discord import app_commands
from discord.ext import commands

from PIL import Image, ImageDraw, ImageFont


# =========================================================
# STANDARD CONSTANTS
# =========================================================
STATE_ID = 789
LEADERSHIP_ROLE_NAME = "Leadership"

# IMPORTANT: makes file path configurable (Railway volume)
DATA_FILE = os.getenv("DATA_FILE", "hq_layouts.json")

# Default anchor (used only before calibration is fitted)
ANCHOR_SLOT_DEFAULT = "F6"
ANCHOR_X_DEFAULT = 220
ANCHOR_Y_DEFAULT = 484

# Absolute locked structure coordinates (never move)
LOCKED_STRUCTURES = [
    {"key": "FORT", "name": "ALLIANCE FORT", "s": 789, "x": 220, "y": 484},
    {"key": "SONIC", "name": "SONIC", "s": 789, "x": 222, "y": 486},
    {"key": "RSS", "name": "ALLIANCE RSS", "s": 789, "x": 216, "y": 468},
    {"key": "WAREHOUSE", "name": "ALLIANCE WAREHOUSE", "s": 789, "x": 218, "y": 466},
    {"key": "HOSPITAL", "name": "HOSPITAL", "s": 789, "x": 220, "y": 464},
]

# Default locked slots (can be overridden per-server with /layout unlockslot)
LOCKED_SLOTS = {
    "F6": {"name": "ALLIANCE FORT", "s": 789, "x": 220, "y": 484},
    "F7": {"name": "SONIC", "s": 789, "x": 222, "y": 486},
    "I1": {"name": "ALLIANCE RSS", "s": 789, "x": 216, "y": 468},
    "J1": {"name": "ALLIANCE WAREHOUSE", "s": 789, "x": 218, "y": 466},
    "K1": {"name": "HOSPITAL", "s": 789, "x": 220, "y": 464},
}

GUILD_ID_ENV = os.getenv("DISCORD_GUILD_ID")


# =========================================================
# CLAIM COLOR SUPPORT
# =========================================================
COLOR_PRESETS = {
    "red": (220, 60, 60),
    "orange": (230, 140, 40),
    "yellow": (220, 200, 60),
    "green": (60, 170, 90),
    "teal": (50, 170, 170),
    "blue": (70, 120, 220),
    "purple": (150, 90, 210),
    "pink": (220, 90, 170),
    "white": (235, 235, 235),
    "gray": (140, 140, 140),
    "black": (30, 30, 30),
}


def parse_color(color_str: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not color_str:
        return None
    s = color_str.strip().lower()
    if not s:
        return None

    if s in COLOR_PRESETS:
        return COLOR_PRESETS[s]

    if s.startswith("#"):
        s = s[1:]

    if len(s) == 6 and all(ch in "0123456789abcdef" for ch in s):
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)

    return None


def ideal_text_color(bg_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = bg_rgb
    lum = (0.299 * r + 0.587 * g + 0.114 * b)
    return (20, 20, 20) if lum > 150 else (245, 245, 245)


def color_help_text() -> str:
    presets = ", ".join(sorted(COLOR_PRESETS.keys()))
    return f"Use a preset ({presets}) or hex like `#FFAA00`."


# =========================================================
# FONT LOADER (UNICODE / SYMBOL FRIENDLY)
# =========================================================
def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        os.path.join("fonts", "NotoSans-Regular.ttf"),
        os.path.join("fonts", "NotoSansSymbols2-Regular.ttf"),
        os.path.join("fonts", "DejaVuSans.ttf"),
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# =========================================================
# PERSISTENCE
# =========================================================
def load_data() -> Dict[str, Any]:
    # Ensure directory exists if DATA_FILE has a folder (e.g. /app/data/...)
    data_dir = os.path.dirname(DATA_FILE)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: Dict[str, Any]) -> None:
    data_dir = os.path.dirname(DATA_FILE)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# BASIC HELPERS
# =========================================================
def col_labels(n: int) -> List[str]:
    return list(string.ascii_uppercase[:n])


def normalize_slot(slot: str) -> Optional[str]:
    slot = slot.strip().upper()
    if len(slot) < 2:
        return None
    col = slot[0]
    row = slot[1:]
    if not col.isalpha() or not row.isdigit():
        return None
    return f"{col}{int(row)}"


def slot_to_indices(slot: str) -> Tuple[int, int]:
    s = normalize_slot(slot)
    if not s:
        raise ValueError("Invalid slot")
    return ord(s[0]) - ord("A"), int(s[1:]) - 1


def indices_to_slot(col: int, row: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def is_leadership(member: discord.Member) -> bool:
    return any(role.name == LEADERSHIP_ROLE_NAME for role in getattr(member, "roles", []))


def is_locked_xy(s: int, x: int, y: int) -> Optional[Dict[str, Any]]:
    for st in LOCKED_STRUCTURES:
        if st["s"] == s and st["x"] == x and st["y"] == y:
            return st
    return None


def is_slot_locked(layout: Dict[str, Any], slot: str) -> bool:
    s = normalize_slot(slot)
    if not s or s not in LOCKED_SLOTS:
        return False
    overrides = layout.get("locked_slot_overrides", {})
    # default = locked; override False = unlocked
    return overrides.get(s, True) is True


def get_ring(slot: str) -> str:
    slot = normalize_slot(slot) or slot
    col = slot[0]
    row = int(slot[1:])

    if col in {"D", "E", "F", "G"} and row in {4, 5, 6, 7}:
        return "HQ CORE"
    if "C" <= col <= "H" and 3 <= row <= 8:
        return "RING 1"
    if "B" <= col <= "I" and 2 <= row <= 9:
        return "RING 2"
    return "RING 3"


# =========================================================
# CALIBRATION MATH (NO NUMPY)
# X = a0 + aC*col + aR*row
# Y = b0 + bC*col + bR*row
# =========================================================
def solve_3x3(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    M = [A[0][:] + [b[0]], A[1][:] + [b[1]], A[2][:] + [b[2]]]
    for i in range(3):
        pivot = i
        for r in range(i + 1, 3):
            if abs(M[r][i]) > abs(M[pivot][i]):
                pivot = r
        if abs(M[pivot][i]) < 1e-12:
            return None
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]

        piv = M[i][i]
        for c in range(i, 4):
            M[i][c] /= piv

        for r in range(3):
            if r == i:
                continue
            factor = M[r][i]
            for c in range(i, 4):
                M[r][c] -= factor * M[i][c]
    return [M[0][3], M[1][3], M[2][3]]


def fit_affine(points: Dict[str, Tuple[int, int]]) -> Optional[Dict[str, float]]:
    if len(points) < 3:
        return None

    ATA = [[0.0, 0.0, 0.0] for _ in range(3)]
    ATbx = [0.0, 0.0, 0.0]
    ATby = [0.0, 0.0, 0.0]

    for slot, (x, y) in points.items():
        c, r = slot_to_indices(slot)
        v = [1.0, float(c), float(r)]

        for i in range(3):
            for j in range(3):
                ATA[i][j] += v[i] * v[j]
        for i in range(3):
            ATbx[i] += v[i] * float(x)
            ATby[i] += v[i] * float(y)

    px = solve_3x3(ATA, ATbx)
    py = solve_3x3(ATA, ATby)
    if px is None or py is None:
        return None

    return {
        "a0": float(px[0]), "aC": float(px[1]), "aR": float(px[2]),
        "b0": float(py[0]), "bC": float(py[1]), "bR": float(py[2]),
    }


def affine_slot_to_xy(slot: str, coef: Dict[str, float]) -> Tuple[int, int]:
    c, r = slot_to_indices(slot)
    x = coef["a0"] + coef["aC"] * c + coef["aR"] * r
    y = coef["b0"] + coef["bC"] * c + coef["bR"] * r
    return int(round(x)), int(round(y))


def affine_xy_to_slot(x: int, y: int, coef: Dict[str, float], cols: int, rows: int) -> Optional[str]:
    aC, aR = coef["aC"], coef["aR"]
    bC, bR = coef["bC"], coef["bR"]
    det = (aC * bR) - (aR * bC)
    if abs(det) < 1e-12:
        return None

    dx = x - coef["a0"]
    dy = y - coef["b0"]

    col = (dx * bR - aR * dy) / det
    row = (aC * dy - dx * bC) / det

    col_i = int(round(col))
    row_i = int(round(row))

    if col_i < 0 or col_i >= cols or row_i < 0 or row_i >= rows:
        return None
    return indices_to_slot(col_i, row_i)


def slot_to_xy_layout(layout: Dict[str, Any], slot: str) -> Tuple[int, int]:
    coef = layout.get("calibration")
    if coef:
        return affine_slot_to_xy(slot, coef)

    anchor = layout.get("anchor", {})
    anchor_slot = anchor.get("slot", ANCHOR_SLOT_DEFAULT)
    ax = int(anchor.get("x", ANCHOR_X_DEFAULT))
    ay = int(anchor.get("y", ANCHOR_Y_DEFAULT))

    s = normalize_slot(slot)
    a = normalize_slot(anchor_slot)
    if not s or not a:
        raise ValueError("Invalid slot/anchor")

    dc = (ord(s[0]) - ord(a[0]))
    dr = (int(s[1:]) - int(a[1:]))

    return ax + dc, ay + dr


def xy_to_slot_layout(layout: Dict[str, Any], x: int, y: int) -> Optional[str]:
    coef = layout.get("calibration")
    cols = layout["cols"]
    rows = layout["rows"]
    if coef:
        return affine_xy_to_slot(x, y, coef, cols, rows)

    anchor = layout.get("anchor", {})
    anchor_slot = anchor.get("slot", ANCHOR_SLOT_DEFAULT)
    ax = int(anchor.get("x", ANCHOR_X_DEFAULT))
    ay = int(anchor.get("y", ANCHOR_Y_DEFAULT))

    a = normalize_slot(anchor_slot)
    if not a:
        return None

    dc = x - ax
    dr = y - ay
    col = chr(ord(a[0]) + dc)
    row = int(a[1:]) + dr

    valid_cols = set(col_labels(cols))
    if col not in valid_cols or row < 1 or row > rows:
        return None
    return f"{col}{row}"


# =========================================================
# COMMAND GROUPS
# =========================================================
layout_group = app_commands.Group(name="layout", description="HQ layout tools")
slot_group = app_commands.Group(name="slot", description="Slot tools")
cal_group = app_commands.Group(name="cal", description="Calibration tools")


# =========================================================
# BOT
# =========================================================
class HQCorePlanner(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)
        self.data = load_data()

    async def setup_hook(self):
        self.tree.add_command(layout_group)
        self.tree.add_command(slot_group)
        self.tree.add_command(cal_group)

        # Faster dev sync if DISCORD_GUILD_ID is set
        if GUILD_ID_ENV and GUILD_ID_ENV.isdigit():
            gid = int(GUILD_ID_ENV)
            guild = discord.Object(id=gid)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


bot = HQCorePlanner()


# =========================================================
# VALIDATION HELPERS
# =========================================================
def validate_slot_in_grid(layout: Dict[str, Any], slot: str) -> Optional[str]:
    cols = layout["cols"]
    rows = layout["rows"]
    valid_cols = set(col_labels(cols))
    if slot[0] not in valid_cols:
        return "That slot is outside the current grid."
    r = int(slot[1:])
    if r < 1 or r > rows:
        return "That slot is outside the current grid."
    return None


def validate_not_locked(layout: Dict[str, Any], slot: str) -> Optional[str]:
    s = normalize_slot(slot)
    if not s:
        return "Invalid slot."

    # Slot-level locks
    if is_slot_locked(layout, s):
        st = LOCKED_SLOTS[s]
        return (
            f"🔒 **LOCKED SLOT** ({st['name']}) — {s} is reserved and cannot be assigned.\n"
            f"Standard: **S:{st['s']} X:{st['x']} Y:{st['y']}**"
        )

    # Coordinate-level locks
    x, y = slot_to_xy_layout(layout, s)
    locked = is_locked_xy(STATE_ID, x, y)
    if locked:
        return (
            f"🔒 **LOCKED COORDINATE** ({locked['name']}) — S:{STATE_ID} X:{x} Y:{y}.\n"
            f"This indicates your mapping is placing {s} on a reserved structure tile."
        )

    return None


def ensure_settings(layout: Dict[str, Any]) -> Dict[str, Any]:
    layout.setdefault("settings", {"war_mode": False, "core_highlight": False})
    layout["settings"].setdefault("war_mode", False)
    layout["settings"].setdefault("core_highlight", False)
    return layout["settings"]


# =========================================================
# LAYOUT COMMANDS
# =========================================================
@layout_group.command(name="create", description="Create/overwrite layout for this server (default 11x11).")
@app_commands.describe(cols="Columns (default 11)", rows="Rows (default 11)", title="Layout title")
async def layout_create(
    interaction: discord.Interaction,
    cols: int = 11,
    rows: int = 11,
    title: str = "ALLIANCE HQ PLACEMENT — HQ CORE",
):
    if cols < 2 or cols > 26:
        await interaction.response.send_message("Cols must be between 2 and 26.", ephemeral=True)
        return
    if rows < 2 or rows > 50:
        await interaction.response.send_message("Rows must be between 2 and 50.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    bot.data[gid] = {
        "title": title,
        "cols": cols,
        "rows": rows,
        "claims": {},
        "slot_labels": {},
        "locked_slot_overrides": {},
        "anchor": {"state": STATE_ID, "slot": ANCHOR_SLOT_DEFAULT, "x": ANCHOR_X_DEFAULT, "y": ANCHOR_Y_DEFAULT},
        "cal_points": {},
        "calibration": None,
        "settings": {"war_mode": False, "core_highlight": False},
    }
    save_data(bot.data)

    await interaction.response.send_message(
        f"✅ Layout created: **{title}** ({cols}x{rows}).\n"
        f"🔒 Default locked slots: {', '.join(sorted(LOCKED_SLOTS.keys()))}.\n"
        f"Colors supported. {color_help_text()}\n"
        f"Recommended: add calibration points with `/cal add` then `/cal fit`."
    )


@layout_group.command(name="war", description="Toggle WAR mode styling for PNG (dark theme).")
@app_commands.describe(mode="on or off")
async def layout_war(interaction: discord.Interaction, mode: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    m = mode.strip().lower()
    if m not in ("on", "off"):
        await interaction.response.send_message("Mode must be `on` or `off`.", ephemeral=True)
        return

    settings = ensure_settings(layout)
    settings["war_mode"] = (m == "on")
    save_data(bot.data)

    await interaction.response.send_message(f"✅ WAR mode is now **{m.upper()}**.")


@layout_group.command(name="corehighlight", description="Toggle HQ CORE highlight outline on PNG.")
@app_commands.describe(mode="on or off")
async def layout_corehighlight(interaction: discord.Interaction, mode: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    m = mode.strip().lower()
    if m not in ("on", "off"):
        await interaction.response.send_message("Mode must be `on` or `off`.", ephemeral=True)
        return

    settings = ensure_settings(layout)
    settings["core_highlight"] = (m == "on")
    save_data(bot.data)

    await interaction.response.send_message(f"✅ CORE highlight is now **{m.upper()}**.")


@layout_group.command(name="unlockslot", description="Leadership: Unlock a default locked slot for this server.")
@app_commands.describe(slot="Example: F7")
async def layout_unlockslot(interaction: discord.Interaction, slot: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s or s not in LOCKED_SLOTS:
        await interaction.response.send_message(
            f"That slot is not a default locked standard. Standards: {', '.join(sorted(LOCKED_SLOTS.keys()))}.",
            ephemeral=True,
        )
        return

    layout.setdefault("locked_slot_overrides", {})
    layout["locked_slot_overrides"][s] = False
    save_data(bot.data)

    await interaction.response.send_message(f"✅ **{s}** is now **UNLOCKED** for this server.")


@layout_group.command(name="lockslot", description="Leadership: Re-lock a default locked slot for this server.")
@app_commands.describe(slot="Example: F7")
async def layout_lockslot(interaction: discord.Interaction, slot: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s or s not in LOCKED_SLOTS:
        await interaction.response.send_message(
            f"That slot is not a default locked standard. Standards: {', '.join(sorted(LOCKED_SLOTS.keys()))}.",
            ephemeral=True,
        )
        return

    layout.setdefault("locked_slot_overrides", {})
    layout["locked_slot_overrides"][s] = True
    save_data(bot.data)

    await interaction.response.send_message(f"🔒 **{s}** is now **LOCKED** again for this server.")


@layout_group.command(name="lockreport", description="Audit locks, mapping mismatches, and coordinate collisions.")
async def layout_lockreport(interaction: discord.Interaction):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    coef = layout.get("calibration")
    cal_status = "CALIBRATED ✅" if coef else "NOT CALIBRATED ⚠ (fallback mapping)"

    cols = layout["cols"]
    rows = layout["rows"]
    labels = col_labels(cols)

    lines: List[str] = []
    lines.append(f"**🔒 Lock Report** — Mapping: {cal_status}")
    lines.append("")
    lines.append("**1) Default locked slots (after overrides):**")
    for slot in sorted(LOCKED_SLOTS.keys()):
        st = LOCKED_SLOTS[slot]
        status = "LOCKED" if is_slot_locked(layout, slot) else "UNLOCKED (override)"
        lines.append(f"- {slot}: {status} — {st['name']} (X:{st['x']} Y:{st['y']})")

    lines.append("")
    lines.append("**2) Mapping prediction for default locked slots (audit):**")
    for slot in sorted(LOCKED_SLOTS.keys()):
        if validate_slot_in_grid(layout, slot):
            lines.append(f"- {slot}: outside grid")
            continue
        px, py = slot_to_xy_layout(layout, slot)
        st = LOCKED_SLOTS[slot]
        match = "MATCH ✅" if (px == st["x"] and py == st["y"]) else "MISMATCH ⚠"
        lines.append(f"- {slot}: pred X:{px} Y:{py} | std X:{st['x']} Y:{st['y']} → {match}")

    coord_to_slot: Dict[Tuple[int, int], str] = {}
    collisions: List[Tuple[str, str, int, int]] = []

    for r in range(1, rows + 1):
        for c in labels:
            slot = f"{c}{r}"
            x, y = slot_to_xy_layout(layout, slot)
            key = (x, y)
            if key in coord_to_slot:
                collisions.append((coord_to_slot[key], slot, x, y))
            else:
                coord_to_slot[key] = slot

    lines.append("")
    lines.append("**3) Coordinate collisions (two slots map to same X/Y):**")
    if not collisions:
        lines.append("- None ✅")
    else:
        for a, b, x, y in collisions[:25]:
            lines.append(f"- {a} and {b} both map to X:{x} Y:{y}")
        if len(collisions) > 25:
            lines.append(f"- (showing 25 of {len(collisions)} total)")

    lines.append("")
    lines.append("**4) Locked structures resolve to which slots?**")
    for st in LOCKED_STRUCTURES:
        s = xy_to_slot_layout(layout, st["x"], st["y"])
        if s:
            lines.append(f"- {st['name']} X:{st['x']} Y:{st['y']} → slot **{s}**")
        else:
            lines.append(f"- {st['name']} X:{st['x']} Y:{st['y']} → outside grid / not invertible")

    await interaction.response.send_message("\n".join(lines))


@layout_group.command(name="exportcsv", description="Export slot→coordinate table as CSV (ring + lock + claim + color).")
async def layout_exportcsv(interaction: discord.Interaction):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    cols = layout["cols"]
    rows = layout["rows"]
    labels = col_labels(cols)
    claims = layout.get("claims", {})

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["State", "Slot", "Ring", "X", "Y", "Locked", "LockedName", "LockedType", "ClaimedBy", "Color"])

    for r in range(1, rows + 1):
        for c in labels:
            slot = f"{c}{r}"
            x, y = slot_to_xy_layout(layout, slot)

            locked_type = ""
            locked_name = ""
            locked = False

            if is_slot_locked(layout, slot):
                locked = True
                locked_type = "SLOT"
                locked_name = LOCKED_SLOTS[slot]["name"]
            else:
                lxy = is_locked_xy(STATE_ID, x, y)
                if lxy:
                    locked = True
                    locked_type = "COORD"
                    locked_name = lxy["name"]

            claim = claims.get(slot, {})
            claimed_by = claim.get("user_name", "")
            color = claim.get("color", "")

            w.writerow(
                [STATE_ID, slot, get_ring(slot), x, y,
                 "YES" if locked else "NO", locked_name, locked_type, claimed_by, color]
            )

    data = out.getvalue().encode("utf-8")
    file = discord.File(fp=io.BytesIO(data), filename="hq_slot_coordinates.csv")
    await interaction.response.send_message("📎 CSV export:", file=file)


def render_png(layout: Dict[str, Any]) -> Tuple[str, bytes]:
    cols = layout["cols"]
    rows = layout["rows"]
    labels = col_labels(cols)

    claims = layout.get("claims", {})
    slot_labels = layout.get("slot_labels", {})
    title = layout.get("title", "HQ PLACEMENT")
    coef = layout.get("calibration")
    cal_status = "CALIBRATED ✅" if coef else "NOT CALIBRATED ⚠ (use /cal add + /cal fit)"

    settings = layout.get("settings", {})
    war_mode = bool(settings.get("war_mode", False))
    core_highlight = bool(settings.get("core_highlight", False))

    # Visual sizes
    cell_w, cell_h = 210, 112
    margin = 160
    title_h = 190
    w = margin + cols * cell_w + 40
    h = title_h + rows * cell_h + 150

    if war_mode:
        bg = (18, 18, 18)
        text_main = (235, 235, 235)
        text_sub = (170, 170, 170)
        grid_outline = (70, 70, 70)
        ring_color = {
            "HQ CORE": (120, 20, 20),
            "RING 1": (180, 95, 20),
            "RING 2": (150, 140, 40),
            "RING 3": (35, 95, 55),
        }
        locked_fill = (0, 0, 0)
        locked_text = (235, 235, 235)
    else:
        bg = (245, 245, 245)
        text_main = (20, 20, 20)
        text_sub = (70, 70, 70)
        grid_outline = (120, 120, 120)
        ring_color = {
            "HQ CORE": (255, 210, 210),
            "RING 1": (255, 235, 200),
            "RING 2": (255, 250, 200),
            "RING 3": (220, 245, 220),
        }
        locked_fill = (30, 30, 30)
        locked_text = (245, 245, 245)

    ring_tag = {"HQ CORE": "CORE", "RING 1": "R1", "RING 2": "R2", "RING 3": "R3"}

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    font_title = load_font(40)
    font_label = load_font(26)
    font_cell = load_font(20)
    font_small = load_font(18)
    font_micro = load_font(16)
    font_overlay = load_font(20)

    title_line = f"{title} — WAR MODE" if war_mode else title
    draw.text((40, 18), title_line, font=font_title, fill=text_main)
    draw.text((40, 66), f"Mapping: {cal_status}", font=font_small, fill=text_sub)
    draw.text((40, 86), f"Colors: {', '.join(sorted(COLOR_PRESETS.keys()))} or #RRGGBB", font=font_micro, fill=text_sub)

    # Locked standards legend
    y_leg = 104
    draw.text((40, y_leg), "Locked Standards (slot locks can be overridden per server):", font=font_small, fill=text_sub)
    y_leg += 22
    for slot in sorted(LOCKED_SLOTS.keys()):
        st = LOCKED_SLOTS[slot]
        status = "LOCKED" if is_slot_locked(layout, slot) else "UNLOCKED"
        draw.text((40, y_leg), f"- {slot} [{status}] = {st['name']} (X:{st['x']} Y:{st['y']})",
                  font=font_small, fill=text_sub)
        y_leg += 20

    # Determine locked cells in-grid: slot-level + coordinate-level
    locked_inside: Dict[str, Dict[str, Any]] = {}
    for slot, st in LOCKED_SLOTS.items():
        if is_slot_locked(layout, slot):
            try:
                c, r = slot_to_indices(slot)
            except Exception:
                continue
            if 0 <= c < cols and 0 <= r < rows:
                locked_inside[slot] = {"name": st["name"]}

    for rr in range(1, rows + 1):
        for cc in labels:
            slot = f"{cc}{rr}"
            if slot in locked_inside:
                continue
            x, y = slot_to_xy_layout(layout, slot)
            lxy = is_locked_xy(STATE_ID, x, y)
            if lxy:
                locked_inside[slot] = {"name": lxy["name"]}

    # Column labels
    for i, c in enumerate(labels):
        x = margin + i * cell_w
        draw.text((x + cell_w / 2 - 10, title_h - 46), c, font=font_label, fill=text_main)

    # Rows + cells
    for rr in range(1, rows + 1):
        y0 = title_h + (rr - 1) * cell_h
        draw.text((40, y0 + cell_h / 2 - 10), str(rr), font=font_label, fill=text_main)

        for i, cc in enumerate(labels):
            slot = f"{cc}{rr}"
            x0 = margin + i * cell_w

            ring = get_ring(slot)
            fill = ring_color[ring]
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=fill, outline=grid_outline, width=2)

            if core_highlight and ring == "HQ CORE":
                outline = (235, 235, 235) if war_mode else (90, 0, 0)
                draw.rectangle([x0 + 3, y0 + 3, x0 + cell_w - 3, y0 + cell_h - 3], outline=outline, width=5)

            draw.text((x0 + 8, y0 + 6), slot, font=font_cell, fill=text_main)
            tag = ring_tag[ring]
            tag_w = draw.textlength(tag, font=font_cell)
            draw.text((x0 + cell_w - tag_w - 10, y0 + 6), tag, font=font_cell, fill=text_main)

            sx, sy = slot_to_xy_layout(layout, slot)
            draw.text((x0 + 8, y0 + 30), f"S:{STATE_ID} X:{sx} Y:{sy}", font=font_micro, fill=text_sub)

            # Locked badge
            if slot in locked_inside:
                badge = f"LOCKED: {locked_inside[slot]['name']}"
                bw = draw.textlength(badge, font=font_micro)
                bx0, by0 = x0 + 8, y0 + 48
                bx1, by1 = min(x0 + cell_w - 8, bx0 + bw + 16), by0 + 22
                draw.rectangle([bx0, by0, bx1, by1], fill=locked_fill, outline=locked_fill, width=2)
                draw.text((bx0 + 8, by0 + 2), badge, font=font_micro, fill=locked_text)

            # Claimed name + color bar
            if slot in claims:
                name = claims[slot].get("user_name", "")
                color_str = claims[slot].get("color", None)
                rgb = parse_color(color_str) if color_str else None

                if len(name) > 18:
                    name = name[:17] + "…"

                if rgb:
                    bar_x0 = x0 + 8
                    bar_y0 = y0 + 70
                    bar_x1 = x0 + cell_w - 8
                    bar_y1 = bar_y0 + 30
                    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=rgb, outline=grid_outline, width=2)
                    name_color = ideal_text_color(rgb)
                    draw.text((bar_x0 + 8, bar_y0 + 5), name, font=font_cell, fill=name_color)
                else:
                    draw.text((x0 + 8, y0 + 72), name, font=font_cell, fill=text_main)

            # Overlay label
            if slot in slot_labels:
                overlay = str(slot_labels[slot]).strip().upper()
                if len(overlay) > 12:
                    overlay = overlay[:12] + "…"
                bar_h = 26
                bar_y0 = y0 + cell_h - bar_h - 8
                bar_x0 = x0 + 8
                bar_x1 = x0 + cell_w - 8
                bar_y1 = bar_y0 + bar_h
                draw.rectangle(
                    [bar_x0, bar_y0, bar_x1, bar_y1],
                    fill=(0, 0, 0) if war_mode else (30, 30, 30),
                    outline=(120, 120, 120) if war_mode else (80, 80, 80),
                    width=2,
                )
                ow = draw.textlength(overlay, font=font_overlay)
                draw.text(
                    (x0 + (cell_w - ow) / 2, bar_y0 + 2),
                    overlay,
                    font=font_overlay,
                    fill=(235, 235, 235) if war_mode else (245, 245, 245),
                )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "hq_layout.png", buf.getvalue()


@layout_group.command(name="png", description="Generate PNG map (names, colors, coords, locks, labels).")
async def layout_png(interaction: discord.Interaction):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return
    filename, png_bytes = render_png(layout)
    file = discord.File(fp=io.BytesIO(png_bytes), filename=filename)
    await interaction.response.send_message(file=file)


@layout_group.command(name="storage", description="Show where the bot is saving data (debug).")
async def layout_storage(interaction: discord.Interaction):
    path = os.path.abspath(DATA_FILE)
    exists = os.path.exists(DATA_FILE)
    size = os.path.getsize(DATA_FILE) if exists else 0
    await interaction.response.send_message(
        f"DATA_FILE = `{DATA_FILE}`\n"
        f"ABS PATH = `{path}`\n"
        f"EXISTS = `{exists}` | SIZE = `{size}` bytes",
        ephemeral=True,
    )


# =========================================================
# CAL COMMANDS
# =========================================================
@cal_group.command(name="add", description="Add calibration point: slot + state + X + Y.")
@app_commands.describe(slot="Example: F6", state="789", x="X", y="Y")
async def cal_add(interaction: discord.Interaction, slot: str, state: int, x: int, y: int):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can calibrate.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    if state != STATE_ID:
        await interaction.response.send_message(f"This bot is standardized for S:{STATE_ID}.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: F6).", ephemeral=True)
        return

    layout.setdefault("cal_points", {})
    layout["cal_points"][s] = [int(x), int(y)]
    save_data(bot.data)

    await interaction.response.send_message(f"✅ Saved: **{s}** → S:{STATE_ID} X:{x} Y:{y}")


@cal_group.command(name="show", description="Show saved calibration points.")
async def cal_show(interaction: discord.Interaction):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can view calibration.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    pts = layout.get("cal_points", {})
    coef = layout.get("calibration")
    lines = [f"**Calibration:** {'CALIBRATED ✅' if coef else 'NOT CALIBRATED ⚠'}", f"**Points:** {len(pts)}"]
    for k in sorted(pts.keys()):
        x, y = pts[k]
        lines.append(f"- {k}: S:{STATE_ID} X:{x} Y:{y}")
    await interaction.response.send_message("\n".join(lines))


@cal_group.command(name="clear", description="Clear all calibration points and fitted mapping.")
async def cal_clear(interaction: discord.Interaction):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can clear calibration.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    layout["cal_points"] = {}
    layout["calibration"] = None
    save_data(bot.data)

    await interaction.response.send_message("🧼 Calibration cleared. Add points with `/cal add`, then run `/cal fit`.")


@cal_group.command(name="fit", description="Fit mapping from saved points and show per-point error.")
async def cal_fit(interaction: discord.Interaction):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can calibrate.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    pts_raw = layout.get("cal_points", {})
    if len(pts_raw) < 3:
        await interaction.response.send_message("Add at least 3 points with `/cal add` (6–10 recommended).", ephemeral=True)
        return

    pts = {k: (v[0], v[1]) for k, v in pts_raw.items()}
    coef = fit_affine(pts)
    if not coef:
        await interaction.response.send_message(
            "Calibration failed. Add more varied points across the grid and retry.",
            ephemeral=True,
        )
        return

    layout["calibration"] = coef
    save_data(bot.data)

    lines = ["**✅ Calibration fitted. Error report (pred → actual):**"]
    worst_slot = None
    worst_err = -1

    for slot in sorted(pts.keys()):
        ax, ay = pts[slot]
        px, py = affine_slot_to_xy(slot, coef)
        err = abs(px - ax) + abs(py - ay)
        if err > worst_err:
            worst_err = err
            worst_slot = slot
        lines.append(f"- {slot}: pred({px},{py}) → actual({ax},{ay}) | Δ={err}")

    lines.append(f"\nWorst: **{worst_slot}** (Δ={worst_err}). If Δ is large, re-check and re-add that point.")
    await interaction.response.send_message("\n".join(lines))


# =========================================================
# SLOT COMMANDS
# =========================================================
@slot_group.command(name="coord", description="Get in-game coordinate for a slot.")
@app_commands.describe(slot="Example: E6")
async def slot_coord(interaction: discord.Interaction, slot: str):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: E6).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(f"**{s}** ({get_ring(s)}) → **S:{STATE_ID} X:{x} Y:{y}**")


@slot_group.command(name="fromcoord", description="Find slot for an in-game coordinate (if inside grid).")
@app_commands.describe(state="State (789)", x="X", y="Y")
async def slot_fromcoord(interaction: discord.Interaction, state: int, x: int, y: int):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    if state != STATE_ID:
        await interaction.response.send_message(f"This layout is standardized for S:{STATE_ID}.", ephemeral=True)
        return

    slot = xy_to_slot_layout(layout, int(x), int(y))
    if not slot:
        await interaction.response.send_message("That coordinate is outside the current grid (or not invertible).", ephemeral=True)
        return

    await interaction.response.send_message(f"S:{state} X:{x} Y:{y} → **{slot}** ({get_ring(slot)})")


@slot_group.command(name="claim", description="Members claim Ring 2 / Ring 3. Ring 1 & CORE are Leadership-assigned.")
@app_commands.describe(slot="Example: A10")
async def slot_claim(interaction: discord.Interaction, slot: str):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: A10).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    ring = get_ring(s)
    if ring in ("HQ CORE", "RING 1"):
        await interaction.response.send_message("HQ CORE and RING 1 are assigned by Leadership.", ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    claims = layout["claims"]
    if s in claims:
        await interaction.response.send_message(f"{s} is already claimed by **{claims[s]['user_name']}**.", ephemeral=True)
        return

    claims[s] = {"user_id": str(interaction.user.id), "user_name": interaction.user.display_name}
    save_data(bot.data)

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(f"✅ Claimed **{s}** → **S:{STATE_ID} X:{x} Y:{y}** for **{interaction.user.display_name}**.")


@slot_group.command(name="claimcolor", description="Members claim Ring 2/3 with a color (preset or #RRGGBB).")
@app_commands.describe(slot="Example: A10", color="Preset (red/blue/etc) or hex (#RRGGBB)")
async def slot_claimcolor(interaction: discord.Interaction, slot: str, color: str):
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: A10).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    ring = get_ring(s)
    if ring in ("HQ CORE", "RING 1"):
        await interaction.response.send_message("HQ CORE and RING 1 are assigned by Leadership.", ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    rgb = parse_color(color)
    if rgb is None:
        await interaction.response.send_message(f"Invalid color. {color_help_text()}", ephemeral=True)
        return

    claims = layout["claims"]
    if s in claims:
        await interaction.response.send_message(f"{s} is already claimed by **{claims[s]['user_name']}**.", ephemeral=True)
        return

    claims[s] = {"user_id": str(interaction.user.id), "user_name": interaction.user.display_name, "color": color.strip()}
    save_data(bot.data)

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(
        f"✅ Claimed **{s}** → **S:{STATE_ID} X:{x} Y:{y}** for **{interaction.user.display_name}** with color **{color}**."
    )


@slot_group.command(name="claimfor", description="Leadership: Assign a slot to a Discord member.")
@app_commands.describe(member="Member", slot="Example: E6")
async def slot_claimfor(interaction: discord.Interaction, member: discord.Member, slot: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: E6).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    claims = layout["claims"]
    if s in claims:
        await interaction.response.send_message(f"{s} is already claimed by **{claims[s]['user_name']}**.", ephemeral=True)
        return

    claims[s] = {"user_id": str(member.id), "user_name": member.display_name}
    save_data(bot.data)

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(
        f"🏰 Assigned **{s}** ({get_ring(s)}) → **S:{STATE_ID} X:{x} Y:{y}** to **{member.display_name}**."
    )


@slot_group.command(name="claimforcolor", description="Leadership: Assign a slot to a Discord member with a color.")
@app_commands.describe(member="Member", slot="Example: E6", color="Preset (red/blue/etc) or hex (#RRGGBB)")
async def slot_claimforcolor(interaction: discord.Interaction, member: discord.Member, slot: str, color: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: E6).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    rgb = parse_color(color)
    if rgb is None:
        await interaction.response.send_message(f"Invalid color. {color_help_text()}", ephemeral=True)
        return

    claims = layout["claims"]
    if s in claims:
        await interaction.response.send_message(f"{s} is already claimed by **{claims[s]['user_name']}**.", ephemeral=True)
        return

    claims[s] = {"user_id": str(member.id), "user_name": member.display_name, "color": color.strip()}
    save_data(bot.data)

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(
        f"🏰 Assigned **{s}** ({get_ring(s)}) → **S:{STATE_ID} X:{x} Y:{y}** to **{member.display_name}** with color **{color}**."
    )


@slot_group.command(name="claimname", description="Leadership: Assign a slot to a typed in-game name (symbols allowed).")
@app_commands.describe(slot="Example: E6", name="Type the player's in-game name (symbols allowed).")
async def slot_claimname(interaction: discord.Interaction, slot: str, name: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: E6).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    cleaned = name.strip()
    if not cleaned:
        await interaction.response.send_message("Name cannot be empty.", ephemeral=True)
        return
    if len(cleaned) > 60:
        cleaned = cleaned[:60] + "…"

    claims = layout["claims"]
    if s in claims:
        await interaction.response.send_message(
            f"{s} is already claimed by **{claims[s].get('user_name','(unknown)')}**.",
            ephemeral=True,
        )
        return

    claims[s] = {"user_id": None, "user_name": cleaned, "manual": True}
    save_data(bot.data)

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(
        f"✍️ Assigned **{s}** ({get_ring(s)}) → **S:{STATE_ID} X:{x} Y:{y}** to:\n**{cleaned}**"
    )


@slot_group.command(name="claimnamecolor", description="Leadership: Assign a slot to a typed in-game name with a color.")
@app_commands.describe(slot="Example: E6", name="In-game name (symbols allowed)", color="Preset (red/blue/etc) or hex (#RRGGBB)")
async def slot_claimnamecolor(interaction: discord.Interaction, slot: str, name: str, color: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format (example: E6).", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    cleaned = name.strip()
    if not cleaned:
        await interaction.response.send_message("Name cannot be empty.", ephemeral=True)
        return
    if len(cleaned) > 60:
        cleaned = cleaned[:60] + "…"

    rgb = parse_color(color)
    if rgb is None:
        await interaction.response.send_message(f"Invalid color. {color_help_text()}", ephemeral=True)
        return

    claims = layout["claims"]
    if s in claims:
        await interaction.response.send_message(
            f"{s} is already claimed by **{claims[s].get('user_name','(unknown)')}**.",
            ephemeral=True,
        )
        return

    claims[s] = {"user_id": None, "user_name": cleaned, "manual": True, "color": color.strip()}
    save_data(bot.data)

    x, y = slot_to_xy_layout(layout, s)
    await interaction.response.send_message(
        f"✍️ Assigned **{s}** ({get_ring(s)}) → **S:{STATE_ID} X:{x} Y:{y}** to:\n"
        f"**{cleaned}** with color **{color}**."
    )


@slot_group.command(name="claimcoord", description="Leadership: Assign a slot by in-game coordinate (S/X/Y).")
@app_commands.describe(member="Member", state="789", x="X", y="Y")
async def slot_claimcoord(interaction: discord.Interaction, member: discord.Member, state: int, x: int, y: int):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    if state != STATE_ID:
        await interaction.response.send_message(f"This bot is standardized for S:{STATE_ID}.", ephemeral=True)
        return

    slot = xy_to_slot_layout(layout, int(x), int(y))
    if not slot:
        await interaction.response.send_message("That coordinate is outside the current grid (or not invertible).", ephemeral=True)
        return

    lock_err = validate_not_locked(layout, slot)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    claims = layout["claims"]
    if slot in claims:
        await interaction.response.send_message(f"{slot} is already claimed by **{claims[slot]['user_name']}**.", ephemeral=True)
        return

    claims[slot] = {"user_id": str(member.id), "user_name": member.display_name}
    save_data(bot.data)

    await interaction.response.send_message(f"🏰 Assigned by coord: **{member.display_name}** → **{slot}** ({get_ring(slot)})")


@slot_group.command(name="recolor", description="Leadership: Change the color of an existing claim without unclaiming.")
@app_commands.describe(slot="Example: E6", color="Preset (red/blue/etc) or hex (#RRGGBB)")
async def slot_recolor(interaction: discord.Interaction, slot: str, color: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    claims = layout["claims"]
    if s not in claims:
        await interaction.response.send_message("That slot is not claimed.", ephemeral=True)
        return

    rgb = parse_color(color)
    if rgb is None:
        await interaction.response.send_message(f"Invalid color. {color_help_text()}", ephemeral=True)
        return

    claims[s]["color"] = color.strip()
    save_data(bot.data)
    await interaction.response.send_message(f"🎨 Updated color for **{s}** to **{color}**.")


@slot_group.command(name="forceclaimfor", description="Leadership: Overwrite an existing claim for a slot (Discord member).")
@app_commands.describe(member="Member", slot="Example: E6", color="Optional: preset or #RRGGBB")
async def slot_forceclaimfor(interaction: discord.Interaction, member: discord.Member, slot: str, color: str = ""):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    if color and parse_color(color) is None:
        await interaction.response.send_message(f"Invalid color. {color_help_text()}", ephemeral=True)
        return

    layout["claims"][s] = {"user_id": str(member.id), "user_name": member.display_name}
    if color:
        layout["claims"][s]["color"] = color.strip()
    save_data(bot.data)

    await interaction.response.send_message(
        f"✅ Force assigned **{s}** to **{member.display_name}**." + (f" Color: **{color}**." if color else "")
    )


@slot_group.command(name="forceclaimname", description="Leadership: Overwrite an existing claim for a slot (typed name).")
@app_commands.describe(slot="Example: E6", name="In-game name (symbols allowed)", color="Optional: preset or #RRGGBB")
async def slot_forceclaimname(interaction: discord.Interaction, slot: str, name: str, color: str = ""):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return
    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_err = validate_not_locked(layout, s)
    if lock_err:
        await interaction.response.send_message(lock_err, ephemeral=True)
        return

    cleaned = name.strip()
    if not cleaned:
        await interaction.response.send_message("Name cannot be empty.", ephemeral=True)
        return
    if len(cleaned) > 60:
        cleaned = cleaned[:60] + "…"

    if color and parse_color(color) is None:
        await interaction.response.send_message(f"Invalid color. {color_help_text()}", ephemeral=True)
        return

    layout["claims"][s] = {"user_id": None, "user_name": cleaned, "manual": True}
    if color:
        layout["claims"][s]["color"] = color.strip()
    save_data(bot.data)

    await interaction.response.send_message(
        f"✅ Force assigned **{s}** to **{cleaned}**." + (f" Color: **{color}**." if color else "")
    )


@slot_group.command(name="swap", description="Leadership: Swap two claimed slots instantly.")
@app_commands.describe(slot_a="Example: D7", slot_b="Example: E6")
async def slot_swap(interaction: discord.Interaction, slot_a: str, slot_b: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    a = normalize_slot(slot_a)
    b = normalize_slot(slot_b)
    if not a or not b:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, a) or validate_slot_in_grid(layout, b)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    lock_a = validate_not_locked(layout, a)
    lock_b = validate_not_locked(layout, b)
    if lock_a or lock_b:
        await interaction.response.send_message(lock_a or lock_b, ephemeral=True)
        return

    claims = layout["claims"]
    if a not in claims or b not in claims:
        await interaction.response.send_message("Both slots must be claimed to swap.", ephemeral=True)
        return

    claims[a], claims[b] = claims[b], claims[a]
    save_data(bot.data)

    await interaction.response.send_message(
        f"🔁 Swapped **{a}** ↔ **{b}**\n- {a}: **{claims[a]['user_name']}**\n- {b}: **{claims[b]['user_name']}**"
    )


@slot_group.command(name="unclaimfor", description="Leadership: Remove a claim from a slot.")
@app_commands.describe(slot="Example: D7")
async def slot_unclaimfor(interaction: discord.Interaction, slot: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    claims = layout["claims"]
    if s not in claims:
        await interaction.response.send_message("That slot is not claimed.", ephemeral=True)
        return

    removed = claims[s].get("user_name", "")
    del claims[s]
    save_data(bot.data)

    await interaction.response.send_message(f"🧹 Unclaimed **{s}** (was **{removed}**).")


@slot_group.command(name="label", description="Leadership: Add overlay label on a slot (shows on PNG).")
@app_commands.describe(slot="Example: E6", label="Example: RALLY")
async def slot_label(interaction: discord.Interaction, slot: str, label: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    err = validate_slot_in_grid(layout, s)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    layout.setdefault("slot_labels", {})
    layout["slot_labels"][s] = label.strip()
    save_data(bot.data)

    await interaction.response.send_message(f"🏷️ Label set: **{s}** → **{layout['slot_labels'][s]}**")


@slot_group.command(name="labelclear", description="Leadership: Clear overlay label from a slot.")
@app_commands.describe(slot="Example: E6")
async def slot_labelclear(interaction: discord.Interaction, slot: str):
    if not is_leadership(interaction.user):
        await interaction.response.send_message("Only Leadership can use this.", ephemeral=True)
        return

    gid = str(interaction.guild_id)
    layout = bot.data.get(gid)
    if not layout:
        await interaction.response.send_message("Run `/layout create` first.", ephemeral=True)
        return

    s = normalize_slot(slot)
    if not s:
        await interaction.response.send_message("Invalid slot format.", ephemeral=True)
        return

    labels = layout.setdefault("slot_labels", {})
    if s not in labels:
        await interaction.response.send_message("No label exists on that slot.", ephemeral=True)
        return

    del labels[s]
    save_data(bot.data)

    await interaction.response.send_message(f"🧽 Label cleared on **{s}**.")


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set. Set it in Railway Variables.")
    bot.run(token)
