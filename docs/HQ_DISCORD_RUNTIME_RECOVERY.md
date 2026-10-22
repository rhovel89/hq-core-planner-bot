# HQ Discord bot recovery — safe redeployment candidate

**Recovery branch only. Do not deploy until the existing Discord application's identity, current runtime file, and persistent volume are positively identified.** This branch retains the original command handlers; no saved assignments or player IDs were changed.

## Investigated state, October 21, 2026

- Original Discord bot: `rhovel89/hq-core-planner-bot`. Source implements `/layout storage`.
- Existing GitHub `hq_layouts.json`: 1 guild, 12 assignments, 19 reference calibration points, E6 anchor. This may **not** be the newest live data.
- Across the six Railway projects accessible to the owner's connected Railway account, no running service sourced from `hq-core-planner-bot` was found. An external service/account remains possible. Do not use unrelated `pns-alliance-bot` or `alliance-bot-prod` credentials.
- `/layout storage` timing out is consistent with a disconnected Discord process, but the exact application's current host/identity is not yet proven.

## What this branch changes

- `railway.json` supplies an explicit Python worker start command, rather than relying on automatic entrypoint inference.
- `scripts/preflight_storage.py --require-volume` **fails closed before the bot starts** if the correct token is missing, the volume is not reported, the configured `DATA_FILE` is outside the volume, JSON is missing/invalid, or guild layouts are malformed.
- Preflight reads only and logs **record counts and a file SHA-256**, never names, guild IDs, Discord tokens, or layout contents.
- CI runs Python syntax verification and synthetic preflight tests. No Discord login occurs in CI.

This gate intentionally refuses to start a bot on an empty volume. **Do not bypass it by creating a new layout** or copying over a potentially newer live file.

## Safe recovery procedure

1. Determine the exact **existing** Discord bot application (Discord Developer Portal application ID; never paste its token into a chat).
2. Identify the bot's actual deployed host/service. If it is under another Railway workspace/account, gain appropriate access before recovery. Verify `DISCORD_BOT_TOKEN` belongs to this application; do not reuse tokens from unrelated Railway services.
3. Confirm or retrieve its **actual latest** `DATA_FILE` and preserve at least one offsite copy. If stored on a Railway volume, select the **correct** project, environment, and volume using the CLI:
   ```sh
   railway volume list
   railway volume files list /
   railway volume files download /hq_layouts.json ./hq_layouts-live.json
   ```
   These commands assume the selected volume is mounted at `/app/data` and the file is in its root. The GitHub snapshot is a separate fallback, not proof of the latest live state.
4. Compare the runtime copy with the GitHub snapshot using the existing read-only Command Center comparison tool. Verify the Discord guild corresponds to the intended Supabase alliance; review F6/F7 calibration mismatches in the game.
5. On the **verified original bot service**, ensure its existing persistent volume is mounted at `/app/data` and its latest file lives at `/app/data/hq_layouts.json`. Configure:
   - `DATA_FILE=/app/data/hq_layouts.json`
   - `DISCORD_BOT_TOKEN`: existing bot application's token, set in Railway's secrets UI; never disclose
   - `DISCORD_GUILD_ID`: optional ID of the existing server for guild-scoped sync
   - `RAILWAY_VOLUME_MOUNT_PATH`: should reflect Railway's volume mount; if unavailable, investigate instead of bypassing preflight
6. Only after the stored file is preserved and matches the target application, test this branch in the verified worker's deployment path. Review logs for `PASS` from the preflight; a passing check does **not** guarantee Discord connectivity.
7. In Discord, run nondestructive `/layout storage` then `/slot coord`. Confirm an ephemeral response, runtime path, and that existing assignments remain unchanged. Monitor runtime logs and exit status.
8. Keep the Command Center `main` release and Supabase layout import on hold until real R5 and member tests, restore rehearsal, calibration, and explicit final release acceptance.

## Emergency stop

If the file is absent, malformed, or older than a known snapshot, **stop and preserve the data**. Do not run `/layout create` or a forced migration; that can replace a guild layout. No changes to unrelated Railway services or pending staged changes are authorized.

Railway references:
- [Railway config-as-code start command](https://docs.railway.com/config-as-code/reference#start-command)
- [Railway volume file retrieval](https://docs.railway.com/cli/volume#download-a-file-from-a-volume)
