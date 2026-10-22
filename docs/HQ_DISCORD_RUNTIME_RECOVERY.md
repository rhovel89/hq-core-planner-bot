# HQ Discord bot recovery — safe redeployment candidate

**Recovery branch only. Do not deploy until the existing Discord application's identity, current runtime file, and persistent volume are positively identified.** This branch retains the original command handlers; no saved assignments or player IDs were changed.

## Investigated state, October 21, 2026

- Original Discord bot: `rhovel89/hq-core-planner-bot`. Source implements `/layout storage`.
- **October 21 owner-provided Discord Application ID: `1456045977403130018`.** This is an owner identification statement, **not yet independently bound to a live Discord bot token, original Railway service or current runtime file**. This ID is not a secret and is safe to use for identification; the token is never shared in chat.
- The recovery branch now fails closed in `setup_hook` before any slash-command registration/synchronization if Discord reports a different or unavailable authenticated application ID. A match emits `HQ_APPLICATION_ID_VERIFIED app_id=1456045977403130018`; mismatch emits `HQ_APPLICATION_ID_MISMATCH` without leaking credentials. This prevents accidental command registration against an unrelated bot when credentials are configured in the host.
- Existing GitHub `hq_layouts.json`: historical guild key **`1438307693579206666`**, 12 assignments, 19 reference calibration points and E6 anchor. This is a **source-file guild ID, not the Discord Application ID**. It is not independently confirmed as the guild of the currently running bot, and this GitHub snapshot may **not** be the newest live data.
- Across the six Railway projects accessible to the owner's connected Railway account, no running service sourced from `hq-core-planner-bot` was found. An external service/account remains possible. Do not use unrelated `pns-alliance-bot` or `alliance-bot-prod` credentials.
- `/layout storage` timing out is consistent with a disconnected Discord process, but the exact application's current host/identity is not yet proven.

## What this branch changes

- `scripts/start_hq_worker.sh` is a portable guarded Python worker entrypoint. On the **verified original service**, set the Railway Start Command to `sh scripts/start_hq_worker.sh` after preserving the current layout. No service configuration has been changed automatically. Railway deprecated `railway.json`/`railway.toml` for new services and recommends modern Infrastructure as Code (`.railway/railway.ts`) when infrastructure ownership is established.
- `scripts/preflight_storage.py --require-volume` **fails closed before the bot starts** if the correct token is missing, the volume is not reported, the configured `DATA_FILE` is outside the volume, JSON is missing/invalid, or guild layouts are malformed.
- Preflight reads only and logs **record counts and a file SHA-256**, never names, guild IDs, Discord tokens, or layout contents.
- Layout saves use same-volume atomic file replacement and keep one automatically rotated `hq_layouts.json.bak` generation. A failed serialization or backup write leaves the live JSON untouched. The backup is a local fallback, not an offsite backup.
- CI runs Python syntax verification, synthetic storage validation, atomic-save, `/layout storage` handler, and startup/Discord readiness logging tests. No Discord login occurs in CI.
- The original bot now writes **sanitized startup milestones**: `HQ_SYNC_START scope=global|guild`, `HQ_SYNC_OK scope=... commands=N`, and `HQ_BOT_READY app_id=... guilds=N`. A failure to register Discord commands writes `HQ_SYNC_FAILED type=...` and fails startup. It does **not** log tokens, player names, saved assignments, or guild IDs. These logs are not a substitute for actual Discord command invocation.

This gate intentionally refuses to start a bot on an empty volume. **Do not bypass it by creating a new layout** or copying over a potentially newer live file.

## Connected Railway review — October 22, 2026

The owner's currently connected Railway workspace `rhovel89's Projects` exposes six projects; four contain application services. **None is a verified host of `rhovel89/hq-core-planner-bot`.** Don't mistake another bot's GitHub source, variables, or volume for the original Python bot:

| Railway project | Actual accessible source | Relevant constraint |
|---|---|---|
| `hopeful-forgiveness` | source unlinked | No attached volume; old deployments REMOVED; original bot identity unverified |
| `alliance-bot-prod` | `rhovel89/alliance-bot-prod` (Node) | Not the original Python `/layout storage` bot |
| `lucky-radiance` | `rhovel89/pns-alliance-bot` (Node) | Service currently offline. **Existing 500 MB volume `alliancebot-volume` at `/app/data` belongs to this Node app; latest failed deployment reports `MIGRATE_VOLUMES`.** Preserve it and do not attach it to the Python bot |
| `radiant-rebirth` | `rhovel89/pns-events-intel-bot` (Node) + Postgres | Unrelated events/intelligence bot and database; do not reconfigure |
| `hearty-dedication`, `helpful-smile` | no application services | Empty project alone is not proof of the original bot's latest persisted data |

The Railway OAuth connector reports the account **registered**, but does **not** expose confirmation of a paid plan or available deployment spending. The above are read-only observations, not authority to purchase, create a substitute bot, redeploy an unrelated service, or overwrite a volume.

## Offline/read-only storage inspection (new)

Run this only against a **confirmed current HQ bot layout file** from the correct original host. The newly added inspection mode intentionally does not need a bot token, does not contact Discord, and does not change file bytes:

```sh
# If examining a safely downloaded local copy (no Railway mount):
DATA_FILE=./hq-layouts-recovered.json python scripts/preflight_storage.py --inspect-only

# On the correctly identified Railway worker, for a volume already
# mounted at /app/data, with the live DATA_FILE verified first:
DATA_FILE=/app/data/hq_layouts.json python scripts/preflight_storage.py --require-volume --inspect-only
```

The output contains only safe counts, a file-size and SHA-256 fingerprint, not guild IDs or player names. Inspection **still fails closed** on missing volume, missing file, corrupt JSON or invalid guild layout shape. This mode does not bypass the production `sh scripts/start_hq_worker.sh` startup gate: normal startup still requires `DISCORD_BOT_TOKEN`, the verified mounted volume file, and the authenticated Discord application-ID match (`1456045977403130018`).

If the original HQ volume is found, Railway's official file-management CLI can download rather than upload/overwrite its contents. Use `railway volume files --volume <VERIFIED_HQ_VOLUME_ID> list /` and `railway volume files --volume <VERIFIED_HQ_VOLUME_ID> download /hq_layouts.json ./hq-layouts-recovered.json`; specify verified project/service/environment context and confirm the actual remote file path first. The known `alliancebot-volume` is **not** verified as this Python bot's data volume.

Source: Railway [volume documentation](https://docs.railway.com/volumes), [CLI file-management reference](https://docs.railway.com/cli/volume), and [volume backups](https://docs.railway.com/volumes/backups). Inspect and preserve backups before any recovery or volume reattachment.

## Safe recovery procedure

1. Use the owner-supplied **Discord Application ID `1456045977403130018`** as the expected existing HQ bot identity; verify it in the Discord Developer Portal. Do not paste its token into chat. The public ID does **not** establish where the bot is hosted.
2. Identify the bot's actual deployed host/service. If it is under another Railway workspace/account, gain appropriate access before recovery. Verify `DISCORD_BOT_TOKEN` belongs to this application; do not reuse tokens from unrelated Railway services.
3. Confirm or retrieve its **actual latest** `DATA_FILE` and preserve at least one offsite copy. If stored on a Railway volume, select the **correct** project, environment, and volume using the CLI:
   ```sh
   railway volume list
   railway volume files list /
   railway volume files download /hq_layouts.json ./hq_layouts-live.json
   ```
   These commands assume the selected volume is mounted at `/app/data` and the file is in its root. The GitHub snapshot is a separate fallback, not proof of the latest live state.
4. Compare the runtime copy with the GitHub snapshot using the existing read-only Command Center comparison tool. Its archived guild key is `1438307693579206666`; independently confirm that this is the exact Discord server where `/layout storage` is used, then verify the guild corresponds to the intended Supabase alliance. Review F6/F7 calibration mismatches in the game. Do **not** use the Application ID `1456045977403130018` as a guild ID, and do not auto-import the snapshot.
5. On the **verified original bot service**, ensure its existing persistent volume is mounted at `/app/data` and its latest file lives at `/app/data/hq_layouts.json`. Configure:
   - `DATA_FILE=/app/data/hq_layouts.json`
   - `DISCORD_BOT_TOKEN`: existing bot application's token, set in Railway's secrets UI; never disclose
   - `DISCORD_GUILD_ID`: optional ID of the existing server for guild-scoped sync
   - `RAILWAY_VOLUME_MOUNT_PATH`: should reflect Railway's volume mount; if unavailable, investigate instead of bypassing preflight
6. Only after the stored file is preserved and matches the target application, set Start Command to `sh scripts/start_hq_worker.sh`, then test this branch in the verified worker's deployment path. The code itself compares the token-authenticated application to `1456045977403130018` **before** registering commands, then emits `HQ_APPLICATION_ID_VERIFIED` if matched. A mismatch is a stop, not a reason to override the code or swap an unrelated token. Review storage preflight `PASS`, application-ID verification, `HQ_SYNC_OK`, and `HQ_BOT_READY` in order; none alone proves the Discord command responds.
7. Check the verified service's deployment logs **in this order**:
   - No preflight PASS: stopped at stored-file, volume, or credential validation. Do not bypass preflight.
   - `HQ_APPLICATION_ID_MISMATCH`: this token belongs to another/unknown application, or the expected ID is unavailable. **Stop before any slash-command sync**; do not use another bot's credentials.
   - `HQ_APPLICATION_ID_VERIFIED` followed by `HQ_SYNC_START` without `HQ_SYNC_OK`: the expected application is authenticated, but slash-command synchronization failed; review startup and permissions.
   - `HQ_SYNC_OK` without `HQ_BOT_READY`: synchronization completed, but gateway readiness is not verified; inspect Discord network/connection logs and application identity.
   - `HQ_BOT_READY` present: the original worker connected and reports an application ID, but command response still needs to be tested from the intended server. Compare application ID with the developer portal before accepting the deployment.
8. In Discord, run nondestructive `/layout storage` then `/slot coord`. Confirm an ephemeral response, runtime path, and that existing assignments remain unchanged. Monitor runtime logs and exit status.
9. Keep the Command Center `main` release and Supabase layout import on hold until real R5 and member tests, restore rehearsal, calibration, and explicit final release acceptance.

## Emergency stop

If the file is absent, malformed, or older than a known snapshot, **stop and preserve the data**. Do not run `/layout create` or a forced migration; that can replace a guild layout. No changes to unrelated Railway services or pending staged changes are authorized.

Railway references:
- [Railway Start Command](https://docs.railway.com/deployments/start-command) and [Railway Infrastructure as Code](https://docs.railway.com/infrastructure-as-code)
- [Railway volume file retrieval](https://docs.railway.com/cli/volume#download-a-file-from-a-volume)
