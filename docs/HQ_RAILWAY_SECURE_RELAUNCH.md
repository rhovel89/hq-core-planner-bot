# Secure Railway relaunch for the original State 789 HQ Discord bot
**Status: HOLD — Railway-wide Oct 21–22, 2026 incident, new-project provisioning unavailable when attempted; no live host identified.**

This is a recovery procedure for **the original Python `rhovel89/hq-core-planner-bot`**, Discord Application ID `1456045977403130018` (owner-provided). The separate Node bots and their volumes are NOT substitutes. Neither the archived GitHub layout nor the present Command Center Supabase layout is proof of the bot's latest data.

## 0. Incident/security gate — PRIORITY

Railway published a security disclosure on Oct 21, 2026 reporting possible access to customer environment variables (some potentially decrypted). They recommend all users rotate environment-variable credentials and inspect logs.

Official disclosure: https://blog.railway.com/p/security-disclosure-wednesday-october-21st

Official incident status: https://status.railway.com/

Before any live bot deployment:
- Confirm Railway's build/deploy service is restored **for the account's actual plan**; the creation API previously returned `Deploys have been paused due to an upstream issue`.
- Rotate all secrets previously stored in Railway **at their issuing providers**. Prioritize old Discord bot tokens, Discord OAuth client secrets, Supabase service-role credentials, database credentials, app session secrets, and dashboard passwords, where applicable.
- Rotate / recreate Railway API tokens as required; Railway reports previous tokens invalidated. Reauthorize necessary integrations safely if invalidated.
- Do not copy old Railway secret values to a new project, place credentials in GitHub, display them in logs, or paste them into chat.
- Determine whether any connected downstream apps require coordinated secret changes before revoking shared credentials. **Do not revoke production database or service-role keys without an outage-safe rotation plan.**
- Review available provider audit/activity logs for suspicious access.

The incident does **not** establish that this workspace was compromised; treat the rotation as a precaution consistent with Railway's official recommendation.

## 1. Identity and data recovery gates

1. In Discord Developer Portal verify existing HQ bot Application ID exactly `1456045977403130018`. Confirm the actual intended server separately; the source's historic guild ID `1438307693579206666` is not an Application ID.
2. Identify and preserve the **latest** bot `DATA_FILE`. Its content and volume are not verified in the currently connected Railway workspace. A committed `hq_layouts.json` with 12 historic assignments is an **archival snapshot only**.
3. Save at least one backup outside Railway and record its bytes, SHA-256, guild count and claim count. Do not overwrite the live/historical file and do not assume the separate Node bot's `alliancebot-volume` contains HQ Core Planner layouts.
4. Use `python scripts/preflight_storage.py --inspect-only` against a safely copied layout file. For an actual mounted volume, add `--require-volume`. Inspection never reads a Discord token or writes any data.
5. Verify the source guild and coordinates are the correct intended alliance and map; keep the current web Command Center layout independent.

## 2. Create only an isolated new project — once Railway is available

Use private project name `state789-hq-core-planner` in the owner's connected personal Railway workspace. Do not connect or change existing services.

- One dedicated worker named `hq-core-planner-worker` sourced only from `rhovel89/hq-core-planner-bot`, branch `repair/hq-runtime-20261021` until independently accepted. Pin the reviewed SHA if Railway supports it.
- Provision a **new dedicated** persistent volume for this worker at `/app/data`. Do NOT attach an existing bot or database volume.
- Before starting the worker, restore the verified latest backup to `/app/data/hq_layouts.json` on its own persistent volume, confirm hash, shape and permissions, and record an offsite copy.
- Set `DATA_FILE=/app/data/hq_layouts.json`.
- Set `DISCORD_BOT_TOKEN` using a **fresh credential from the original verified Discord application**, through Railway's secure variables UI. **Do not set it from another bot's variables.**
- Optional: set `DISCORD_GUILD_ID` only after checking the actual intended server ID.
- Start command: `sh scripts/start_hq_worker.sh`.
- Do not expose a public HTTP domain: this is a Discord gateway worker, not an HTTP server. Begin with one replica and no cron.
- Confirm expected Railway volume mount path is reported to the worker (`RAILWAY_VOLUME_MOUNT_PATH`), and preflight sees `DATA_FILE` inside that volume.
- Review Railway charges/plan separately before enabling resources. Do not change the account billing plan automatically.

The worker **intentionally fails closed** if any prerequisite is missing. A stopped deployment is safer than silently creating an empty layout or deploying with the wrong application identity.

## 3. Live acceptance after recovery

Review sanitized startup logs, in order:
```text
HQ preflight: READ ONLY
PASS: <N> guild layout(s), <N> saved assignment(s)
HQ_APPLICATION_ID_VERIFIED app_id=1456045977403130018
HQ_SYNC_START
HQ_SYNC_OK
HQ_BOT_READY app_id=1456045977403130018
```

Any `HQ_APPLICATION_ID_MISMATCH`, missing volume, invalid JSON, absent secret, sync failure, or bot not ready is a HARD STOP: do not weaken the guard or replace the layout.

From the intended Discord server, run nondestructive `/layout storage` and `/slot coord`. Confirm an ephemeral response and match actual live layouts with the preserved backup. Existing assignments must remain intact.

Only then consider connecting bot exports to the Command Center. Guild→alliance mapping, real R5/member access, Supabase backup/restore, in-game calibration and release QA remain independently required.

## Last verified outcome

- Oct 22, 2026: creating `state789-hq-core-planner` via the connected Railway API returned `Deploys have been paused due to an upstream issue`; subsequent listing confirmed no project was created.
- Four connected application services are unrelated to the original Python repo. The existing `pns-alliance-bot` is offline, with a preserved separate volume and last failure stage `MIGRATE_VOLUMES`.
- Recovery branch offline CI passed before the blocked deployment attempt; **no Railway or Discord live integration test occurred**.
