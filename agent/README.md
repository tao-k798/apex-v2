# sn66 miner agent

A miner agent for **Bittensor subnet 66 (tau)**. Built on top of the [pi-mono](https://github.com/badlogic/pi-mono) coding agent and aligned with the [unarbos/tau](https://github.com/unarbos/tau) harness.

The harness clones this repo at a pinned commit, builds `packages/coding-agent`, and runs it against SWE tasks generated from real GitHub change sets. Scoring is positional line-level exact matching of unified diffs against a hidden reference solution. See [AGENTS.md](AGENTS.md) for the full operating contract that the agent loads as system context on every run.

## Layout

This is a pi-mono workspace; the harness expects `packages/coding-agent` at the repo root.

| Package | Role |
|---------|------|
| [`packages/coding-agent`](packages/coding-agent) | The agent CLI invoked by the tau harness |
| [`packages/ai`](packages/ai) | Multi-provider LLM client |
| [`packages/agent`](packages/agent) | Agent runtime, tool calling, state |
| [`packages/tui`](packages/tui) | Terminal UI primitives (build dependency) |
| [`packages/pods`](packages/pods), [`mom`](packages/mom), [`web-ui`](packages/web-ui) | Unused on-chain; kept so the workspace builds |

## Local development

```bash
npm install
npm run build
```

## License

MIT — see [LICENSE](LICENSE).
