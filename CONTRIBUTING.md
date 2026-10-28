# Contributing to AgentRisk

Thanks for helping improve AgentRisk. Start with the README, which documents
the session classes, the reproduce steps, and the evaluation protocol for
new policies.

## Checking a change

There is no automated test suite in this repo. To verify a code change,
re-run the documented reproduce steps from the README (for example,
`python run.py main`, which takes about 8 minutes on 2 cores) and confirm
the results still land in `results/`.

## Proposing a change

1. Fork the repo and create a branch for your change.
2. Make your change and verify it with the steps above.
3. Open a pull request describing what changed and why.

Please keep new data synthetic: every session in this repo is simulated,
and no real user data belongs here.
