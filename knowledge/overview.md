# Project Atlas

Project Atlas is a fictional internal tool used to coordinate data ingestion and policy review.
It consists of two phases: Harvest and Clarify.

## Harvest
- Collects text data from partners
- Normalizes it into a single schema
- Produces a daily snapshot

## Clarify
- Runs compliance checks
- Generates a summary report
- Requires manual signoff by a reviewer

The main CLI command is `atlas sync` which runs both phases.
