# Security policy

Security fixes currently target the `main` branch. There is no long-term support release or hosted service.

Report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/ahmedjawedaj/open-wm-vlm/security/advisories/new). Include a minimal reproduction, affected revision and expected impact. Avoid including credentials or personal data. Use public issues for ordinary correctness bugs that do not require coordinated disclosure.

This repository currently generates and audits synthetic local datasets. Model weights and inference services are not bundled. Review third-party dependency and model licenses before adding them. Keep secrets in local environment variables and out of tracked files.

Updates are reviewed through pull requests and CI. A security report does not imply a response-time guarantee.
