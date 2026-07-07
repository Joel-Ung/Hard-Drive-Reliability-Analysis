# Hard-Drive-Reliability-Analysis
Dataset: Backblaze's Public Hard Drive Failure Dataset

Among drive models with sufficient failure events (n >= Y failures), which SMART attributes — measured at each drive's most recent reading in the quarter — are most associated with whether that drive ever failed during Q1 2025?

**Target variable:** binary — did the drive ever fail during Q1 2025 (yes/no), using its last available SMART reading in the quarter as the feature snapshot.

**20 (minimum failure threshold):** A common rule of thumb is ~20-30 failure events per group before treating a comparison as meaningful.*



SMART stands for Self-Monitoring, Analysis, and Reporting Technology. A "drive dataset" refers to the continuous logs and internal metrics recorded by the drive's firmware to predict failures and monitor the storage device's overall health.

SMART attributes act as a drive's medical chart, tracking dozens of specific, numerical performance and health indicators. When analysts refer to a "drive dataset" (such as the popular Backblaze or NASA datasets used for machine learning and predictive maintenance), they are looking at these exact parameters over time to detect impending failures.