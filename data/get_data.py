"""Download the IBM Telco Customer Churn sample dataset.

Source: IBM sample data repository on GitHub (telco-customer-churn-on-icp4d).
URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Expected SHA-256 of the downloaded file:
    16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
Expected size: 7,043 data rows, 21 columns.

The raw CSV is written to data/raw/telco_churn.csv. Run from the repo root:

    python data/get_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
OUT_PATH = Path(__file__).resolve().parent / "raw" / "telco_churn.csv"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, OUT_PATH)
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    print(f"SHA-256: {digest}")
    if digest != EXPECTED_SHA256:
        print(
            "Warning: checksum differs from the recorded value. "
            "The upstream file may have changed; verify before trusting results.",
            file=sys.stderr,
        )
        return 1
    print("Checksum matches the recorded value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
