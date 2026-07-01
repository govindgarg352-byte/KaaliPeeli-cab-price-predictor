# =============================================================================
# holiday_fetcher.py — Fetch Indian Public Holidays via Calendarific API
#
# Makes exactly ONE API call → saves holidays to data/raw/holidays_2026.json
# All other scripts import from that JSON — no repeat API calls ever needed
#
# Usage:
#   python data_collection/holiday_fetcher.py
# =============================================================================

import os
import json
import requests
from datetime import datetime

# Import API key from config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import API_KEYS

# =============================================================================
# CONFIGURATION
# =============================================================================

CALENDARIFIC_URL = "https://calendarific.com/api/v2/holidays"
COUNTRY          = "IN"
YEAR             = 2026
OUTPUT_PATH      = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'raw', 'holidays_2026.json'
)

# =============================================================================
# FETCH HOLIDAYS
# =============================================================================

def fetch_holidays() -> list[dict]:
    """
    Calls Calendarific API and returns list of Indian public holidays for 2026.
    Returns list of dicts with keys: name, date (YYYY-MM-DD), type
    """
    print(f"Fetching Indian holidays for {YEAR} from Calendarific...")

    params = {
        "api_key": API_KEYS["calendarific"],
        "country":  COUNTRY,
        "year":     YEAR,
        "type":     "national",          # national public holidays only
    }

    response = requests.get(CALENDARIFIC_URL, params=params, timeout=10)

    if response.status_code != 200:
        raise Exception(
            f"Calendarific API error: {response.status_code} — {response.text}"
        )

    data = response.json()

    # Check API-level errors
    if data.get("meta", {}).get("code") != 200:
        raise Exception(f"Calendarific returned error: {data}")

    raw_holidays = data["response"]["holidays"]
    print(f"  → {len(raw_holidays)} holidays returned from API")

    return raw_holidays


# =============================================================================
# PARSE & CLEAN
# =============================================================================

def parse_holidays(raw_holidays: list[dict]) -> list[str]:
    """
    Extracts just the date strings (YYYY-MM-DD) from raw API response.
    This is all we need — a flat list to check 'is this date a holiday?'
    """
    holiday_dates = []

    for h in raw_holidays:
        try:
            date_str = h["date"]["iso"][:10]   # take YYYY-MM-DD, drop time
            holiday_dates.append(date_str)
            print(f"  ✅ {date_str} — {h['name']}")
        except KeyError:
            print(f"  ⚠️  Skipping malformed entry: {h}")

    return sorted(set(holiday_dates))           # deduplicate + sort


# =============================================================================
# SAVE TO FILE
# =============================================================================

def save_holidays(holiday_dates: list[str]) -> None:
    """Saves holiday dates to JSON for use by all other scripts."""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    output = {
        "country":       COUNTRY,
        "year":          YEAR,
        "fetched_at":    datetime.now().isoformat(),
        "total":         len(holiday_dates),
        "holiday_dates": holiday_dates,         # ["2026-01-26", "2026-03-25", ...]
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Saved {len(holiday_dates)} holiday dates → {OUTPUT_PATH}")


# =============================================================================
# LOAD HOLIDAYS (used by all other scripts — no API call)
# =============================================================================

def load_holidays() -> set[str]:
    """
    Loads holiday dates from saved JSON.
    Returns a set for O(1) lookup: '2026-01-26' in holidays

    Usage in other scripts:
        from data_collection.holiday_fetcher import load_holidays
        HOLIDAYS = load_holidays()
        is_holiday = row_date in HOLIDAYS   # True / False
    """
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(
            f"Holiday file not found at {OUTPUT_PATH}\n"
            f"Run: python data_collection/holiday_fetcher.py"
        )

    with open(OUTPUT_PATH, "r") as f:
        data = json.load(f)

    return set(data["holiday_dates"])


# =============================================================================
# HELPER: check if a specific date is a holiday
# =============================================================================

def is_holiday(date_str: str, holidays: set[str] = None) -> bool:
    """
    Check if a given date string (YYYY-MM-DD) is an Indian public holiday.

    Args:
        date_str: Date in 'YYYY-MM-DD' format
        holidays: Optional pre-loaded set (pass to avoid reloading each time)

    Returns:
        True if holiday, False otherwise

    Usage:
        # Load once outside your loop
        HOLIDAYS = load_holidays()

        # Check inside loop
        flag = is_holiday("2026-01-26", HOLIDAYS)   # True (Republic Day)
        flag = is_holiday("2026-01-27", HOLIDAYS)   # False
    """
    if holidays is None:
        holidays = load_holidays()
    return date_str in holidays


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  HOLIDAY FETCHER — Indian Public Holidays 2026")
    print("=" * 60)

    # Step 1: Fetch
    raw = fetch_holidays()

    # Step 2: Parse
    dates = parse_holidays(raw)

    # Step 3: Save
    save_holidays(dates)

    # Step 4: Quick sanity check
    print("\n📅 Sanity Check — Loading back from file:")
    holidays_set = load_holidays()
    print(f"  Total holidays loaded: {len(holidays_set)}")

    # Test a known holiday
    test_date = f"{YEAR}-01-26"
    print(f"  Is {test_date} (Republic Day) a holiday? → {is_holiday(test_date, holidays_set)}")

    print("\n✅ Holiday fetcher complete!")
    print(f"   Import in other scripts with:")
    print(f"   from data_collection.holiday_fetcher import load_holidays, is_holiday")
