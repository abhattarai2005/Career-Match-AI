

import html
import re
from urllib.parse import urlencode

import requests

REMOTIVE_API = "https://remotive.com/api/remote-jobs"


def linkedin_search_link(keywords: str, location: str) -> str:
    """Link to LinkedIn's own real job-search interface (no scraping)."""
    params = {"keywords": keywords.strip(), "location": location.strip()}
    return "https://www.linkedin.com/jobs/search/?" + urlencode(params)


def clean_html(value: str) -> str:
    """Remove basic HTML tags so job descriptions can be analyzed as text."""
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value or "", flags=re.I | re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def fetch_remote_jobs(limit: int = 400) -> list[dict]:
    """Retrieve real public remote job listings from Remotive.

    The calling UI must cache results. Remotive recommends no more
    than four requests a day. Their listings can lag by 24 hours.
    """
    response = requests.get(
        REMOTIVE_API,
        params={"limit": limit},
        timeout=20,
        headers={"Accept": "application/json", "User-Agent": "CareerMatchAI portfolio project"},
    )
    response.raise_for_status()
    payload = response.json()
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Unexpected response format from Remotive")
    return jobs


def filter_remote_jobs(jobs: list[dict], keywords: str) -> list[dict]:
    """Search titles and categories locally (no resume sent to provider)."""
    words = [word for word in re.findall(r"[\w+#.]+", keywords.lower())
             if word not in {"junior", "entry", "level", "remote", "jobs", "job", "and", "or"}]
    if not words:
        return list(jobs)
    matches = []
    for job in jobs:
        title = str(job.get("title", "")).lower()
        category = str(job.get("category", "")).lower()
        text = title + " " + category
        if any(word in text for word in words):
            matches.append(job)
    return matches
