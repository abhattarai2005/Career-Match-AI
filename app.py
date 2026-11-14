

import html
import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from job_sources import (
    clean_html,
    fetch_remote_jobs,
    filter_remote_jobs,
    linkedin_search_link,
)

# --------------------------------------------------------------
# 1. FILES AND APPLICATION SETTINGS
# --------------------------------------------------------------
BASE = Path(__file__).resolve().parent
JOBS_FILE = BASE / "data" / "jobs.csv"
RESUME_FILE = BASE / "data" / "sample_resume.txt"
STYLE_FILE = BASE / "assets" / "styles.css"

st.set_page_config(
    page_title="CareerMatch AI • Career Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

if STYLE_FILE.exists():
    st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

if not JOBS_FILE.exists() or not RESUME_FILE.exists():
    st.error("Project data files are missing. Keep the 'data' folder next to app.py.")
    st.stop()

jobs = pd.read_csv(JOBS_FILE)
sample_resume = RESUME_FILE.read_text(encoding="utf-8")

# --------------------------------------------------------------
# 2. THE SMALL MACHINE-LEARNING / TEXT-ANALYSIS ENGINE
# --------------------------------------------------------------
# Named skills and the words we recognize for each one.
SKILLS = {
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b"],
    "Excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "Power BI": [r"\bpower\s*bi\b"],
    "Tableau": [r"\btableau\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Scikit-learn": [r"\bscikit[- ]learn\b", r"\bsklearn\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical analysis\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml models?\b"],
    "Data Cleaning": [r"\bdata clean(?:ing)?\b", r"\bdata preprocessing\b"],
    "Data Visualization": [r"\bdata visuali[sz]ation\b", r"\bvisuali[sz]e data\b"],
    "Git": [r"\bgit\b", r"\bgithub\b"],
    "PyTorch": [r"\bpytorch\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Streamlit": [r"\bstreamlit\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Azure": [r"\bazure\b"],
    "PostgreSQL": [r"\bpostgres(?:ql)?\b"],
    "ETL": [r"\betl\b", r"\bextract transform load\b"],
    "Airflow": [r"\bairflow\b"],
    "Docker": [r"\bdocker\b"],
    "APIs": [r"\bapis?\b", r"\brest api\b"],
    "PowerShell": [r"\bpowershell\b"],
    "Troubleshooting": [r"\btroubleshoot(?:ing)?\b"],
    "Communication": [r"\bcommunication\b", r"\bstakeholder presentations?\b"],
}


def find_skills(text: str) -> set[str]:
    """Find recognized skill names in a resume or job description."""
    lower_text = text.lower()
    return {
        skill for skill, patterns in SKILLS.items()
        if any(re.search(pattern, lower_text) for pattern in patterns)
    }


def text_similarity(resume: str, description: str) -> float:
    """TF-IDF makes numerical word features; cosine similarity compares them."""
    try:
        vectors = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2)
        ).fit_transform([resume, description])
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except ValueError:
        return 0.0


def analyze(resume: str, description: str) -> dict:
    """Return easy-to-explain skill coverage and text similarity."""
    resume_skills = find_skills(resume)
    job_skills = find_skills(description)
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    return {
        "matched": matched,
        "missing": missing,
        "coverage": len(matched) / len(job_skills) if job_skills else 0.0,
        "similarity": text_similarity(resume, description),
        "required_count": len(job_skills),
        "resume_skills": sorted(resume_skills),
    }


def read_resume(upload) -> str:
    """Extract text from TXT, DOCX and text-based PDF files."""
    raw = upload.getvalue()
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Please upload a file smaller than 5 MB.")
    ext = Path(upload.name).suffix.lower()
    if ext == ".txt":
        return raw.decode("utf-8-sig", errors="replace")
    if ext == ".docx":
        doc = Document(io.BytesIO(raw))
        lines = [paragraph.text for paragraph in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                lines.extend(cell.text for cell in row.cells)
        return "\n".join(lines)
    if ext == ".pdf":
        pdf = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
    raise ValueError("Upload a PDF, DOCX or TXT resume.")


def make_report(role: str, result: dict) -> str:
    """A readable text summary that users can download."""
    return (
        f"CareerMatch AI Pro | {role}\n"
        "========================================\n"
        f"Skill coverage: {result['coverage']:.0%}\n"
        f"Matched: {len(result['matched'])}/{result['required_count']} recognized skills\n"
        f"TF-IDF similarity: {result['similarity']:.1%}\n"
        f"Shared skills: {', '.join(result['matched']) or 'None'}\n"
        f"Not detected: {', '.join(result['missing']) or 'None'}\n\n"
        "These are approximate text/keyword metrics, NOT ATS scores,\n"
        "hiring odds, or a verified assessment of skills.\n"
    )


# --------------------------------------------------------------
# 3. SMALL VISUAL COMPONENTS (HTML is only for presentation)
# --------------------------------------------------------------
def safe(value) -> str:
    """Escape external words before putting them in HTML."""
    return html.escape(str(value), quote=True)


def section(icon: str, title: str, description: str):
    st.markdown(
        f'<div class="cm-section-top"><span class="cm-section-icon">{safe(icon)}</span>'
        f'<span class="cm-section-title">{safe(title)}</span></div>'
        f'<div class="cm-section-subtitle">{safe(description)}</div>',
        unsafe_allow_html=True,
    )


def tags(names, variant="") -> str:
    if not names:
        return '<span class="cm-tag neutral">Nothing detected</span>'
    return '<div class="cm-tags">' + ''.join(
        f'<span class="cm-tag {variant}">{safe(name)}</span>' for name in names
    ) + '</div>'


def stat_card(label: str, number: str, subtitle: str, theme: str = "") -> str:
    return (
        f'<div class="cm-stat {theme}"><div class="cm-stat-label">{safe(label)}</div>'
        f'<div class="cm-stat-number">{safe(number)}</div>'
        f'<div class="cm-stat-foot">{safe(subtitle)}</div></div>'
    )


def render_stats(result: dict):
    content = (
        stat_card("Recognized skill coverage", f"{result['coverage']:.0%}",
                  "Share of known job skills mentioned")
        + stat_card("Text similarity", f"{result['similarity']:.1%}",
                    "TF-IDF cosine similarity", "teal")
        + stat_card("Matched skills", str(len(result["matched"])) +
                    f"/{result['required_count']}", "Shared recognized keywords", "gold")
    )
    st.markdown(f'<div class="cm-stats">{content}</div>', unsafe_allow_html=True)


def ranking_table(resume: str) -> pd.DataFrame:
    rows = []
    for _, job in jobs.iterrows():
        result = analyze(resume, job["description"])
        rows.append({
            "Job title": job["role"],
            "Skills matched": f"{len(result['matched'])}/{result['required_count']}",
            "Coverage (%)": round(result["coverage"] * 100),
            "Similarity (%)": round(result["similarity"] * 100, 1),
        })
    return pd.DataFrame(rows).sort_values(
        ["Coverage (%)", "Similarity (%)"], ascending=False
    )


@st.cache_data(ttl=21600, show_spinner=False)
def cached_remote_jobs():
    """Keep Remotive fetches to roughly four per day or fewer per app cache."""
    return fetch_remote_jobs()


# --------------------------------------------------------------
# 4. THE MAIN DASHBOARD
# --------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="cm-sidebar-logo"><span class="cm-logo-square">✦</span>'
        '<div class="cm-brand">CareerMatch AI<small>CAREER INTELLIGENCE</small></div></div>'
        '<div class="cm-sidebar-section">YOUR WORKSPACE</div>'
        '<div class="cm-nav-item active">🎯 &nbsp; Resume match studio</div>'
        '<div class="cm-nav-item">💼 &nbsp; Job discovery</div>'
        '<div class="cm-nav-item">📈 &nbsp; Career insights</div>'
        '<div class="cm-sidebar-section">HOW TO GET STARTED</div>'
        '<div class="cm-step"><span class="cm-step-num">1</span><span>Upload or paste your resume.</span></div>'
        '<div class="cm-step"><span class="cm-step-num">2</span><span>Select a job or paste its description.</span></div>'
        '<div class="cm-step"><span class="cm-step-num">3</span><span>Analyze your match and explore jobs.</span></div>'
        '<div class="cm-privacy"><strong>🔒 Privacy first</strong>Resume analysis stays in the local app. '
        'Do not commit personal resumes to GitHub. LinkedIn login happens on LinkedIn itself.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("CAREERMATCH AI PRO • V3")
    st.caption("Built with Python · Streamlit · scikit-learn")

st.markdown(
    '<section class="cm-hero">'
    '<div class="cm-hero-kicker">✦ YOUR PERSONAL CAREER INTELLIGENCE STUDIO</div>'
    '<h1>Make every application<br><em>a smarter move.</em></h1>'
    '<p>Analyze your resume, uncover skill gaps, and discover relevant remote roles — '
    'all from one thoughtfully designed workspace.</p>'
    '<div class="cm-hero-badges"><span>✦ Resume intelligence</span>'
    '<span>◈ Skill gap insights</span><span>↗ Real job discovery</span></div>'
    '</section>', unsafe_allow_html=True,
)

st.markdown(
    '<div class="cm-feature-row">'
    '<div class="cm-mini-feature"><span class="ico">🎯</span><span><b>Understand your fit</b><small>Compare skills and job language</small></span></div>'
    '<div class="cm-mini-feature"><span class="ico">⚡</span><span><b>Move with clarity</b><small>See what to improve next</small></span></div>'
    '<div class="cm-mini-feature"><span class="ico">🌍</span><span><b>Explore opportunities</b><small>LinkedIn search + remote jobs</small></span></div>'
    '</div>', unsafe_allow_html=True,
)

if "saved_jobs" not in st.session_state:
    st.session_state["saved_jobs"] = {}

match_tab, jobs_tab, insights_tab = st.tabs([
    "🎯  RESUME STUDIO", "💼  JOB EXPLORER", "📈  MY INSIGHTS"
])

# ------------------------- TAB 1 -------------------------------
with match_tab:
    section("🧠", "Resume Match Studio", "Understand what your resume says about you — and where it could be stronger.")
    role = st.selectbox("Choose a sample job", jobs["role"].tolist(), key="sample_role")
    selected_description = jobs.loc[jobs["role"] == role, "description"].iloc[0]

    with st.form("resume_match_form"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 📄 Your resume")
            upload = st.file_uploader(
                "Upload a PDF, Word or TXT resume", type=["pdf", "docx", "txt"]
            )
            entered_resume = st.text_area(
                "Or paste your resume", value=sample_resume, height=235,
                help="The uploaded file is used instead of this text when one is provided.",
            )
            st.caption("PDF, DOCX, TXT • 5 MB maximum • Your file stays local to this app")
        with right:
            st.markdown("#### 💼 Job description")
            entered_job = st.text_area(
                "Paste a job description or edit this example",
                value=selected_description, height=314, key=f"description_{role}",
            )
            st.caption("For a real job, paste the full description so there is enough context.")
        submitted = st.form_submit_button("✨ Analyze my resume", type="primary", use_container_width=True)

    if submitted:
        try:
            resume = read_resume(upload) if upload else entered_resume
            if not resume.strip() or not entered_job.strip():
                st.error("Please provide both readable resume text and a job description.")
            else:
                result = analyze(resume, entered_job)
                st.session_state["last_analysis"] = {
                    "role": role,
                    "resume": resume,
                    "description": entered_job,
                    "result": result,
                }
                st.toast("Your comparison is ready!", icon="✅")
        except Exception as exc:
            st.error(f"Could not analyze that resume: {exc}")

    if "last_analysis" not in st.session_state:
        st.markdown(
            '<div class="cm-blank"><span class="emoji">✨</span>'
            '<strong>Your match results will appear here</strong>'
            '<small>Upload a resume, choose a job, then select Analyze my resume.</small></div>',
            unsafe_allow_html=True,
        )
    else:
        saved = st.session_state["last_analysis"]
        result = saved["result"]
        st.markdown('<div class="cm-eyebrow">YOUR PERSONALIZED BREAKDOWN</div>', unsafe_allow_html=True)
        section("📊", "Your Match Snapshot", f"Compared against: {saved['role']}")
        render_stats(result)
        st.markdown(
            '<div class="cm-panel-title">Recognized skills coverage</div>'
            f'<div class="cm-score-track"><div style="width:{result["coverage"] * 100:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                st.markdown("#### ✅ Skills found in both")
                st.markdown(tags(result["matched"]), unsafe_allow_html=True)
                st.caption("These skill terms appear in both the resume and description.")
        with b:
            with st.container(border=True):
                st.markdown("#### ✨ Skills to investigate")
                st.markdown(tags(result["missing"], "missing"), unsafe_allow_html=True)
                st.caption("Not found as keywords; this does not prove you lack the skills.")
        if result["missing"]:
            st.info("**Your next move:** If you really know a missing skill, add a truthful project example to your resume. Otherwise, practice it before claiming it.")
        elif result["required_count"] == 0:
            st.warning("No recognized skills appeared in that description. Try a more detailed job posting.")
        else:
            st.success("Every recognized skill in this job text was found in your resume. Check the actual experience requirements too.")

        st.download_button(
            "⬇ Download my analysis report", data=make_report(saved["role"], result),
            file_name="careermatch_analysis.txt", mime="text/plain",
        )
        with st.expander("📋 Compare my resume with all sample roles"):
            st.dataframe(ranking_table(saved["resume"]), hide_index=True, use_container_width=True)
        st.caption("Skill coverage and cosine similarity are NOT an ATS score or chance of being hired.")

# ------------------------- TAB 2 -------------------------------
with jobs_tab:
    section("🌍", "Discover Real Opportunities", "Search LinkedIn directly, or rank available Remotive remote jobs against your resume.")
    st.markdown('<div class="cm-eyebrow">01 / SEARCH ON LINKEDIN</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 🔗 Search LinkedIn job listings")
        x, y = st.columns(2, gap="medium")
        with x:
            linkedin_keywords = st.text_input("Job title / keywords", value="Junior Data Analyst", key="li_keywords")
        with y:
            linkedin_location = st.text_input("Preferred location", value="United States", key="li_location")
        st.link_button(
            "↗ Open jobs on LinkedIn",
            linkedin_search_link(linkedin_keywords, linkedin_location),
            type="primary", use_container_width=True,
        )
        st.caption("Opens LinkedIn's real search page. Your LinkedIn account stays on LinkedIn — no password sharing or scraping.")

    st.markdown('<div class="cm-eyebrow">02 / AI-ASSISTED REMOTE JOB DISCOVERY</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### ✨ Explore remote openings")
        col1, col2 = st.columns([3, 1], gap="medium")
        with col1:
            search = st.text_input("Search job titles", "data", key="remote_search", help="Try data, software, engineer, or python")
        with col2:
            limit = st.selectbox("Show top", [5, 10, 15], index=1, key="remote_limit")
        use_own = "last_analysis" in st.session_state
        st.caption(
            "Matching against your last analyzed resume." if use_own
            else "First analyze your resume in Resume Studio; currently the fictional sample resume will be used."
        )
        find_clicked = st.button("🔎 Find and rank remote roles", type="primary", use_container_width=True)

    st.markdown(
        '<div class="cm-info-banner">🔎 <b>Where these jobs come from:</b> '
        'Remote listings are retrieved from the Remotive public feed, not LinkedIn. '
        'Results can lag, and remote jobs may still have country restrictions. '
        'Confirm availability and visa/work-authorization requirements with the employer.</div>',
        unsafe_allow_html=True,
    )

    if find_clicked:
        try:
            with st.spinner("Finding remote openings and checking resume signals..."):
                all_jobs = cached_remote_jobs()
                candidates = filter_remote_jobs(all_jobs, search)
                resume_text = (st.session_state["last_analysis"]["resume"]
                               if use_own else sample_resume)
                scored = []
                for candidate in candidates:
                    description = clean_html(str(candidate.get("description") or ""))
                    if not description:
                        continue
                    title = str(candidate.get("title") or "Untitled role")
                    match = analyze(resume_text, title + " " + description)
                    scored.append({
                        "title": title,
                        "company": str(candidate.get("company_name") or "Company not provided"),
                        "location": str(candidate.get("candidate_required_location") or "Location varies"),
                        "date": str(candidate.get("publication_date") or "")[:10],
                        "url": str(candidate.get("url") or ""),
                        "match": match,
                        "rank": 0.65 * match["coverage"] + 0.35 * match["similarity"],
                    })
                scored.sort(key=lambda item: item["rank"], reverse=True)
                st.session_state["recommended_jobs"] = scored[:limit]
                st.session_state["remote_result_total"] = len(scored)
                st.session_state["remote_query"] = search
        except Exception as exc:
            st.error(f"Couldn't retrieve remote jobs: {exc}")
            st.info("You can still use the LinkedIn search above, or try again when you have internet access.")

    if "recommended_jobs" not in st.session_state:
        st.markdown(
            '<div class="cm-blank"><span class="emoji">💼</span>'
            '<strong>Your job discovery feed is ready</strong>'
            '<small>Enter a keyword and click Find and rank remote roles to load current listings.</small></div>',
            unsafe_allow_html=True,
        )
    else:
        items = st.session_state["recommended_jobs"]
        st.markdown(
            f'<div class="cm-section-top"><span class="cm-section-icon">⭐</span>'
            f'<span class="cm-section-title">Top role suggestions</span></div>'
            f'<div class="cm-section-subtitle">{st.session_state.get("remote_result_total", 0)} matching listings returned for '
            f'“{safe(st.session_state.get("remote_query", ""))}” • showing {len(items)}</div>',
            unsafe_allow_html=True,
        )
        if not items:
            st.warning("No matches found in the current remote feed. Try a broader keyword, such as 'data'.")
        for index, item in enumerate(items, 1):
            with st.container(border=True):
                m = item["match"]
                st.markdown(
                    f'<div class="cm-job-head"><div>'
                    f'<div class="cm-job-title">{index:02d}. {safe(item["title"])}</div>'
                    f'<div class="cm-job-company">🏢 {safe(item["company"])} &nbsp; · &nbsp; '
                    f'🌐 {safe(item["location"])} &nbsp; · &nbsp; '
                    f'📅 {safe(item["date"] or "Date not provided")}</div>'
                    f'</div><div class="cm-job-score">{m["coverage"]:.0%} skills found</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="cm-divider"></div>'
                    '<div class="cm-panel-title">Recognized overlapping skills</div>'
                    + tags(m["matched"])
                    + f'<div class="cm-note">Text similarity: {m["similarity"]:.1%} '
                    f' · Not detected: {safe(", ".join(m["missing"]) or "none")}</div>',
                    unsafe_allow_html=True,
                )
                actions = st.columns([1, 1, 2], gap="small")
                with actions[0]:
                    # Only point to a vetted provider URL; never follow unknown arbitrary domains.
                    if item["url"].startswith("https://remotive.com/"):
                        st.link_button("↗ View job", item["url"], use_container_width=True)
                with actions[1]:
                    job_key = item["url"] or item["title"] + item["company"]
                    already_saved = job_key in st.session_state["saved_jobs"]
                    if st.button("✓ Saved" if already_saved else "☆ Save role", key=f"save_role_{index}",
                                 disabled=already_saved, use_container_width=True):
                        st.session_state["saved_jobs"][job_key] = {
                            "Job": item["title"], "Company": item["company"],
                            "Location": item["location"], "Link": item["url"],
                        }
                        st.rerun()
                with actions[2]:
                    st.caption("Ranking is a text/keyword heuristic, not a hiring recommendation or guarantee.")

# ------------------------- TAB 3 -------------------------------
with insights_tab:
    section("📈", "Your Career Insights", "A clear overview of your resume signals and saved opportunities.")
    if "last_analysis" in st.session_state:
        saved = st.session_state["last_analysis"]
        result = saved["result"]
        render_stats(result)
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### 🧩 Skills recognized in your resume")
                st.markdown(tags(result["resume_skills"], "neutral"), unsafe_allow_html=True)
                st.caption("Skills detected through a limited keyword dictionary.")
        with right:
            with st.container(border=True):
                st.markdown("#### 🎯 Your development focus")
                st.markdown(tags(result["missing"], "missing"), unsafe_allow_html=True)
                st.caption("Missing keywords are suggestions to investigate, not validated skill gaps.")
        st.markdown("#### 📋 Compare your resume against sample roles")
        st.dataframe(ranking_table(saved["resume"]), hide_index=True, use_container_width=True)
    else:
        st.markdown(
            '<div class="cm-blank"><span class="emoji">📄</span>'
            '<strong>Start with a resume comparison</strong>'
            '<small>Your personal insights will appear here after you click Analyze my resume.</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### 🔖 Your saved remote jobs")
    saved_rows = list(st.session_state["saved_jobs"].values())
    if saved_rows:
        saved_frame = pd.DataFrame(saved_rows)
        st.dataframe(saved_frame, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇ Export saved jobs as CSV", data=saved_frame.to_csv(index=False).encode("utf-8"),
            file_name="my_saved_jobs.csv", mime="text/csv",
        )
        if st.button("Clear saved jobs", key="clear_saved_jobs"):
            st.session_state["saved_jobs"] = {}
            st.rerun()
        st.caption("Saved jobs remain in this browser session. Export the CSV to keep them.")
    else:
        st.markdown(
            '<div class="cm-summary-box"><h4>No saved roles yet</h4>'
            '<p>Use the ☆ Save role button under any Remotive job suggestion. '
            'Your saved roles will appear here.</p></div>',
            unsafe_allow_html=True,
        )
    with st.expander("🔬 How this project works (great for interviews)"):
        st.markdown(
            "**Step 1:** Pandas reads job descriptions from CSV.\n\n"
            "**Step 2:** Our regex dictionary detects certain skill names in text.\n\n"
            "**Step 3:** TF-IDF turns resume/job words into numerical features.\n\n"
            "**Step 4:** Cosine similarity compares the numerical feature vectors.\n\n"
            "**Step 5:** Streamlit displays results; Remotive supplies remote listings.\n\n"
            "**Limitation:** This is an interpretable text-matching tool, not a trained hiring-decision system."
        )

st.markdown(
    '<div class="cm-footer">CareerMatch AI Pro • Built with Python, pandas, scikit-learn & Streamlit'
    '<br>Transparent matching • Private local analysis • Not affiliated with LinkedIn or Remotive</div>',
    unsafe_allow_html=True,
)
