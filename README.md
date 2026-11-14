# CareerMatch AI — Real Job Finder 

A beginner-friendly portfolio project made with Python, scikit-learn, pandas, and Streamlit.

## What it does

1. Analyze a resume against a sample or manually pasted job description.
2. Extract recognized skills and calculate TF-IDF text similarity.
3. Search **LinkedIn's real Jobs website** using a button (search happens on LinkedIn, not inside the app).
4. Show **real remote listings from Remotive's public API**, ranked against the uploaded resume or example resume.
5. Link directly back to Remotive for the original job/application page.
6. Download a resume-to-job analysis report.

**Important:** LinkedIn does not offer an open job-search API for this use case. Connecting a LinkedIn account with OpenID Connect can authenticate a person but does not grant access to LinkedIn job listings. The app never requests LinkedIn credentials. Do not use scraping or bots.

## Start in PyCharm (Windows)

1. Extract the ZIP and open the `CareerMatch_AI_JobFinder` folder in PyCharm.
2. Open Terminal at the **project root**, next to `app.py`.
3. Run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Open the local URL Streamlit prints (normally http://localhost:8501).
5. Analyze your own resume in the first section and click **Analyze match**.
6. Scroll to **Find real jobs**. Enter your title/location for LinkedIn and click **Open LinkedIn job search** to browse on LinkedIn.
7. Enter `data`, `analyst`, or `python` in **Search remote job titles** and click **Find & rank remote jobs** to see Remotive listings inside the app.

The Remotive feed is cached for 6 hours, and the service recommends requesting it no more than 4 times per day. New job listings may lag up to 24 hours. Location and work authorization are **not** verified. There is no API key required for the Remotive public feed.

## Project files

- `app.py`: the Streamlit interface and resume analysis functions
- `job_sources.py`: LinkedIn search URL and Remotive feed helpers
- `data/jobs.csv`: fictional sample jobs for practice
- `data/sample_resume.txt`: fictional resume for practice
- `.streamlit/config.toml`: styling
- `tests/test_job_sources.py`: basic tests

## Privacy

Your uploaded resume stays in your local Streamlit session for matching and is **not** sent to LinkedIn or Remotive. The remote jobs endpoint receives only a request for its public feed. Don't publish personal resumes, API tokens or credentials to GitHub. 

## Attribution

Remote job feed by [Remotive](https://remotive.com/remote-jobs/api). Listings link directly to the original Remotive page as required by its terms. No LinkedIn data is scraped or downloaded.
