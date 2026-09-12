# Introduction-page implementation

This document records how the introduction page is connected to the existing
Streamlit scheduling application.

## 1. Preserve the existing scheduling model

The former `streamlit_app.py` was renamed to `inspection_planner.py`. Its
calculation, inputs, charts, and results were left in place. Only its
`st.set_page_config(...)` call was removed because page configuration now belongs
to the application entry point.

## 2. Add a small navigation entry point

The new `streamlit_app.py` defines two Streamlit pages:

- `introduction.py`, which is the default page;
- `inspection_planner.py`, which contains the original scheduling tool.

Streamlit Community Cloud should continue to use `streamlit_app.py` as the main
file. No deployment setting needs to change.

## 3. Add the introduction layout

`introduction.py` creates the university-branded header, developer cards,
abstract, documentation, link to the scheduler, and research-use notice. Local
images are converted to data URIs so they render reliably on Streamlit Community
Cloud without an external image host.

## 4. Store editable copy separately

The page reads its text from:

- `content/ABSTRACT.md`;
- `content/DOCUMENTATION.md`.

To change the page in GitHub, open either file, select the pencil icon, edit the
Markdown, and commit the change. The Streamlit deployment will rebuild from the
new commit automatically.

## 5. Store images in the repository

Portraits and the official university logo are located in `assets/`. To replace
a portrait, upload another PNG with the same filename. Keeping the filename
unchanged avoids any Python modification.

## 6. Preview locally

From the repository directory:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Check both pages at desktop and narrow browser widths before publishing.
