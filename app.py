
from html import escape

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from main import graph

app = FastAPI(title="LangGraph Multi-Agent News Analyzer")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Agent News Analyzer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 850px;
                margin: 40px auto;
                padding: 20px;
                background: #f4f7fb;
                color: #222;
            }
            .card {
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 3px 15px #0001;
            }
            input, button {
                box-sizing: border-box;
                width: 100%;
                padding: 13px;
                margin-top: 12px;
                font-size: 16px;
            }
            button {
                background: #2457c5;
                color: white;
                border: 0;
                border-radius: 6px;
                cursor: pointer;
            }
            .agents {
                line-height: 1.8;
                background: #eef3ff;
                padding: 15px;
                border-radius: 8px;
            }
            pre {
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>LangGraph Multi-Agent News Analyzer</h1>
            <p>Enter a topic to collect, analyze, and summarize news.</p>

            <div class="agents">
                <b>Agent 1:</b> News Collector<br>
                <b>Agent 2:</b> News Analyzer<br>
                <b>Agent 3:</b> Report Generator
            </div>

            <form action="/analyze" method="post">
                <input name="topic" placeholder="e.g. Artificial Intelligence"
                       maxlength="200" required>
                <button type="submit">Analyze News</button>
            </form>

            <p>Powered by Python and LangGraph</p>
        </div>
    </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
def analyze(topic: str = Form(...)):
    topic = topic.strip()

    if not topic or len(topic) > 200:
        return HTMLResponse(
            "<h2>Please enter a valid topic.</h2><a href='/'>Go back</a>",
            status_code=400
        )

    try:
        result = graph.invoke({
            "topic": topic,
            "articles": [],
            "analysis": "",
            "final_report": ""
        })

        report = escape(result.get("final_report", "No report generated."))

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>News Analysis Results</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 900px;
                    margin: 35px auto;
                    padding: 20px;
                    background: #f4f7fb;
                }}
                .card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                }}
                pre {{
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    line-height: 1.6;
                }}
                a {{ color: #2457c5; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>News Analysis Report</h1>
                <p><b>Topic:</b> {escape(topic)}</p>
                <hr>
                <pre>{report}</pre>
                <p><a href="/">Analyze another topic</a></p>
            </div>
        </body>
        </html>
        """

    except Exception:
        return HTMLResponse(
            "<h2>Analysis failed.</h2>"
            "<p>Check the server logs and Gemini quota, then try again.</p>"
            "<a href='/'>Go back</a>",
            status_code=500
        )
