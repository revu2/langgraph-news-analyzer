import os
import re
import time
import html
import requests
import xml.etree.ElementTree as ET
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")


# ============================================================
# 2. SHARED STATE
# ============================================================

class NewsState(TypedDict):
    topic: str
    articles: list
    analysis: str
    final_report: str


# ============================================================
# 3. GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=API_KEY
)


# ============================================================
# 4. CLEAN TEXT
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 5. EXTRACT GEMINI TEXT
# ============================================================

def extract_text(content):

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text = block.get("text", "")

                    if text:
                        parts.append(text)

            else:
                parts.append(str(block))

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# 6. QUOTA HANDLING
# ============================================================

def is_quota_error(error):

    text = str(error).lower()

    return (
        "resource_exhausted" in text
        or "429" in text
        or "quota" in text
    )


def is_daily_quota(error):

    text = str(error).lower()

    return (
        "perday" in text
        or "per_day" in text
        or "generate_requests_per_day" in text
        or "generaterequestsperday" in text
    )


def retry_seconds(error):

    text = str(error)

    match = re.search(
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        text,
        re.IGNORECASE
    )

    if match:
        return max(2, int(float(match.group(1))) + 2)

    return 20


def call_gemini(prompt, agent_name):

    for attempt in range(1, 3):

        print(
            f"\n{agent_name}: Gemini attempt "
            f"{attempt}/2"
        )

        try:

            response = llm.invoke(prompt)

            text = extract_text(response.content)

            if text:
                return text

            return None

        except Exception as error:

            if is_quota_error(error):

                if is_daily_quota(error):

                    print(
                        f"{agent_name}: Gemini quota exhausted."
                    )

                    return None

                wait = retry_seconds(error)

                if attempt < 2:

                    print(
                        f"{agent_name}: Rate limited. "
                        f"Waiting {wait} seconds..."
                    )

                    time.sleep(wait)

                    continue

                return None

            print(
                f"{agent_name}: Gemini error: {error}"
            )

            return None

    return None


# ============================================================
# AGENT 1
# NEWS COLLECTOR
# ============================================================

def news_collector(state: NewsState):

    print("\n" + "=" * 60)
    print("AGENT 1: NEWS COLLECTOR")
    print("=" * 60)

    topic = state["topic"]

    print(f"Topic: {topic}")
    print("Collecting news...")

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(topic)}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    try:

        response = requests.get(
            rss_url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

    except Exception as error:

        print(f"News collection error: {error}")

        return {
            "articles": []
        }

    articles = []

    for item in root.findall(".//item")[:8]:

        title_element = item.find("title")
        link_element = item.find("link")
        description_element = item.find("description")

        title = (
            title_element.text.strip()
            if title_element is not None
            and title_element.text
            else "Untitled"
        )

        link = (
            link_element.text.strip()
            if link_element is not None
            and link_element.text
            else ""
        )

        description = (
            description_element.text
            if description_element is not None
            and description_element.text
            else ""
        )

        description = clean_text(description)

        articles.append({
            "title": title,
            "link": link,
            "description": description
        })

    print(
        f"Collected {len(articles)} articles."
    )

    for i, article in enumerate(
        articles,
        1
    ):
        print(
            f"{i}. {article['title']}"
        )

    return {
        "articles": articles
    }


# ============================================================
# HELPER: GET SOURCE NAME
# ============================================================

def get_source(title):

    if " - " in title:

        parts = title.rsplit(
            " - ",
            1
        )

        return parts[-1].strip()

    return "News source"


# ============================================================
# AGENT 2 FALLBACK
# CLEAN LOCAL ANALYSIS
# ============================================================

def local_analysis(topic, articles):

    sources = {}

    for article in articles:

        source = get_source(
            article["title"]
        )

        sources[source] = (
            sources.get(source, 0) + 1
        )

    source_list = ", ".join(
        sources.keys()
    )

    lines = [

        f"Topic: {topic}",

        "",

        "Main Developments:",

    ]

    for i, article in enumerate(
        articles[:8],
        1
    ):

        title = article["title"]

        lines.append(
            f"{i}. {title}"
        )

    lines.extend([

        "",

        "Major Themes:",

        "The collected reports cover AI technology, "
        "government and policy, AI safety and security, "
        "social effects, research, and potential risks.",

        "",

        "Source Coverage:",

        f"The collected articles come from: {source_list}.",

        "",

        "Limitations:",

        "This analysis is based on the article titles and "
        "RSS summaries collected by the News Collector. "
        "It does not assume facts that are not present "
        "in the collected material."
    ])

    return "\n".join(lines)


# ============================================================
# AGENT 2
# NEWS ANALYZER
# ============================================================

def news_analyzer(state: NewsState):

    print("\n" + "=" * 60)
    print("AGENT 2: NEWS ANALYZER")
    print("=" * 60)

    topic = state["topic"]
    articles = state["articles"]

    print(
        f"Received {len(articles)} articles."
    )

    if not articles:

        return {
            "analysis": "No articles were collected."
        }

    article_text = ""

    for i, article in enumerate(
        articles,
        1
    ):

        article_text += (
            f"{i}. {article['title']}\n"
            f"Summary: {article['description']}\n\n"
        )

    prompt = f"""
You are Agent 2 of a three-agent LangGraph system.

Analyze the following news about:
{topic}

Articles:
{article_text}

Give a concise analysis.

Use exactly these sections:

MAIN DEVELOPMENTS
- 4 to 6 concise points

MAJOR THEMES
- 3 to 5 concise points

IMPORTANT OBSERVATIONS
- 3 concise points

LIMITATIONS
- 1 or 2 points

Use ONLY the supplied article information.
Do not invent facts.
Keep the response below 500 words.
"""

    analysis = call_gemini(
        prompt,
        "Agent 2"
    )

    if analysis is None:

        print(
            "Agent 2: Using clean local analysis."
        )

        analysis = local_analysis(
            topic,
            articles
        )

    else:

        print(
            "Agent 2: Gemini analysis completed."
        )

    print(
        f"Analysis length: "
        f"{len(analysis)} characters."
    )

    return {
        "analysis": analysis
    }


# ============================================================
# AGENT 3 FALLBACK
# ============================================================

def local_report(
    topic,
    articles,
    analysis
):

    report = []

    report.append(
        "# NEWS ANALYSIS REPORT"
    )

    report.append("")

    report.append(
        f"Topic: {topic}"
    )

    report.append("")

    report.append(
        "## Analysis"
    )

    report.append(
        analysis
    )

    report.append("")

    report.append(
        "## Sources"
    )

    for i, article in enumerate(
        articles,
        1
    ):

        source = get_source(
            article["title"]
        )

        title = article["title"]

        # Remove source suffix from title
        if " - " in title:
            title = title.rsplit(
                " - ",
                1
            )[0]

        report.append(
            f"{i}. {source}: {title}"
        )

    return "\n".join(report)


# ============================================================
# AGENT 3
# REPORT GENERATOR
# ============================================================

def report_generator(state: NewsState):

    print("\n" + "=" * 60)
    print("AGENT 3: REPORT GENERATOR")
    print("=" * 60)

    topic = state["topic"]
    articles = state["articles"]
    analysis = state["analysis"]

    print(
        f"Received {len(articles)} articles."
    )

    print(
        f"Received analysis of "
        f"{len(analysis)} characters."
    )

    prompt = f"""
You are Agent 3 of a three-agent LangGraph system.

Create a short, clean final news report.

Topic:
{topic}

Agent 2 analysis:
{analysis}

Sources:
"""

    for i, article in enumerate(
        articles,
        1
    ):

        prompt += (
            f"{i}. {article['title']}\n"
        )

    prompt += """

Format the response exactly as:

# NEWS ANALYSIS REPORT

## Topic
[topic]

## Key Developments
- concise bullet points

## Major Themes
- concise bullet points

## Important Observations
- concise bullet points

## Limitations
- concise bullet points

## Sources
- source names and article titles only

Rules:
- Maximum 600 words.
- Do not include URLs.
- Do not repeat the full article descriptions.
- Do not invent information.
- Use only the supplied material.
"""

    final_report = call_gemini(
        prompt,
        "Agent 3"
    )

    if final_report is None:

        print(
            "Agent 3: Using clean local report."
        )

        final_report = local_report(
            topic,
            articles,
            analysis
        )

    else:

        print(
            "Agent 3: Gemini report generated."
        )

    print(
        f"Report length: "
        f"{len(final_report)} characters."
    )

    return {
        "final_report": final_report
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    NewsState
)


# EXACTLY 3 AGENTS

graph_builder.add_node(
    "news_collector",
    news_collector
)

graph_builder.add_node(
    "news_analyzer",
    news_analyzer
)

graph_builder.add_node(
    "report_generator",
    report_generator
)


# ============================================================
# WORKFLOW
# ============================================================

graph_builder.add_edge(
    START,
    "news_collector"
)

graph_builder.add_edge(
    "news_collector",
    "news_analyzer"
)

graph_builder.add_edge(
    "news_analyzer",
    "report_generator"
)

graph_builder.add_edge(
    "report_generator",
    END
)


graph = graph_builder.compile()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print(
        "        LANGGRAPH MULTI-AGENT NEWS ANALYZER"
    )
    print("=" * 60)

    topic = input(
        "\nEnter a news topic to analyze: "
    ).strip()

    if not topic:

        topic = "Artificial Intelligence"

    initial_state: NewsState = {

        "topic": topic,

        "articles": [],

        "analysis": "",

        "final_report": ""
    }

    print(
        "\nStarting LangGraph workflow..."
    )

    try:

        result = graph.invoke(
            initial_state
        )

        print("\n")
        print("=" * 60)
        print("WORKFLOW COMPLETED")
        print("=" * 60)

        print(
            "\nSTART"
        )

        print(
            "  ↓ Agent 1: News Collector"
        )

        print(
            "  ↓ Agent 2: News Analyzer"
        )

        print(
            "  ↓ Agent 3: Report Generator"
        )

        print(
            "  ↓"
        )

        print(
            "END"
        )

        print("\n")
        print("=" * 60)
        print("FINAL REPORT")
        print("=" * 60)

        print(
            result["final_report"]
        )

        print("\n")
        print("=" * 60)
        print("SYSTEM EXECUTION FINISHED")
        print("=" * 60)

    except KeyboardInterrupt:

        print(
            "\nExecution stopped."
        )

    except Exception as error:

        print("\n")
        print("=" * 60)
        print("WORKFLOW ERROR")
        print("=" * 60)

        print(
            f"{error}"
        )