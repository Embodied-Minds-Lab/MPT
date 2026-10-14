from __future__ import annotations

# OpenResearcher's agent-facing strings, PINNED-COPIED from the vendored repo.
#
# Copied rather than imported because their modules cannot be imported here:
# data_utils.py imports torch at module scope, eval.py imports prettytable and
# runs dotenv.load_dotenv() at import time, and neither module is part of what
# their pyproject installs (packages=["utils","tevatron"]).

# Their system prompt (third_party/benchmarks/OpenResearcher/data_utils.py:8-19,
# DEVELOPER_CONTENT). The {cursor}/{line_start}/{line_end} braces are LITERAL —
# the string is never .format()ed. "browsering" is their typo, kept.
DEVELOPER_CONTENT = """
You are a helpful assistant and harmless assistant.

You will be able to use a set of browsering tools to answer user queries.

Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
Do not quote more than 10 words directly from the tool output.
sources=web
""".strip()

# Their tool schemas (third_party/benchmarks/OpenResearcher/data_utils.py:21-117,
# TOOL_CONTENT), the JSON their chat template served to the model. tools.py
# derives the smolagents input schemas FROM this string, so descriptions and
# defaults cannot drift from what their agents saw. Two knowingly-false
# advertisements are theirs, kept: `topn` claims default 10 but the runtime
# always requests 20 (gpt-oss SimpleBrowserTool discards the argument), and
# `search.query` is typed string although their backend also accepts a list.
TOOL_CONTENT = """
[
  {
    "type": "function",
    "function": {
      "name": "browser.search",
      "description": "Searches for information related to a query and displays top N results. Returns a list of search results with titles, URLs, and summaries.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query string"
          },
          "topn": {
            "type": "integer",
            "description": "Number of results to display",
            "default": 10
          }
        },
        "required": [
          "query"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "browser.open",
      "description": "Opens a link from the current page or a fully qualified URL. Can scroll to a specific location and display a specific number of lines. Valid link ids are displayed with the formatting: 【{id}†.*】.",
      "parameters": {
        "type": "object",
        "properties": {
          "id": {
            "type": [
              "integer",
              "string"
            ],
            "description": "Link id from current page (integer) or fully qualified URL (string). Default is -1 (most recent page)",
            "default": -1
          },
          "cursor": {
            "type": "integer",
            "description": "Page cursor to operate on. If not provided, the most recent page is implied",
            "default": -1
          },
          "loc": {
            "type": "integer",
            "description": "Starting line number. If not provided, viewport will be positioned at the beginning or centered on relevant passage",
            "default": -1
          },
          "num_lines": {
            "type": "integer",
            "description": "Number of lines to display",
            "default": -1
          },
          "view_source": {
            "type": "boolean",
            "description": "Whether to view page source",
            "default": false
          },
          "source": {
            "type": "string",
            "description": "The source identifier (e.g., 'web')"
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "browser.find",
      "description": "Finds exact matches of a pattern in the current page or a specified page by cursor.",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "The exact text pattern to search for"
          },
          "cursor": {
            "type": "integer",
            "description": "Page cursor to search in. If not provided, searches in the current page",
            "default": -1
          }
        },
        "required": [
          "pattern"
        ]
      }
    }
  }
]
""".strip()

# Their judge prompt (third_party/benchmarks/OpenResearcher/eval.py:14-33,
# GRADER_TEMPLATE — the HLE grader). Kept byte-for-byte, "if there if there"
# typo included: this prompt decides `correct`, our headline metric.
# NOT browsecomp_plus's grader: that one adds a semantic-equivalence paragraph
# and repeats [correct_answer]; this benchmark is judged the way OpenResearcher
# judges. The parsing side deliberately uses OUR line-anchored parser instead
# of their unanchored regex (which also matches inside "incorrect:") — same
# amendment policy as FATE's sorry-check.
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def build_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(question=question, response=response,
                                  correct_answer=correct_answer)
