import difflib
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def tool_read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return f"File contents of {path}:\n{content}"


def tool_write_file(path: str, content: str) -> str:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully created {path}"


def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if old_text not in content:
        return f"Error: old_text not found in {path}."

    proposed = content.replace(old_text, new_text, 1)

    diff_lines = list(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )

    if not diff_lines:
        return "No changes detected."

    print(f"\n{BOLD}Proposed changes to {path}:{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")
    for line in diff_lines:
        line = line.rstrip("\n")
        if line.startswith("---") or line.startswith("+++"):
            print(f"{BOLD}{CYAN}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"{YELLOW}{line}{RESET}")
        elif line.startswith("+"):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith("-"):
            print(f"{RED}{line}{RESET}")
        else:
            print(line)
    print(f"{DIM}{'─' * 60}{RESET}")

    answer = input(f"\n{BOLD}Apply these changes? (y/n): {RESET}").strip().lower()
    if answer in ("y", "yes"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(proposed)
        return f"Changes applied successfully to {path}."
    return "Changes rejected by user."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing old_text with new_text. Shows a colored diff and asks for approval. Use read_file first to see current content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "read_file": lambda args: tool_read_file(args["path"]),
    "write_file": lambda args: tool_write_file(args["path"], args["content"]),
    "edit_file": lambda args: tool_edit_file(args["path"], args["old_text"], args["new_text"]),
}

SYSTEM_PROMPT = (
    "You are mdCode, a helpful coding assistant running in the user's terminal. "
    "You can read, write, and edit files in the current working directory. "
    "When editing existing files, always use edit_file (not write_file) so the user "
    "can review a diff and approve changes before they are applied. "
    "Use read_file first to see the current content before editing. "
    "Be concise and direct. Do not use markdown formatting in your responses "
    "since they appear directly in the terminal."
)


def run_agent():
    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input(f"\n{BOLD}You:{RESET} ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )

            assistant_msg = response.choices[0].message

            if assistant_msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_msg.tool_calls
                        ],
                    }
                )

                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    print(f"\n{DIM}[tool: {tool_name}]{RESET}", flush=True)

                    result = TOOL_DISPATCH[tool_name](tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                continue
            text = assistant_msg.content or ""
            messages.append({"role": "assistant", "content": text})
            print(f"\n{BOLD}mdCode:{RESET} {text}")
            break


def main():
    print(f"\n{BOLD}{CYAN}mdCode{RESET} - AI Coding Assistant")
    print(f"Tools: read_file, write_file, edit_file (with diff view)")
    print(f"Type {BOLD}exit{RESET} to quit.\n")
    run_agent()


if __name__ == "__main__":
    main()
